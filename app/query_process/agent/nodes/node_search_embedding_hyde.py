import sys
import os
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from app.utils.task_utils import add_running_task, add_done_task
from app.llm.llm_util import *
from app.llm.embedding_utils import *
from app.clients.milvus_utils import *
from app.core.logger import logger, node_log, step_log
from app.core.load_prompt import load_prompt
from app.query_process.agent.retrieval_utils import build_retrieval_query, search_chunks
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

HYDE_TIMEOUT_SECONDS = float(os.getenv("HYDE_TIMEOUT_SECONDS", "10"))

@step_log("step_1_data_validates")
def step_1_data_validates(state):
    """
    获取参数并校验
    """
    item_names = state.get("item_names") or []
    original_query = state.get("original_query")
    if not original_query:
        logger.error("original_query 不能为空")
        raise ValueError("original_query 不能为空")
    return item_names, original_query

@step_log("step_2_call_llm")
def step_2_call_llm(rewritten_query):
    """
     调用llm模型,给出普通回答
    """
    try:
        llm_client = get_llm_client(
            timeout=HYDE_TIMEOUT_SECONDS,
            max_retries=0,
        )
        prompt = load_prompt("hyde_prompt", rewritten_query=rewritten_query)
        messages = [HumanMessage(content=prompt)]
        llm_chains = llm_client | StrOutputParser()
        return llm_chains.invoke(messages)
    except Exception as exc:
        logger.warning(
            f"HyDE 模型调用超时或失败，跳过 HyDE 召回并保留普通检索：{exc}"
        )
        return ""

@step_log("step_3_rewritten_hyde_vector")
def step_3_rewritten_hyde_vector(rewritten_query, hyde_answer):
    """
     将问题和hyde回答拼接并进行向量化
    """
    vector_str = rewritten_query + " ," + hyde_answer
    result = generate_embeddings([vector_str])
    return result['dense'][0],result['sparse'][0]

@step_log("step_4_mivlus_hybrid_search")
def step_4_mivlus_hybrid_search(dense_vector, sparse_vector, item_names, kb_ids=None, document_ids=None):
    """
     混合搜索步骤:
        1. 创建对应AnnSearchRequest
        2. 定义对应reranker
        3. 调用混合检索方法
    """
    milvus_client = get_milvus_client()
    if not milvus_client:
        raise ValueError("无法连接到 Milvus 数据库")
    return search_chunks(
        milvus_client, dense_vector, sparse_vector, item_names, kb_ids or [], document_ids or []
    )

@node_log(node_name="node_search_embedding_hyde")
def node_search_embedding_hyde(state):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """
    run_id = state.get("run_id") or state["session_id"]
    add_running_task(run_id, sys._getframe().f_code.co_name, state.get("is_stream"))
    item_names, _ = step_1_data_validates(state)
    rewritten_query = build_retrieval_query(state)
    if not state.get("kb_ids") and not state.get("document_ids"):
        logger.info("未选择资料范围，跳过 HyDE 向量检索")
        add_done_task(run_id, sys._getframe().f_code.co_name, state.get("is_stream"))
        return {"hyde_embedding_chunks": []}
    hyde_answer = step_2_call_llm(rewritten_query)
    if not hyde_answer:
        add_done_task(run_id, sys._getframe().f_code.co_name, state.get("is_stream"))
        return {"hyde_embedding_chunks": []}
    try:
        dense_vector,sparse_vector = step_3_rewritten_hyde_vector(rewritten_query, hyde_answer)
    except RuntimeError as exc:
        if "本地模型尚未下载完整" not in str(exc):
            raise
        logger.warning("BGE-M3 尚未就绪，跳过 HyDE 向量检索，保留其他检索分支")
        add_done_task(run_id, sys._getframe().f_code.co_name, state.get("is_stream"))
        return {"hyde_embedding_chunks": []}
    milvus_result = step_4_mivlus_hybrid_search(
        dense_vector, sparse_vector, item_names, state.get("kb_ids", []), state.get("document_ids", [])
    )
    add_done_task(run_id, sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"hyde_embedding_chunks": milvus_result}
