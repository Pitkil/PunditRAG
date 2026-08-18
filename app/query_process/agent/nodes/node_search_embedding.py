import sys

from app.clients.milvus_utils import get_milvus_client
from app.core.logger import logger, node_log, step_log
from app.llm.embedding_utils import generate_embeddings
from app.query_process.agent.retrieval_utils import merge_unique_hits, search_chunks
from app.utils.task_utils import  add_done_task,add_running_task

@step_log("step_1_data_validates")
def step_1_data_validates(state):
    """校验"""
    item_names = state.get("item_names") or []
    rewritten_query = state.get("rewritten_query")
    if not rewritten_query:
        logger.error("rewritten_query 不能为空")
        raise ValueError("rewritten_query 不能为空")
    return item_names, rewritten_query

@step_log("step_2_rewritten_query_embedding")
def step_2_rewritten_query_embedding(state):
    result = generate_embeddings([state["rewritten_query"]])
    return result['dense'][0],result['sparse'][0]

@step_log("step_3_milvus_hybrid_search")
def step_3_milvus_hybrid_search(dense_vector, sparse_vector, item_names, kb_ids=None, document_ids=None):
    """
     混合搜索步骤:
        1. 创建对应AnnSearchRequest
        2. 定义对应reranker
        3. 调用混合检索方法就行
    """
    milvus_client = get_milvus_client()
    if not milvus_client:
        raise ValueError("无法连接到 Milvus 数据库")
    return search_chunks(
        milvus_client, dense_vector, sparse_vector, item_names, kb_ids or [], document_ids or []
    )

@node_log("node_search_embedding")
def node_search_embedding(state):
    """
     节点功能：进行向量内容检索
    """
    run_id = state.get("run_id") or state["session_id"]
    add_running_task(run_id, sys._getframe().f_code.co_name, state.get("is_stream"))
    item_names, _ = step_1_data_validates(state)
    if not state.get("kb_ids") and not state.get("document_ids"):
        logger.info("未选择资料范围，跳过本地向量检索")
        add_done_task(run_id, sys._getframe().f_code.co_name, state.get("is_stream"))
        return {"embedding_chunks": []}
    # 问题向量化
    try:
        dense_vector,sparse_vector = step_2_rewritten_query_embedding(state)
    except RuntimeError as exc:
        if "本地模型尚未下载完整" not in str(exc):
            raise
        logger.warning("BGE-M3 尚未就绪，跳过本地向量检索，保留其他检索分支")
        add_done_task(run_id, sys._getframe().f_code.co_name, state.get("is_stream"))
        return {"embedding_chunks": []}
    #混合检索
    milvus_result = step_3_milvus_hybrid_search(
        dense_vector, sparse_vector, item_names, state.get("kb_ids", []), state.get("document_ids", [])
    )
    if state.get("query_depth") == "deep" and state.get("query_aspects"):
        facet_queries = [
            f"{state['rewritten_query']}；重点检索：{aspect}"
            for aspect in state["query_aspects"][:6]
        ]
        facet_embeddings = generate_embeddings(facet_queries)
        facet_results = []
        for index in range(len(facet_queries)):
            facet_results.append(
                step_3_milvus_hybrid_search(
                    facet_embeddings["dense"][index],
                    facet_embeddings["sparse"][index],
                    item_names,
                    state.get("kb_ids", []),
                    state.get("document_ids", []),
                )
            )
        milvus_result = merge_unique_hits(milvus_result, *facet_results)
    #返回结果即可
    add_done_task(run_id, sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"embedding_chunks": milvus_result}
