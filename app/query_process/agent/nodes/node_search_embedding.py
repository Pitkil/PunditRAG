import json
import sys

from app.clients.milvus_utils import create_hybrid_search_requests, get_milvus_client, hybrid_search
from app.conf.milvus_config import milvus_config
from app.core.logger import logger, node_log, step_log
from app.llm.embedding_utils import generate_embeddings
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
def step_3_milvus_hybrid_search(dense_vector, sparse_vector, item_names, kb_ids=None):
    """
     混合搜索步骤:
        1. 创建对应AnnSearchRequest
        2. 定义对应reranker
        3. 调用混合检索方法就行
    """
    milvus_client = get_milvus_client()
    if not milvus_client:
        raise ValueError("无法连接到 Milvus 数据库")
    #封装请求对象
    filters = []
    if item_names:
        filters.append(f"item_name in {json.dumps(item_names, ensure_ascii=False)}")
    if kb_ids:
        filters.append(f"kb_id in {json.dumps(kb_ids, ensure_ascii=False)}")
    expr_str = " and ".join(filters) or None
    reqs = create_hybrid_search_requests(dense_vector, sparse_vector, expr=expr_str, limit=5)
    # 调用混合检索
    results = hybrid_search(
        client = milvus_client,
        collection_name = milvus_config.chunks_collection,
        reqs = reqs,
        norm_score = True,
        limit = 5,
        output_fields=["chunk_id","item_name","content","title","parent_title","part","file_title","kb_id","document_id"]
    )

    if not results:
        return []
    return list(results[0])

@node_log("node_search_embedding")
def node_search_embedding(state):
    """
     节点功能：进行向量内容检索
    """
    run_id = state.get("run_id") or state["session_id"]
    add_running_task(run_id, sys._getframe().f_code.co_name, state.get("is_stream"))
    item_names, _ = step_1_data_validates(state)
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
        dense_vector, sparse_vector, item_names, state.get("kb_ids", [])
    )
    #返回结果即可
    add_done_task(run_id, sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"embedding_chunks": milvus_result}
