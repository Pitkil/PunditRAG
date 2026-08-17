"""负责同源数据的RRF融合排序"""
import sys
from typing import List, Dict, Any
from app.conf.embedding_config import embedding_config
from app.conf.retrieval_config import retrieval_config
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger, node_log

@node_log("step_1_data_validates")
def step_1_data_validates(state):
    """
    获取参数并进行校验
    """
    embedding_chunks = state.get("embedding_chunks", [])
    hyde_embedding_chunks = state.get("hyde_embedding_chunks", [])
    return embedding_chunks, hyde_embedding_chunks

@node_log("step_2_rrf_list")
def step_2_rrf_list(param_list,k:int = 60,top:int = 5):
    score_dict = {}#存储chunk_id 和对应的分数
    entity_dict = {}#存储chunk_id 和对应的实体信息
    #循环路，有两路
    #[(embedding_chunks, 分路权重), (hyde_embedding_chunks, 分路权重)]
    for chunks_list,weight in param_list:
        """
        {
            {
        "id": 1,
        "distance": 0.85,       # 分数
        "entity": {
            "chunk_id": 123,
            "item_name": "关键词",
            "content": "正文...",
            "title": "标题",
            "parent_title": "父标题",
            "part": 1,
            "file_title": "源文件.md",
            }
        }
        """
        #循环某一路的召回结果
        for rank,chunk in enumerate(chunks_list,start=1):
             chunk_id = chunk.get("id") or chunk['entity']['chunk_id']
             #多路都有的话，叠加分数
             score_dict[chunk_id] = score_dict.get(chunk_id,0.0) + (1.0 / (k + rank))*weight
             ## 如果没有值,才赋值! 第一次已经赋值了,后面就不会更新了
             entity_dict.setdefault(chunk_id,chunk.get("entity",{}))

    entity_list = []
    for chunk_id,score in score_dict.items():
        entity_list.append(
            (
                entity_dict.get(chunk_id,{}),
                score,
            )
        )
    #排序
    entity_list.sort(key = lambda x :x[1] ,reverse=True)
    final_entity_list = [entity for entity,score in entity_list[:top]]
    return final_entity_list

@node_log("node_rrf")
def node_rrf(state):
    """
    节点功能：Reciprocal Rank Fusion
    将多路召回的结果进行加权融合排序。
    """
    run_id = state.get("run_id") or state["session_id"]
    add_running_task(run_id, sys._getframe().f_code.co_name, state.get("is_stream", False))
    embedding_chunks, hyde_embedding_chunks = step_1_data_validates(state)
    #两路贡献相等
    param_list = [
        (embedding_chunks,1.0),
        (hyde_embedding_chunks,1.0)
    ]
    entity_list = step_2_rrf_list(param_list, k=60, top=retrieval_config.rrf_top_k)
    state['rrf_chunks'] = entity_list
    add_done_task(run_id, sys._getframe().f_code.co_name, state.get("is_stream", False))
    return state
