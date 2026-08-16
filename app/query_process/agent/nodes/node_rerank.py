import sys
from app.utils.task_utils import *
from dotenv import load_dotenv
import os
import sys
from pathlib import Path
from app.llm.reranker_utils import get_reranker_model
from app.conf.reranker_config import reranker_config
from app.llm.text_compress_utils import compress_text
from app.utils.task_utils import add_running_task
from app.core.logger import logger, step_log
load_dotenv()

#全局设置参数
RERANK_MAX_TOPK:int = 10
RERANK_MIN_TOPK:int = 1
# 断崖阈值（相对）
RERANK_GAP_RATIO:float = 2
# 断崖阈值（绝对）
RERANK_GAP_ABS:float = 2
# 过滤重排后几乎完全不相关的尾部结果，避免 0 分切片进入答案和引用来源。
RERANK_MIN_SCORE: float = float(os.getenv("RERANK_MIN_SCORE", "0.15"))

@step_log("step_1_data_validates")
def step_1_data_validates(state):
    """
    获取rrf粗排结果以及mcp搜索结果
    """
    rrf_chunks = state.get("rrf_chunks", [])
    web_search_docs = state.get("web_search_docs", [])
    return rrf_chunks, web_search_docs

def step_2_merged_rrf_and_mcp(rrf_chunks, web_search_docs):
    """
    两路（rrf和mcp）格式统一
    """
    final_list = []
    if rrf_chunks and len(rrf_chunks) > 0:
        """
        如果它是 None，判定为 False。
        如果它是空列表 []，判定为 False。
        只有当它是有数据的列表时，才会被判定为 True。
        """
        for chunk in rrf_chunks:
            final_list.append(
               {
                "title":chunk.get("title"),
                "text":chunk.get("content"),
                "url":None,
                "type":"milvus",
                "score":0.0,
                "file_title": chunk.get("file_title"),
                "parent_title": chunk.get("parent_title"),
                "part": chunk.get("part"),
                "kb_id": chunk.get("kb_id", ""),
                "document_id": chunk.get("document_id", ""),
               }
            )

    if web_search_docs and len(web_search_docs) > 0:
        for search_rank, doc in enumerate(web_search_docs, start=1):
            image_urls = doc.get("image_urls") or doc.get("images") or []
            if isinstance(image_urls, str):
                image_urls = [image_urls]
            single_image = doc.get("image_url") or doc.get("image")
            if single_image and single_image not in image_urls:
                image_urls.append(single_image)
            final_list.append(
               {
                "title":doc.get("title"),
                "text":doc.get("content") or doc.get("snippet") or "",
                "url":doc.get("url"),
                "type":"web",
                "score":doc.get("score"),
                "search_rank":search_rank,
                "image_urls":image_urls,
                "file_title": doc.get("site_name") or doc.get("title"),
                "parent_title": doc.get("title"),
                "part": None,
                "kb_id": None,
                "document_id": None,
               }
            )
    return final_list
def step_3_rerank_score_and_sort(state, final_chunk_list):
    """
    rerank打分排序
    """
    #获取重写问题
    rewritten_query = state.get("rewritten_query") or state.get("original_query")
    text_list = [item.get("text") for item in final_chunk_list]
    question_pairs = []
    for text in text_list:
        # 正文超长先压缩，避免 (问题+正文) 超过 reranker 模型输入上限
        compressed_text = compress_text(text)
        # 组装（问题，答案）对的集合
        question_pairs.append((rewritten_query, compressed_text))

    local_path = reranker_config.bge_reranker_large.strip()
    local_dir = Path(local_path) if local_path else None
    local_ready = bool(
        local_dir
        and local_dir.is_dir()
        and any((local_dir / name).is_file() for name in ("model.safetensors", "pytorch_model.bin"))
    )
    if not local_ready and all(item.get("type") == "web" for item in final_chunk_list):
        logger.warning("Reranker 尚未就绪，联网结果按搜索引擎原顺序继续处理")
        return final_chunk_list

    reranker = get_reranker_model()
    score_list = reranker.compute_score(question_pairs,normalize=True)

    for score,chunk in zip(score_list,final_chunk_list): # type: ignore
        chunk['score']  = round(score, 4)

    #基于分数进行集合数据排序
    final_chunk_list.sort(key=lambda x:x.get("score",0.0),reverse=True)
    # 返回数据
    return final_chunk_list

def step_4_chunk_topk(chunk_list_score_sorted):
    """
    断崖式截断
    """
    relevant_chunks = [
        chunk for chunk in chunk_list_score_sorted
        if float(chunk.get("score", 0.0)) >= RERANK_MIN_SCORE
    ]
    if not relevant_chunks:
        return []
    chunk_list_score_sorted = relevant_chunks
    min_topk = RERANK_MIN_TOPK
    max_topk = RERANK_MAX_TOPK
    gap_ratio = RERANK_GAP_RATIO
    max_gap = RERANK_GAP_ABS
    #最多多少个
    max_topk =  min(max_topk,len(chunk_list_score_sorted))
    topk = max_topk
    if topk > min_topk:
        for index in range(min_topk-1,max_topk-1):
             score_1 = chunk_list_score_sorted[index].get("score",0.0)
             score_2 = chunk_list_score_sorted[index+1].get("score",0.0)
             #分差
             abs_score = score_1 - score_2
            #比率
             ratio_score = abs_score / (score_1 + 1e-7)
             if abs_score > max_gap or ratio_score > gap_ratio:
                 #产生断崖了
                 topk = index + 1#第n个可以获取
                 break
    final_chunk_list = chunk_list_score_sorted[:topk]
    return final_chunk_list

def node_rerank(state):
    """
    节点功能：对 RRF 后的结果进行精确打分重排。
    """
    run_id = state.get("run_id") or state["session_id"]
    add_running_task(run_id, sys._getframe().f_code.co_name, state.get("is_stream"))
    rrf_chunks, web_search_docs = step_1_data_validates(state)
    final_chunk_list = step_2_merged_rrf_and_mcp(rrf_chunks,web_search_docs)
    if not final_chunk_list:
        state["reranked_docs"] = []
        state["answer"] = "没有检索到足够的相关资料，暂时无法基于知识库回答这个问题。"
        add_done_task(run_id, sys._getframe().f_code.co_name, state.get("is_stream"))
        return state
    final_chunk_list_score_sorted = step_3_rerank_score_and_sort(state,final_chunk_list)
    if any(chunk.get("score") is None for chunk in final_chunk_list_score_sorted):
        final_chunk_list_score_sorted_topk = final_chunk_list_score_sorted[:RERANK_MAX_TOPK]
    else:
        final_chunk_list_score_sorted_topk = step_4_chunk_topk(final_chunk_list_score_sorted)
    state["reranked_docs"] = final_chunk_list_score_sorted_topk
    if not final_chunk_list_score_sorted_topk:
        state["answer"] = "当前资料中没有找到与问题足够相关的信息，无法可靠作答。"
    add_done_task(run_id, sys._getframe().f_code.co_name, state.get("is_stream"))
    return state
