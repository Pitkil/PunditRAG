import sys
from app.utils.task_utils import *
from dotenv import load_dotenv
import os
import sys
from pathlib import Path
from app.llm.reranker_utils import get_reranker_model
from app.conf.reranker_config import reranker_config
from app.conf.retrieval_config import retrieval_config
from app.utils.task_utils import add_running_task
from app.core.logger import logger, step_log
from app.query_process.agent.source_utils import deduplicate_documents
from app.clients.milvus_utils import get_milvus_client
from app.query_process.agent.retrieval_utils import expand_reranked_neighbors
load_dotenv()

RERANK_MAX_TOPK = retrieval_config.rerank_max_top_k
RERANK_INPUT_TOPK = retrieval_config.rerank_input_top_k
RERANK_MIN_TOPK = retrieval_config.rerank_min_top_k
RERANK_FALLBACK_TOPK = retrieval_config.rerank_fallback_top_k
RERANK_GAP_RATIO = retrieval_config.rerank_gap_ratio
RERANK_GAP_ABS = retrieval_config.rerank_gap_abs
RERANK_MIN_SCORE = retrieval_config.rerank_min_score

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
                "score":float(doc.get("score") or 0.0),
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


def step_2_limit_rerank_candidates(final_chunk_list):
    """限制本地重排输入，同时保留已启用的联网候选。"""
    local_docs = [item for item in final_chunk_list if item.get("type") != "web"][:RERANK_INPUT_TOPK]
    web_docs = [item for item in final_chunk_list if item.get("type") == "web"]
    return local_docs + web_docs
def step_3_rerank_score_and_sort(state, final_chunk_list):
    """
    rerank打分排序
    """
    original_query = state.get("original_query")
    question_pairs = []
    for item in final_chunk_list:
        # 文档名和章节名属于候选自身的元数据，对每个切片一致提供；
        # 用户原问题仍保持不变，避免标题被拼进问题后只抬高标题页。
        candidate_text = "\n".join(
            value
            for value in (
                f"文档：{item.get('file_title')}" if item.get("file_title") else "",
                f"章节：{item.get('parent_title') or item.get('title')}"
                if item.get("parent_title") or item.get("title")
                else "",
                item.get("text") or "",
            )
            if value
        )
        question_pairs.append((original_query, candidate_text))

    local_path = reranker_config.bge_reranker_path.strip()
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

def step_4_chunk_topk(chunk_list_score_sorted, minimum_topk=None):
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
    min_topk = max(RERANK_MIN_TOPK, int(minimum_topk or RERANK_MIN_TOPK))
    max_topk = RERANK_MAX_TOPK
    gap_ratio = RERANK_GAP_RATIO
    max_gap = RERANK_GAP_ABS
    #最多多少个
    max_topk =  min(max_topk,len(chunk_list_score_sorted))
    topk = max_topk
    if topk > min_topk:
        for index in range(0,max_topk-1):
             score_1 = chunk_list_score_sorted[index].get("score",0.0)
             score_2 = chunk_list_score_sorted[index+1].get("score",0.0)
             #分差
             abs_score = score_1 - score_2
            #比率
             ratio_score = abs_score / (score_1 + 1e-7)
             if abs_score > max_gap or ratio_score > gap_ratio:
                 #产生断崖了
                 topk = max(min_topk, index + 1)
                 break
    final_chunk_list = chunk_list_score_sorted[:topk]
    return final_chunk_list


def step_5_low_confidence_fallback(chunk_list_score_sorted):
    """保留非零低分候选供回答模型核验，丢弃已被重排判为完全无关的内容。"""
    fallback = []
    for chunk in chunk_list_score_sorted[:RERANK_FALLBACK_TOPK]:
        if float(chunk.get("score") or 0.0) <= 0.0:
            continue
        fallback.append({**chunk, "low_confidence": True})
    return fallback

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
        state["evidence_quality"] = "none"
        add_done_task(run_id, sys._getframe().f_code.co_name, state.get("is_stream"))
        return state
    final_chunk_list = step_2_limit_rerank_candidates(final_chunk_list)
    final_chunk_list_score_sorted = step_3_rerank_score_and_sort(state,final_chunk_list)
    final_chunk_list_score_sorted = deduplicate_documents(final_chunk_list_score_sorted)
    if any(chunk.get("score") is None for chunk in final_chunk_list_score_sorted):
        final_chunk_list_score_sorted_topk = final_chunk_list_score_sorted[:RERANK_MAX_TOPK]
        state["evidence_quality"] = "unscored"
    else:
        final_chunk_list_score_sorted_topk = step_4_chunk_topk(final_chunk_list_score_sorted)
        if final_chunk_list_score_sorted_topk:
            state["evidence_quality"] = "qualified"
        else:
            final_chunk_list_score_sorted_topk = step_5_low_confidence_fallback(
                final_chunk_list_score_sorted
            )
            state["evidence_quality"] = (
                "low" if final_chunk_list_score_sorted_topk else "none"
            )
    state["reranked_docs"] = final_chunk_list_score_sorted_topk
    if state.get("document_ids") and state["reranked_docs"]:
        client = get_milvus_client()
        if client:
            state["reranked_docs"] = deduplicate_documents(
                expand_reranked_neighbors(
                    client,
                    state["reranked_docs"],
                    retrieval_config.neighbor_expand_parts,
                )
            )
    add_done_task(run_id, sys._getframe().f_code.co_name, state.get("is_stream"))
    return state
