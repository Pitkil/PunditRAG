import sys
from app.utils.task_utils import *
from dotenv import load_dotenv
import sys
from app.llm.reranker_utils import get_reranker_model
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
                "score":0.0
               }
            )

    if web_search_docs and len(web_search_docs) > 0:
        for doc in web_search_docs:
            final_list.append(
               {
                "title":doc.get("title"),
                "text":doc.get("content") or doc.get("snippet") or "",
                "url":doc.get("url"),
                "type":"web",
                "score":0.0
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
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    rrf_chunks, web_search_docs = step_1_data_validates(state)
    final_chunk_list = step_2_merged_rrf_and_mcp(rrf_chunks,web_search_docs)
    if not final_chunk_list:
        state["reranked_docs"] = []
        state["answer"] = "没有检索到足够的相关资料，暂时无法基于知识库回答这个问题。"
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
        return state
    final_chunk_list_score_sorted = step_3_rerank_score_and_sort(state,final_chunk_list)
    final_chunk_list_score_sorted_topk = step_4_chunk_topk(final_chunk_list_score_sorted)
    state["reranked_docs"] = final_chunk_list_score_sorted_topk
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state
