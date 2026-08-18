import json
import sys

import numpy as np
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

from app.clients.milvus_utils import (
    create_hybrid_search_requests,
    get_milvus_client,
    hybrid_search,
)
from app.clients.mongo_history_utils import get_recent_messages, save_chat_message
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger, node_log, step_log
from app.llm.embedding_utils import generate_embeddings
from app.llm.llm_util import get_llm_client
from app.utils.task_utils import add_done_task, add_running_task


def get_direct_chat_answer(query):
    """对无需检索的明确寒暄直接回复，避免无意义地加载向量模型。"""
    normalized = "".join(str(query).strip().lower().split()).rstrip("!！?？。,.，")
    greetings = {"你好", "您好", "嗨", "hi", "hello", "在吗", "你是谁"}
    if normalized in greetings:
        return "你好，我是 PunditRAG。你可以向我询问已导入资料中的内容。"
    return ""


QUERY_MODES = {"lookup", "explain", "summarize", "compare", "clarify"}
QUERY_DEPTHS = {"brief", "normal", "deep"}


def _fallback_query_mode(query: str) -> str:
    value = str(query or "")
    if any(word in value for word in ("比较", "对比", "区别", "差异", "异同")):
        return "compare"
    if any(word in value for word in ("总结", "概括", "梳理", "提纲", "核心内容", "主要内容")):
        return "summarize"
    if any(word in value for word in ("讲解", "解释", "分析", "原理", "流程", "为什么", "详细", "深入", "展开")):
        return "explain"
    return "lookup"


def _fallback_query_depth(query: str) -> str:
    value = str(query or "")
    if any(word in value for word in ("详细", "深入", "全面", "系统", "展开", "继续", "更具体", "再补充")):
        return "deep"
    if any(word in value for word in ("简要", "简短", "一句话")):
        return "brief"
    return "normal"


def _recent_document_ids(history_message_list, original_query: str) -> list[str]:
    follow_up_words = ("继续", "详细一点", "更详细", "展开", "再说", "补充", "这个", "这篇", "该文", "上述", "它")
    if not any(word in str(original_query or "") for word in follow_up_words):
        return []
    for message in reversed(history_message_list):
        if message.get("role") != "assistant":
            continue
        document_ids = [
            source.get("document_id")
            for source in message.get("sources") or []
            if source.get("document_id")
        ]
        if document_ids:
            return list(dict.fromkeys(document_ids))
    return []

@step_log("node_item_name_confirm")
def step_1_data_validates(state):
    """校验并返回会话 ID 和原始问题。"""
    original_query = state.get("original_query")
    session_id = state.get("session_id")
    if not original_query or not session_id:
        logger.error("session_id 和 original_query 不能为空")
        raise ValueError("original_query 和 session_id 不能为空")
    return original_query, session_id

@step_log("step_2_chat_history")
def step_2_chat_history(session_id):
    """获取当前会话的最近聊天记录。"""
    return get_recent_messages(session_id)

@step_log("step_3_llm_itemnames_and_rewrite")
def step_3_llm_itemnames_and_rewrite(history_message_list, original_query):
    """结合历史对话识别主题或实体，并将问题改写为独立查询。"""
    history_lines = []
    for message in history_message_list:
        role = message.get("role", "")
        content = (
            message.get("rewritten_query")
            if role == "user" and message.get("rewritten_query")
            else message.get("text", "")
        )
        related_names = message.get("item_names") or []
        related_documents = message.get("document_ids") or [
            source.get("document_id")
            for source in message.get("sources") or []
            if source.get("document_id")
        ]
        history_lines.append(
            f"角色：{role}，内容：{str(content)[:1600]}，关联主题或实体：{'、'.join(related_names)}，"
            f"关联文档：{'、'.join(related_documents)}"
        )

    prompt = load_prompt(
        "rewritten_query_and_itemnames",
        history_text="\n".join(history_lines),
        query=original_query,
    )
    messages = [
        SystemMessage(
            content="你是通用知识库查询理解助手，负责识别检索主题并改写用户问题。"
        ),
        HumanMessage(content=prompt),
    ]
    result = (get_llm_client(json_mode=True) | JsonOutputParser()).invoke(messages)

    rewritten_query = result.get("rewritten_query") or original_query
    item_names = result.get("item_names") or []
    if isinstance(item_names, str):
        item_names = [item_names]
    if not isinstance(item_names, list):
        logger.warning("模型返回的 item_names 格式无效，已使用空列表")
        item_names = []

    mode = str(result.get("mode") or "").strip().lower()
    depth = str(result.get("depth") or "").strip().lower()
    aspects = result.get("aspects") or []
    if mode not in QUERY_MODES:
        mode = _fallback_query_mode(original_query)
    if depth not in QUERY_DEPTHS:
        depth = _fallback_query_depth(original_query)
    if isinstance(aspects, str):
        aspects = [aspects]
    if not isinstance(aspects, list):
        aspects = []

    return {
        "rewritten_query": rewritten_query,
        "item_names": list(dict.fromkeys(str(name).strip() for name in item_names if str(name).strip())),
        "mode": mode,
        "depth": depth,
        "aspects": list(dict.fromkeys(str(value).strip() for value in aspects if str(value).strip()))[:8],
    }

@step_log("step_4_vector_query_item_name")
def step_4_vector_query_item_name(item_names, kb_ids=None):
    """在主题名称集合中查找与用户问题相关的已导入资料。"""
    if not item_names or not kb_ids:
        return {}

    milvus_client = get_milvus_client()
    collection_name = milvus_config.item_name_collection
    if not milvus_client or not collection_name:
        logger.warning("主题名称集合不可用，将跳过资料范围匹配并继续全库检索")
        return {}
    if not milvus_client.has_collection(collection_name):
        logger.warning(f"主题名称集合不存在：{collection_name}，将继续全库检索")
        return {}

    try:
        embeddings = generate_embeddings(item_names)
    except RuntimeError as exc:
        if "本地模型尚未下载完整" not in str(exc):
            raise
        logger.warning("BGE-M3 尚未就绪，跳过资料主题匹配，保留联网搜索分支")
        return {}
    vector_dict = {}
    for index, item_name in enumerate(item_names):
        requests = create_hybrid_search_requests(
            np.asarray(embeddings["dense"][index], dtype=np.float16),
            embeddings["sparse"][index],
            expr=(
                f"kb_id in {json.dumps(kb_ids, ensure_ascii=False)}"
                if kb_ids
                else None
            ),
        )
        response = hybrid_search(
            client=milvus_client,
            collection_name=collection_name,
            reqs=requests,
            ranker_weights=(0.8, 0.2),
            norm_score=True,
            output_fields=["item_name"],
        )

        candidates = []
        if response and response[0]:
            for hit in response[0]:
                entity = hit.get("entity", {})
                candidate_name = entity.get("item_name", "").strip()
                if candidate_name:
                    candidates.append(
                        {
                            "item_name": candidate_name,
                            "score": float(hit.get("distance", 0)),
                        }
                    )
        vector_dict[item_name] = candidates
    return vector_dict

@step_log("step_5_select_item_list")
def step_5_select_item_list(vector_dict):
    """按相似度划分已确认资料与待确认资料。"""
    confirmed_item_names = []
    optional_item_names = []

    for candidates in vector_dict.values():
        candidates.sort(key=lambda item: item["score"], reverse=True)
        high_candidates = [item for item in candidates if item["score"] >= 0.65]
        low_candidates = [
            item for item in candidates if 0.50 <= item["score"] < 0.65
        ]

        if high_candidates:
            confirmed_item_names.append(high_candidates[0]["item_name"])
        else:
            optional_item_names.extend(
                item["item_name"] for item in low_candidates[:2]
            )

    return {
        "confirmed_item_name_list": list(dict.fromkeys(confirmed_item_names)),
        "options_item_name_list": list(dict.fromkeys(optional_item_names)),
    }

@step_log("step_6_deal_state")
def step_6_deal_state(state, final_result, query_plan):
    """确定主题扩展词；主题不明确时仍在显式知识库范围内检索。"""
    confirmed_names = final_result.get("confirmed_item_name_list", [])
    optional_names = final_result.get("options_item_name_list", [])

    if isinstance(query_plan, str):
        query_plan = {"rewritten_query": query_plan}
    rewritten_query = query_plan.get("rewritten_query") or state.get("original_query", "")
    state["rewritten_query"] = rewritten_query
    state["query_mode"] = query_plan.get("mode") or _fallback_query_mode(rewritten_query)
    state["query_depth"] = query_plan.get("depth") or _fallback_query_depth(rewritten_query)
    state["query_aspects"] = query_plan.get("aspects") or []
    if state["query_aspects"] and state["query_mode"] in {"explain", "compare"}:
        aspect_context = "、".join(state["query_aspects"])
        if aspect_context not in state["rewritten_query"]:
            state["rewritten_query"] = f"{state['rewritten_query']}（需要覆盖：{aspect_context}）"
    state["answer"] = ""

    if confirmed_names:
        state["item_names"] = confirmed_names
        topic_context = "、".join(confirmed_names)
        if topic_context and topic_context not in rewritten_query:
            state["rewritten_query"] = f"{rewritten_query}（资料主题：{topic_context}）"
        return

    state["item_names"] = []
    if optional_names:
        logger.info(f"主题匹配置信度不足，将执行知识库全局检索：{optional_names}")
    if state["query_mode"] == "clarify" and not state.get("document_ids"):
        state["answer"] = "请先明确要查询的资料或对象，或在顶部选择一份具体文档。"

@step_log("step_7_save_user_chat_message")
def step_7_save_user_chat_message(state):
    """保存本次用户问题及查询理解结果。"""
    save_chat_message(
        session_id=state["session_id"],
        role="user",
        text=state["original_query"],
        rewritten_query=state["rewritten_query"],
        item_names=state["item_names"],
        kb_ids=state.get("kb_ids", []),
        document_ids=state.get("document_ids", []),
        query_mode=state.get("query_mode", ""),
        query_depth=state.get("query_depth", ""),
        query_aspects=state.get("query_aspects", []),
    )

@node_log("node_item_name_confirm")
def node_item_name_confirm(state):
    """识别查询主题、匹配资料范围并保存用户消息。"""
    session_id = state.get("session_id")
    run_id = state.get("run_id") or session_id
    is_stream = state.get("is_stream")
    node_name = sys._getframe().f_code.co_name
    add_running_task(run_id, node_name, is_stream)

    original_query, session_id = step_1_data_validates(state)
    history = step_2_chat_history(session_id)
    direct_answer = get_direct_chat_answer(original_query)
    if direct_answer:
        state["history"] = history
        state["rewritten_query"] = original_query
        state["item_names"] = []
        state["answer"] = direct_answer
        step_7_save_user_chat_message(state)
        add_done_task(run_id, node_name, is_stream)
        return state

    query_result = step_3_llm_itemnames_and_rewrite(history, original_query)
    if not state.get("document_ids"):
        state["document_ids"] = _recent_document_ids(history, original_query)
    lookup_terms = query_result["item_names"] or [query_result["rewritten_query"]]
    vector_dict = step_4_vector_query_item_name(lookup_terms, state.get("kb_ids", []))
    final_result = step_5_select_item_list(vector_dict)

    state["history"] = history
    step_6_deal_state(state, final_result, query_result)
    step_7_save_user_chat_message(state)

    add_done_task(run_id, node_name, is_stream)
    return state
