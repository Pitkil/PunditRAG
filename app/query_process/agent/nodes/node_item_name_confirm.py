import json
import os
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
from app.clients.mongo_workspace_utils import (
    get_document,
    list_documents,
    list_knowledge_bases,
)
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger, node_log, step_log
from app.llm.embedding_utils import generate_embeddings
from app.llm.llm_util import get_llm_client
from app.utils.task_utils import add_done_task, add_running_task


PLANNER_TIMEOUT_SECONDS = float(os.getenv("PLANNER_TIMEOUT_SECONDS", "10"))


def get_direct_chat_answer(query):
    """对无需检索的明确寒暄直接回复，避免无意义地加载向量模型。"""
    normalized = "".join(str(query).strip().lower().split()).rstrip("!！?？。,.，")
    greetings = {"你好", "您好", "嗨", "hi", "hello", "在吗", "你是谁"}
    if normalized in greetings:
        return "你好，我是 PunditRAG。你可以向我询问已导入资料中的内容。"
    return ""


def build_scope_context(state):
    """读取本轮显式范围；范围信息只帮助规划，不改变用户原问题。"""
    scope_mode = state.get("scope_mode", "knowledge_base")
    kb_ids = state.get("kb_ids") or []
    document_ids = state.get("document_ids") or []
    try:
        kb_names = {
            item.get("kb_id"): item.get("name") or item.get("kb_id")
            for item in list_knowledge_bases()
        }
        if document_ids:
            documents = [get_document(document_id) for document_id in document_ids]
            documents = [document for document in documents if document]
        elif scope_mode == "knowledge_base" and kb_ids:
            selected_ids = set(kb_ids)
            documents = [
                document for document in list_documents()
                if document.get("kb_id") in selected_ids
            ]
        else:
            documents = []
    except Exception as exc:
        logger.warning(f"读取显式资料范围失败，按请求中的 ID 继续规划：{exc}")
        kb_names = {}
        documents = []

    document_names = [
        document.get("filename") or document.get("file_title") or document.get("document_id")
        for document in documents[:8]
    ]
    state["scope_document_names"] = document_names
    selected_kbs = [kb_names.get(kb_id, kb_id) for kb_id in kb_ids]

    if document_ids:
        scope_label = "用户当前明确选择的文档"
    elif scope_mode == "knowledge_base":
        scope_label = "用户当前明确选择的知识库"
    else:
        scope_label = "全部知识库"

    context = (
        f"范围模式：{scope_mode}\n"
        f"范围含义：{scope_label}\n"
        f"知识库：{'、'.join(selected_kbs) or '全部知识库'}\n"
        f"范围内文档数量：{len(documents) if scope_mode != 'all' else '未限定'}\n"
        f"范围内文档：{'、'.join(document_names) or '未指定单篇文档'}"
    )
    return context

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
def step_3_llm_itemnames_and_rewrite(history_message_list, original_query, scope_context=""):
    """提取召回主题，并判断是否必须读取当前范围的全部正文。"""
    history_lines = []
    for message in history_message_list:
        role = message.get("role", "")
        content = message.get("text", "")
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
        scope_text=scope_context or "未提供显式范围",
        query=original_query,
    )
    messages = [
        SystemMessage(
            content="你是知识库检索规划助手，只提取召回主题和全文读取策略，不得改写或回答用户问题。"
        ),
        HumanMessage(content=prompt),
    ]
    try:
        result = (
            get_llm_client(
                json_mode=True,
                timeout=PLANNER_TIMEOUT_SECONDS,
                max_retries=0,
            )
            | JsonOutputParser()
        ).invoke(messages)
    except Exception as exc:
        logger.warning(f"查询规划超时或失败，按普通检索继续：{exc}")
        return {
            "rewritten_query": original_query,
            "item_names": [],
            "full_document": False,
        }

    item_names = result.get("item_names") or []
    if isinstance(item_names, str):
        item_names = [item_names]
    if not isinstance(item_names, list):
        logger.warning("模型返回的 item_names 格式无效，已使用空列表")
        item_names = []

    return {
        "rewritten_query": original_query,
        "item_names": list(dict.fromkeys(str(name).strip() for name in item_names if str(name).strip())),
        "full_document": result.get("full_document") is True,
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
        query_plan = {}
    state["rewritten_query"] = state.get("original_query", "")
    state["full_document"] = query_plan.get("full_document") is True
    state["answer"] = ""

    if confirmed_names:
        state["item_names"] = confirmed_names
        return

    state["item_names"] = []
    if optional_names:
        logger.info(f"主题匹配置信度不足，将执行知识库全局检索：{optional_names}")

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
        state["full_document"] = False
        state["answer"] = direct_answer
        step_7_save_user_chat_message(state)
        add_done_task(run_id, node_name, is_stream)
        return state

    scope_context = build_scope_context(state)
    query_result = step_3_llm_itemnames_and_rewrite(
        history,
        original_query,
        scope_context,
    )
    vector_dict = step_4_vector_query_item_name(
        query_result["item_names"],
        state.get("kb_ids", []),
    ) if query_result["item_names"] else {}
    final_result = step_5_select_item_list(vector_dict)

    state["history"] = history
    step_6_deal_state(state, final_result, query_result)
    step_7_save_user_chat_message(state)

    add_done_task(run_id, node_name, is_stream)
    return state
