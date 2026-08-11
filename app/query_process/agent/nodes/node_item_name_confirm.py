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
from app.core.logger import logger
from app.llm.embedding_utils import generate_embeddings
from app.llm.llm_util import get_llm_client
from app.utils.task_utils import add_done_task, add_running_task


def step_1_data_validates(state):
    """校验并返回会话 ID 和原始问题。"""
    original_query = state.get("original_query")
    session_id = state.get("session_id")
    if not original_query or not session_id:
        logger.error("session_id 和 original_query 不能为空")
        raise ValueError("original_query 和 session_id 不能为空")
    return original_query, session_id


def step_2_chat_history(session_id):
    """获取当前会话的最近聊天记录。"""
    return get_recent_messages(session_id)


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
        history_lines.append(
            f"角色：{role}，内容：{content}，关联主题或实体：{'、'.join(related_names)}"
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

    return {
        "rewritten_query": rewritten_query,
        "item_names": list(dict.fromkeys(str(name).strip() for name in item_names if str(name).strip())),
    }


def step_4_vector_query_item_name(item_names):
    """在主题名称集合中查找与用户问题相关的已导入资料。"""
    if not item_names:
        return {}

    milvus_client = get_milvus_client()
    collection_name = milvus_config.item_name_collection
    if not milvus_client or not collection_name:
        logger.warning("主题名称集合不可用，将跳过资料范围匹配并继续全库检索")
        return {}
    if not milvus_client.has_collection(collection_name):
        logger.warning(f"主题名称集合不存在：{collection_name}，将继续全库检索")
        return {}

    embeddings = generate_embeddings(item_names)
    vector_dict = {}
    for index, item_name in enumerate(item_names):
        requests = create_hybrid_search_requests(
            np.asarray(embeddings["dense"][index], dtype=np.float16),
            embeddings["sparse"][index],
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


def step_6_deal_state(state, final_result, rewritten_query):
    """确定检索范围；无明确范围时允许后续节点执行全库检索。"""
    confirmed_names = final_result.get("confirmed_item_name_list", [])
    optional_names = final_result.get("options_item_name_list", [])

    state["rewritten_query"] = rewritten_query
    state["answer"] = ""

    if confirmed_names:
        state["item_names"] = confirmed_names
        return

    state["item_names"] = []
    if len(optional_names) > 1:
        option_text = "、".join(optional_names)
        state["answer"] = f"检索到多个可能相关的资料主题：{option_text}。请说明你想查询哪一个。"


def step_7_save_user_chat_message(state):
    """保存本次用户问题及查询理解结果。"""
    save_chat_message(
        session_id=state["session_id"],
        role="user",
        text=state["original_query"],
        rewritten_query=state["rewritten_query"],
        item_names=state["item_names"],
    )


def node_item_name_confirm(state):
    """识别查询主题、匹配资料范围并保存用户消息。"""
    session_id = state.get("session_id")
    is_stream = state.get("is_stream")
    node_name = sys._getframe().f_code.co_name
    add_running_task(session_id, node_name, is_stream)

    original_query, session_id = step_1_data_validates(state)
    history = step_2_chat_history(session_id)
    query_result = step_3_llm_itemnames_and_rewrite(history, original_query)
    vector_dict = step_4_vector_query_item_name(query_result["item_names"])
    final_result = step_5_select_item_list(vector_dict)

    state["history"] = history
    step_6_deal_state(state, final_result, query_result["rewritten_query"])
    step_7_save_user_chat_message(state)

    add_done_task(session_id, node_name, is_stream)
    return state
