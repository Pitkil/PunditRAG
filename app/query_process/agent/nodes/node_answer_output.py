import sys
import time
from app.utils.task_utils import add_running_task, add_done_task, set_task_result
from app.utils.sse_utils import push_to_session, SSEEvent
from app.query_process.agent.state import QueryGraphState
from app.core.logger import logger, step_log, node_log
from app.core.load_prompt import load_prompt
from app.llm.llm_util import get_llm_client
from app.clients.mongo_history_utils import save_chat_message
from app.query_process.agent.source_utils import (
    build_source_records,
    compact_citations,
    deduplicate_documents,
    reject_invalid_citations,
)
import re

GENERAL_KNOWLEDGE_NOTICE = "> **AI 通识回答**：当前知识库未提供直接依据。"

@step_log("step_1_data_validates")
def step_1_data_validates(state):
    """
    验证输入数据的有效性
    """
    answer = state.get("answer")
    is_stream = state.get("is_stream", True)
    run_id = state.get("run_id") or state.get("session_id")
    #完整答案就已经存在
    #说明：1.搜索关键词不确定 2.未找到关键词
    if answer:
        if is_stream and not state.get("answer_streamed", False):
            push_to_session(run_id, SSEEvent.DELTA, {"delta": answer})
        set_task_result(run_id, "answer", answer)
        return True #返回 True：已有答案，本轮处理完毕，不用走后续流程
    return False #返回 False：无答案，需要继续走完整检索流程

@step_log("step_2_data_validates")
def step_2_data_validates(state):
    history = state.get("history", [])
    reranked_docs = deduplicate_documents(state.get("reranked_docs", []))
    state["reranked_docs"] = reranked_docs
    item_names = state.get("item_names", [])
    original_query = state.get("original_query", "")
    if not original_query:
        logger.warning("original_query为空，无法继续处理。")
        raise ValueError("original_query为空，无法继续处理。")

    return history, reranked_docs, item_names, original_query

@ step_log("step_3_make_prompt")
def _clean_history_text(text, limit=1200):
    """移除旧引用编号并限制长度，历史仅用于对话理解。"""
    cleaned = re.sub(r"(?:\[\d+\])+", "", str(text or ""))
    return cleaned.strip()[:limit]


def step_3_make_prompt(state, history, reranked_docs, item_names, original_query):
    context_chunk_list = []
    for number, chunk in enumerate(reranked_docs,start=1):
        score_text = (
            f"匹配度得分:{chunk['score']}"
            if chunk.get("score") is not None
            else f"搜索排序:第{chunk.get('search_rank', number)}位（未经过本地重排序）"
        )
        context_chunk_list.append(
            f"<source id=\"{number}\" type=\"{'web' if chunk.get('type') == 'web' else 'knowledge_base'}\">\n"
            f"标题:{chunk.get('title') or '未命名来源'} {score_text}\n"
            f"内容:\n{chunk.get('text') or chunk.get('content') or ''}\n"
            f"</source>"
        )
    context_chunk_str = "\n\n".join(context_chunk_list) or "无可用参考内容"

    history_text = "没有历史聊天记录!"
    if history:
        history_lines = []
        for msg in history:
            role = msg.get("role")
            if role not in {"user", "assistant"}:
                continue
            content = msg.get("text") or msg.get("rewritten_query") or ""
            cleaned = _clean_history_text(content)
            if not cleaned:
                continue
            history_lines.append(
                f"角色:{role},内容:{cleaned}"
            )
        history_text = "\n".join(history_lines) or "没有可用历史对话!"

    item_name = "本次关联主体:" + ",".join(item_names) if item_names and len(item_names) > 0 else '没有关联主体'
    # 加载提示词
    available_images = state.get("image_urls", [])
    evidence_quality = state.get("evidence_quality", "qualified")
    evidence_notice = {
        "full_context": "已读取用户明确选择范围内的完整正文；请直接依据正文回答并逐项标注来源。",
        "qualified": "候选内容已通过相关性阈值，但仍须逐条核对正文后作答。",
        "low": "候选内容来自用户选定范围，但重排分较低（跨语言或指代查询也可能导致低分）；请直接核对正文，只使用其中确实支持问题的内容。",
        "unscored": "候选内容未经过本地重排，请严格依据正文判断是否足以作答。",
    }.get(
        evidence_quality,
        "未提供可用候选内容，本轮不存在任何 source id。请使用稳定通识回答，且不得输出任何 [n] 引用。",
    )
    prompt = load_prompt("answer_out",context = context_chunk_str,
                history = history_text, item_names = item_name, question = original_query,
                evidence_notice = evidence_notice,
                image_urls = "\n".join(available_images) if available_images else "无可用图片")

    return prompt

@step_log("step_4_generate_answer")
def step_4_generate_answer(state, prompt):
    """
    最终答案生成
    """
    is_stream = state.get("is_stream", True)
    run_id = state.get("run_id") or state.get("session_id")
    llm_client = get_llm_client()
    final_answer = ""
    if is_stream:
        buffer = ""
        last_flush = time.monotonic()
        for chunk in llm_client.stream(prompt):
            delta_content = str(chunk.content)
            final_answer += delta_content
            buffer += delta_content
            now = time.monotonic()
            if len(buffer) >= 16 or now - last_flush >= 0.06:
                push_to_session(run_id, SSEEvent.DELTA, {"delta": buffer})
                buffer = ""
                last_flush = now
        if buffer:
            push_to_session(run_id, SSEEvent.DELTA, {"delta": buffer})
    else:
        response = llm_client.invoke(prompt)
        final_answer = str(response.content)
    set_task_result(run_id,"answer",final_answer)
    state['answer'] = final_answer

@step_log("step_5_extract_chunk_images")
def step_5_extract_chunk_images(state, reranked_docs):
    """
    提取切片中图片数据
    """
    image_urls = []
    #编译一个正则表达式，用来匹配 Markdown 格式的图片语法，例如：![描述](图片链接)
    reg = re.compile(r"\!\[.*?\]\((.*?)\)") 

    for chunk in reranked_docs:
        url = chunk.get("url")
        text = chunk.get("text", "")
        #web搜索字段里可能有图片url
        if url:
            if url.endswith((".png", ".jpg",".gif",".jpeg",".svg")):
                if url not in image_urls:
                    image_urls.append(url)

        if text:
            image_list = reg.findall(text)
            for image_url in image_list:
                if image_url not in image_urls:
                    image_urls.append(image_url)

        for image_url in chunk.get("image_urls") or []:
            if isinstance(image_url, str) and image_url not in image_urls:
                image_urls.append(image_url)

    state['image_urls'] = image_urls


@step_log("step_5_filter_answer_images")
def step_5_filter_answer_images(state):
    """仅保留参考资料中真实存在的图片 URL，并移除模型编造的图片区块。"""
    answer = state.get("answer") or ""
    allowed_images = set(state.get("image_urls") or [])
    # 模型可能在无图片时输出“【图片】无可用图片”；先识别末尾图片区块，再只保留白名单 URL。
    image_block = re.compile(r"\n*【图片】\s*\n(?P<body>[\s\S]*?)\s*$", re.IGNORECASE)
    match = image_block.search(answer)
    selected_images = []
    if match:
        for image_url in re.findall(r"https?://[^\s>]+", match.group("body"), re.IGNORECASE):
            if image_url in allowed_images and image_url not in selected_images:
                selected_images.append(image_url)
        answer = answer[:match.start()].rstrip()

    state["answer"] = answer
    state["image_urls"] = selected_images
    set_task_result(state.get("run_id") or state.get("session_id"), "answer", answer)


@step_log("step_5_build_sources")
def step_5_build_sources(state, reranked_docs):
    """仅返回答案实际引用的去重来源，避免把候选上下文冒充证据。"""
    candidates = build_source_records(reranked_docs)
    original_answer = state.get("answer", "")
    validated_answer = reject_invalid_citations(original_answer, candidates)
    invalid_citation = validated_answer != original_answer
    if invalid_citation:
        logger.warning("答案引用了本轮不存在的来源编号，已拒绝该答案")
    answer, sources = compact_citations(validated_answer, candidates)
    if invalid_citation:
        state["answer_basis"] = "refused"
    elif sources:
        state["answer_basis"] = "sources"
        stripped = answer.lstrip()
        if stripped.startswith(GENERAL_KNOWLEDGE_NOTICE):
            answer = stripped[len(GENERAL_KNOWLEDGE_NOTICE):].lstrip()
    else:
        state["answer_basis"] = "general"
        if answer.strip() and not answer.lstrip().startswith(GENERAL_KNOWLEDGE_NOTICE):
            answer = f"{GENERAL_KNOWLEDGE_NOTICE}\n\n{answer.strip()}"
    state["answer"], state["sources"] = answer, sources
    set_task_result(
        state.get("run_id") or state.get("session_id"),
        "answer",
        state["answer"],
    )

@step_log("step_6_save_chat_history")
def step_6_save_chat_history(state):
    """
    保存聊天记录
    """
    message_id = save_chat_message(
        session_id= state['session_id'],
        role="assistant",
        text=state.get("answer"),
        rewritten_query=state.get("rewritten_query") or state.get("original_query"),
        item_names=state.get("item_names",[]),
        image_urls = state.get("image_urls",[]),
        sources=state.get("sources", []),
        kb_ids=state.get("kb_ids", []),
        document_ids=state.get("document_ids", []),
    )
    state["assistant_message_id"] = message_id
    set_task_result(
        state.get("run_id") or state["session_id"],
        "assistant_message_id",
        message_id,
    )
    return message_id

@node_log("node_answer_output")
def node_answer_output(state):
    """
    节点功能：进行过处理可以是流式输出可以整体输出
    """
    run_id = state.get("run_id") or state["session_id"]
    add_running_task(run_id, sys._getframe().f_code.co_name, state.get("is_stream", False))
    has_answer = step_1_data_validates(state)
    if not has_answer:
        history, reranked_docs, item_names, original_query = step_2_data_validates(state)
        step_5_extract_chunk_images(state, reranked_docs)
        prompt = step_3_make_prompt(state, history, reranked_docs, item_names, original_query)
        step_4_generate_answer(state, prompt)
        step_5_filter_answer_images(state)
        step_5_build_sources(state, reranked_docs)
    step_6_save_chat_history(state)
    add_done_task(run_id, sys._getframe().f_code.co_name, state.get("is_stream", False))
    return state
