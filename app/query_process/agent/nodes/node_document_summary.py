import json
import os
import sys
import time
from typing import Dict, Iterable, List

from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.conf.retrieval_config import retrieval_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger, node_log, step_log
from app.llm.llm_util import get_llm_client
from app.query_process.agent.source_utils import (
    build_source_records,
    compact_citations,
    reject_invalid_citations,
)
from app.utils.task_utils import add_done_task, add_running_task
from app.utils.sse_utils import push_to_session, SSEEvent


SUMMARY_BATCH_CHARS = int(os.getenv("SUMMARY_BATCH_CHARS", "24000"))
SUMMARY_MAX_CHUNKS = int(os.getenv("SUMMARY_MAX_CHUNKS", "2000"))
DIRECT_DOCUMENT_MAX_CHARS = retrieval_config.direct_document_max_chars


def is_document_summary_request(state) -> bool:
    """只有明确的整份资料总结才进入全文综合链路。"""
    return state.get("full_document") is True


@step_log("step_1_load_summary_chunks")
def step_1_load_summary_chunks(state) -> List[Dict]:
    kb_ids = state.get("kb_ids") or []
    if not kb_ids:
        logger.info("未选择知识库，跳过整库摘要")
        return []

    client = get_milvus_client()
    collection = milvus_config.chunks_collection
    if not client or not collection or not client.has_collection(collection):
        return []

    filters = []
    document_ids = state.get("document_ids") or []
    item_names = state.get("item_names") or []
    if document_ids:
        filters.append(f"document_id in {json.dumps(document_ids, ensure_ascii=False)}")
    else:
        filters.append(f"kb_id in {json.dumps(kb_ids, ensure_ascii=False)}")
    if item_names and not document_ids:
        filters.append(f"item_name in {json.dumps(item_names, ensure_ascii=False)}")
    rows = client.query(
        collection_name=collection,
        filter=" and ".join(filters),
        output_fields=[
            "chunk_id", "content", "title", "parent_title", "part", "chunk_index", "file_title",
            "item_name", "kb_id", "document_id",
        ],
        limit=SUMMARY_MAX_CHUNKS,
    )
    documents = [
        {
            **row,
            "text": row.get("content", ""),
            "type": "milvus",
            "score": None,
        }
        for row in rows
    ]
    documents.sort(key=lambda item: (
        str(item.get("file_title") or ""),
        int(item.get("chunk_index") or item.get("chunk_id") or 0),
    ))
    return documents


def _make_batches(entries: Iterable[str], max_chars: int = SUMMARY_BATCH_CHARS) -> List[str]:
    batches: List[str] = []
    current: List[str] = []
    current_length = 0
    for entry in entries:
        if current and current_length + len(entry) > max_chars:
            batches.append("\n\n".join(current))
            current = []
            current_length = 0
        current.append(entry)
        current_length += len(entry)
    if current:
        batches.append("\n\n".join(current))
    return batches


def _invoke_text(prompt: str, state=None) -> str:
    llm = get_llm_client()
    if not state or not state.get("is_stream"):
        response = llm.invoke(prompt)
        return str(response.content).strip()

    run_id = state.get("run_id") or state.get("session_id")
    final_answer = ""
    buffer = ""
    last_flush = time.monotonic()
    for chunk in llm.stream(prompt):
        delta = str(chunk.content)
        final_answer += delta
        buffer += delta
        now = time.monotonic()
        if len(buffer) >= 16 or now - last_flush >= 0.06:
            push_to_session(run_id, SSEEvent.DELTA, {"delta": buffer})
            buffer = ""
            last_flush = now
    if buffer:
        push_to_session(run_id, SSEEvent.DELTA, {"delta": buffer})
    state["answer_streamed"] = True
    return final_answer.strip()


@step_log("step_2_direct_synthesis")
def step_2_direct_synthesis(question: str, sources: List[Dict], state=None) -> str:
    context = "\n\n".join(
        f"<source id=\"{source['index']}\">\n"
        f"标题：{source.get('file_title') or source.get('title')} / "
        f"{source.get('parent_title') or source.get('title')}\n"
        f"内容：\n{source.get('content', '')}\n</source>"
        for source in sources
    )
    return _invoke_text(
        load_prompt("document_synthesis", question=question, context=context),
        state,
    )


@step_log("step_2_map_summary")
def step_2_map_summary(question: str, sources: List[Dict]) -> List[str]:
    entries = [
        f"来源[{source['index']}] {source.get('file_title') or source.get('title')} / "
        f"{source.get('parent_title') or source.get('title')}\n{source.get('content', '')}"
        for source in sources
    ]
    batches = _make_batches(entries)
    prompts = [
        load_prompt("summary_map", question=question, context=batch)
        for batch in batches
    ]
    return [_invoke_text(prompt) for prompt in prompts]


@step_log("step_3_reduce_summary")
def step_3_reduce_summary(question: str, summaries: List[str], state=None) -> str:
    current = summaries
    while len(current) > 1:
        batches = _make_batches(current)
        final_round = len(batches) == 1
        current = [
            _invoke_text(
                load_prompt("summary_reduce", question=question, summaries=batch),
                state if final_round else None,
            )
            for batch in batches
        ]
    if current and state and state.get("is_stream") and not state.get("answer_streamed"):
        push_to_session(
            state.get("run_id") or state.get("session_id"),
            SSEEvent.DELTA,
            {"delta": current[0]},
        )
        state["answer_streamed"] = True
    return current[0] if current else ""


@node_log("node_document_summary")
def node_document_summary(state):
    run_id = state.get("run_id") or state["session_id"]
    node_name = sys._getframe().f_code.co_name
    add_running_task(run_id, node_name, state.get("is_stream", False))

    documents = step_1_load_summary_chunks(state)
    candidates = build_source_records(documents)
    if not candidates:
        state["answer"] = "当前范围内没有可用于整体概括的资料。"
        state["sources"] = []
    else:
        question = state.get("original_query") or "概括资料"
        content_chars = sum(len(source.get("content") or "") for source in candidates)
        if content_chars <= DIRECT_DOCUMENT_MAX_CHARS:
            draft = step_2_direct_synthesis(question, candidates, state)
            mapped_count = 1
        else:
            mapped = step_2_map_summary(question, candidates)
            draft = step_3_reduce_summary(question, mapped, state)
            mapped_count = len(mapped)
        validated = reject_invalid_citations(draft, candidates)
        state["answer"], state["sources"] = compact_citations(validated, candidates)
        logger.info(
            f"文档综合完成：读取{len(candidates)}个去重切片，"
            f"生成{mapped_count}个证据批次，引用{len(state['sources'])}个来源"
        )

    add_done_task(run_id, node_name, state.get("is_stream", False))
    return state
