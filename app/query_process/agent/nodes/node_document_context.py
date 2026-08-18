import json
import os
import sys

from app.clients.milvus_utils import get_milvus_client
from app.clients.mongo_workspace_utils import get_document, update_document
from app.conf.milvus_config import milvus_config
from app.conf.retrieval_config import retrieval_config
from app.core.logger import logger, node_log
from app.utils.task_utils import add_done_task, add_running_task


DOCUMENT_CONTEXT_QUERY_LIMIT = int(os.getenv("SUMMARY_MAX_CHUNKS", "2000"))
DOCUMENT_CONTEXT_FIELDS = [
    "chunk_id",
    "content",
    "title",
    "parent_title",
    "part",
    "chunk_index",
    "file_title",
    "item_name",
    "kb_id",
    "document_id",
]


def _as_context_document(row):
    return {
        **row,
        "text": row.get("content", ""),
        "type": "milvus",
        "score": None,
    }


def _load_document_chunks(document_ids, expected_count):
    client = get_milvus_client()
    collection = milvus_config.chunks_collection
    if not client or not collection or not client.has_collection(collection):
        return []
    limit = min(max(1, int(expected_count or 1)), DOCUMENT_CONTEXT_QUERY_LIMIT)
    return client.query(
        collection_name=collection,
        filter=f"document_id in {json.dumps(document_ids, ensure_ascii=False)}",
        output_fields=DOCUMENT_CONTEXT_FIELDS,
        limit=limit,
    )


def prepare_document_context(state):
    """Load complete explicitly selected documents only when they fit the input budget."""
    state["document_context_complete"] = False
    if (
        state.get("answer")
        or state.get("full_document") is True
        or state.get("enable_web_search") is True
    ):
        return state

    document_ids = list(dict.fromkeys(state.get("document_ids") or []))
    if not document_ids:
        return state

    documents = [get_document(document_id) for document_id in document_ids]
    if any(not document or document.get("status") != "completed" for document in documents):
        return state

    expected_count = sum(int(document.get("chunk_count") or 0) for document in documents)
    if not expected_count or expected_count > DOCUMENT_CONTEXT_QUERY_LIMIT:
        return state

    known_sizes = [document.get("content_chars") for document in documents]
    if all(isinstance(value, int) and value >= 0 for value in known_sizes):
        if sum(known_sizes) > retrieval_config.direct_document_max_chars:
            return state

    try:
        rows = _load_document_chunks(document_ids, expected_count)
    except Exception as exc:
        logger.warning(f"读取完整文档上下文失败，降级到常规检索：{exc}")
        return state

    if len(rows) < expected_count:
        logger.warning(
            f"完整文档切片数量不一致，降级到常规检索：预期{expected_count}，实际{len(rows)}"
        )
        return state

    content_chars = sum(len(str(row.get("content") or "")) for row in rows)
    rows_by_document = {
        document_id: [row for row in rows if row.get("document_id") == document_id]
        for document_id in document_ids
    }
    document_by_id = {document["document_id"]: document for document in documents}
    for document_id, document_rows in rows_by_document.items():
        measured_chars = sum(len(str(row.get("content") or "")) for row in document_rows)
        metadata = document_by_id[document_id]
        if (
            int(metadata.get("chunk_count") or 0) == len(document_rows)
            and metadata.get("content_chars") == measured_chars
        ):
            continue
        try:
            update_document(
                document_id,
                chunk_count=len(document_rows),
                content_chars=measured_chars,
            )
        except Exception as exc:
            logger.warning(f"回写文档正文统计失败，不影响本轮问答：{exc}")

    if content_chars > retrieval_config.direct_document_max_chars:
        return state

    document_order = {document_id: index for index, document_id in enumerate(document_ids)}
    rows.sort(
        key=lambda row: (
            document_order.get(row.get("document_id"), len(document_order)),
            int(row.get("chunk_index") or row.get("chunk_id") or 0),
        )
    )
    state["reranked_docs"] = [_as_context_document(row) for row in rows if row.get("content")]
    state["evidence_quality"] = "full_context"
    state["document_context_complete"] = bool(state["reranked_docs"])
    if state["document_context_complete"]:
        logger.info(
            f"完整文档上下文已就绪：{len(state['reranked_docs'])}个切片，{content_chars}字符"
        )
    return state


@node_log("node_document_context")
def node_document_context(state):
    run_id = state.get("run_id") or state["session_id"]
    node_name = sys._getframe().f_code.co_name
    add_running_task(run_id, node_name, state.get("is_stream", False))
    prepare_document_context(state)
    add_done_task(run_id, node_name, state.get("is_stream", False))
    return state
