import json
from typing import Any, Iterable

from app.clients.milvus_utils import create_hybrid_search_requests, hybrid_search
from app.conf.milvus_config import milvus_config
from app.conf.retrieval_config import retrieval_config
from app.core.logger import logger


CHUNK_OUTPUT_FIELDS = [
    "chunk_id", "item_name", "content", "title", "parent_title", "part",
    "chunk_index", "file_title", "kb_id", "document_id",
]


def build_retrieval_query(state, aspect: str = "") -> str:
    """保留用户原问题，并用本轮显式范围补足“这个文档”等无主题指代。"""
    query = str(state.get("original_query") or "").strip()
    document_names = [
        str(name).strip()
        for name in state.get("scope_document_names") or []
        if str(name).strip()
    ]
    item_names = [
        str(name).strip()
        for name in state.get("item_names") or []
        if str(name).strip()
    ]
    parts = [query]
    if document_names:
        parts.append(f"当前所选文档：{'、'.join(document_names[:3])}")
    elif item_names:
        parts.append(f"相关主题：{'、'.join(item_names[:5])}")
    if aspect:
        parts.append(f"重点检索：{aspect}")
    return "；".join(part for part in parts if part)


def _hit_key(hit: dict[str, Any]) -> Any:
    entity = hit.get("entity") or {}
    return hit.get("id") or entity.get("chunk_id") or (
        entity.get("document_id"), entity.get("parent_title"), entity.get("part")
    )


def merge_unique_hits(*groups: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = []
    seen = set()
    for group in groups:
        for hit in group:
            key = _hit_key(hit)
            if key in seen:
                continue
            seen.add(key)
            merged.append(hit)
    return merged


def search_chunks(client, dense_vector, sparse_vector, item_names, kb_ids, document_ids=None):
    """Search an explicit KB scope and use topic matches only as recall expansion."""
    document_ids = document_ids or []
    if not kb_ids and not document_ids:
        logger.info("本轮未指定资料范围，跳过本地向量检索")
        return []

    scope_filter = (
        f"document_id in {json.dumps(document_ids, ensure_ascii=False)}"
        if document_ids
        else f"kb_id in {json.dumps(kb_ids, ensure_ascii=False)}"
    )
    broad_limit = retrieval_config.retrieval_top_k
    broad_requests = create_hybrid_search_requests(
        dense_vector,
        sparse_vector,
        expr=scope_filter,
        limit=broad_limit,
    )
    broad_response = hybrid_search(
        client=client,
        collection_name=milvus_config.chunks_collection,
        reqs=broad_requests,
        norm_score=True,
        limit=broad_limit,
        output_fields=CHUNK_OUTPUT_FIELDS,
    )
    broad_hits = list(broad_response[0]) if broad_response and broad_response[0] else []

    if not item_names:
        return broad_hits

    topic_limit = retrieval_config.topic_expansion_top_k
    topic_filter = (
        f"{scope_filter} and item_name in {json.dumps(item_names, ensure_ascii=False)}"
    )
    topic_requests = create_hybrid_search_requests(
        dense_vector,
        sparse_vector,
        expr=topic_filter,
        limit=topic_limit,
    )
    topic_response = hybrid_search(
        client=client,
        collection_name=milvus_config.chunks_collection,
        reqs=topic_requests,
        norm_score=True,
        limit=topic_limit,
        output_fields=CHUNK_OUTPUT_FIELDS,
    )
    topic_hits = list(topic_response[0]) if topic_response and topic_response[0] else []
    return merge_unique_hits(broad_hits, topic_hits)


def expand_reranked_neighbors(client, documents, window: int) -> list[dict[str, Any]]:
    """Add same-section neighbors around final local anchors without changing anchor ranking."""
    if window <= 0:
        return list(documents)

    local_anchors = [
        document
        for document in documents
        if document.get("type") != "web"
        and document.get("document_id")
        and document.get("parent_title")
        and isinstance(document.get("part"), int)
    ]
    if not local_anchors:
        return list(documents)

    groups = list(dict.fromkeys(
        (anchor["document_id"], anchor["parent_title"], int(anchor["part"]))
        for anchor in local_anchors
    ))
    expressions = [
        "("
        f"document_id == {json.dumps(document_id, ensure_ascii=False)} and "
        f"parent_title == {json.dumps(parent_title, ensure_ascii=False)} and "
        f"part >= {max(1, part - window)} and part <= {part + window}"
        ")"
        for document_id, parent_title, part in groups
    ]
    try:
        rows = client.query(
            collection_name=milvus_config.chunks_collection,
            filter=" or ".join(expressions),
            output_fields=CHUNK_OUTPUT_FIELDS,
            limit=max(1, len(groups) * (window * 2 + 1)),
        )
    except Exception as exc:
        logger.warning(f"相邻切片扩展失败，保留原重排结果：{exc}")
        return list(documents)

    def key(document):
        return (
            document.get("document_id"),
            document.get("parent_title"),
            int(document.get("part") or 0),
        )

    anchor_by_key = {key(anchor): anchor for anchor in local_anchors}
    row_by_key = {key(row): row for row in rows}
    expanded = []
    seen = set()
    for document in documents:
        if document.get("type") == "web" or key(document) not in anchor_by_key:
            identity = (document.get("type"),) + key(document)
            if identity not in seen:
                seen.add(identity)
                expanded.append(document)
            continue

        document_id, parent_title, part = key(document)
        for neighbor_part in range(max(1, part - window), part + window + 1):
            neighbor_key = (document_id, parent_title, neighbor_part)
            selected = anchor_by_key.get(neighbor_key)
            if selected is None:
                row = row_by_key.get(neighbor_key)
                if not row or not row.get("content"):
                    continue
                selected = {
                    **row,
                    "text": row.get("content", ""),
                    "type": "milvus",
                    "score": None,
                    "expanded_context": True,
                }
            identity = (selected.get("type"),) + neighbor_key
            if identity in seen:
                continue
            seen.add(identity)
            expanded.append(selected)
    return expanded
