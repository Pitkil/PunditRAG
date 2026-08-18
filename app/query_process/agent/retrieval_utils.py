import json
from typing import Any, Iterable

from app.clients.milvus_utils import create_hybrid_search_requests, hybrid_search
from app.conf.milvus_config import milvus_config
from app.conf.retrieval_config import retrieval_config
from app.core.logger import logger


CHUNK_OUTPUT_FIELDS = [
    "chunk_id", "item_name", "content", "title", "parent_title", "part",
    "file_title", "kb_id", "document_id",
]


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
