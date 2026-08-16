import re
from typing import Any, Dict, Iterable, List


def _normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def deduplicate_documents(documents: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按来源位置和正文去重，避免重复导入挤占 Top-K 与引用列表。"""
    result: List[Dict[str, Any]] = []
    seen = set()
    for document in documents:
        content = _normalized_text(document.get("text") or document.get("content"))
        if document.get("type") == "web":
            key = ("web", document.get("url") or "", content)
        else:
            key = (
                "local",
                _normalized_text(document.get("file_title")),
                _normalized_text(document.get("parent_title")),
                content,
            )
        if not content or key in seen:
            continue
        seen.add(key)
        result.append(dict(document))
    return result


def build_source_records(documents: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sources = []
    for index, chunk in enumerate(deduplicate_documents(documents), start=1):
        sources.append(
            {
                "index": index,
                "title": chunk.get("title") or "未命名来源",
                "file_title": chunk.get("file_title") or chunk.get("title") or "",
                "parent_title": chunk.get("parent_title") or "",
                "content": chunk.get("text") or chunk.get("content") or "",
                "score": chunk.get("score"),
                "search_rank": chunk.get("search_rank"),
                "type": chunk.get("type", "milvus"),
                "url": chunk.get("url"),
                "kb_id": chunk.get("kb_id"),
                "document_id": chunk.get("document_id"),
                "part": chunk.get("part"),
            }
        )
    return sources


def select_cited_sources(answer: str, candidates: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """只返回答案中以 [n] 明确引用的来源，保持首次出现顺序。"""
    cited_numbers = []
    for value in re.findall(r"\[(\d+)]", answer or ""):
        number = int(value)
        if number not in cited_numbers:
            cited_numbers.append(number)
    source_by_index = {int(source["index"]): source for source in candidates}
    return [source_by_index[number] for number in cited_numbers if number in source_by_index]
