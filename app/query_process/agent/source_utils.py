import re
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List


def _normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _comparison_text(value: Any) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(value or "").lower())


def _source_title(document: Dict[str, Any]) -> str:
    return _comparison_text(document.get("file_title") or document.get("title"))


def _near_duplicate(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    left_text = _comparison_text(left.get("text") or left.get("content"))
    right_text = _comparison_text(right.get("text") or right.get("content"))
    if not left_text or not right_text:
        return False
    if left_text == right_text:
        return True
    shorter, longer = sorted((left_text, right_text), key=len)
    if len(shorter) >= 120 and shorter in longer:
        return True
    same_title = _source_title(left) and _source_title(left) == _source_title(right)
    if not same_title or len(shorter) < 240:
        return False
    return SequenceMatcher(None, left_text, right_text, autojunk=False).ratio() >= 0.82


def deduplicate_documents(documents: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """去除精确和跨来源近似重复，重复时优先保留本地原文。"""
    result: List[Dict[str, Any]] = []
    for document in documents:
        content = _normalized_text(document.get("text") or document.get("content"))
        if not content:
            continue
        duplicate_index = next(
            (index for index, existing in enumerate(result) if _near_duplicate(existing, document)),
            None,
        )
        if duplicate_index is None:
            result.append(dict(document))
            continue
        existing = result[duplicate_index]
        if existing.get("type") == "web" and document.get("type") != "web":
            replacement = dict(document)
            existing_score = existing.get("score")
            replacement_score = replacement.get("score")
            if existing_score is not None and replacement_score is not None:
                replacement["score"] = max(float(existing_score), float(replacement_score))
            result[duplicate_index] = replacement
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


def extract_citation_numbers(answer: str) -> List[int]:
    """按首次出现顺序提取答案中的 [n] 引用编号。"""
    numbers = []
    for value in re.findall(r"\[(\d+)]", answer or ""):
        number = int(value)
        if number not in numbers:
            numbers.append(number)
    return numbers


def reject_invalid_citations(answer: str, candidates: Iterable[Dict[str, Any]]) -> str:
    """引用了本轮不存在的来源编号时拒绝整份答案，避免展示假引用。"""
    candidate_list = list(candidates)
    valid_indexes = {int(source["index"]) for source in candidate_list}
    cited_numbers = extract_citation_numbers(answer)
    if any(number not in valid_indexes for number in cited_numbers):
        return "当前资料中没有足够信息。"
    return answer


def select_cited_sources(answer: str, candidates: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """只返回答案中以 [n] 明确引用的来源，保持首次出现顺序。"""
    cited_numbers = extract_citation_numbers(answer)
    source_by_index = {int(source["index"]): source for source in candidates}
    return [source_by_index[number] for number in cited_numbers if number in source_by_index]


def compact_citations(answer: str, candidates: Iterable[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
    """按首次引用顺序压缩编号，使答案和来源面板保持连续。"""
    candidate_list = list(candidates)
    selected = select_cited_sources(answer, candidate_list)
    mapping = {int(source["index"]): index for index, source in enumerate(selected, start=1)}
    compacted_answer = re.sub(
        r"\[(\d+)]",
        lambda match: f"[{mapping[int(match.group(1))]}]" if int(match.group(1)) in mapping else match.group(0),
        answer or "",
    )
    compacted_sources = [{**source, "index": mapping[int(source["index"])]} for source in selected]
    return compacted_answer, compacted_sources
