import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_done_task, add_running_task

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE_TOKENS", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP_TOKENS", "80"))
DENSE_SPEC_GROUP_LINES = int(os.getenv("DENSE_SPEC_GROUP_LINES", "5"))
DENSE_SPEC_OVERLAP_LINES = int(os.getenv("DENSE_SPEC_OVERLAP_LINES", "1"))


def estimate_token_count(text: str) -> int:
    """Approximate multilingual token count without loading a model tokenizer."""
    cjk_count = len(re.findall(r"[\u3400-\u9fff]", text))
    latin_word_count = len(re.findall(r"[A-Za-z0-9_]+", text))
    remaining = re.sub(r"[\u3400-\u9fffA-Za-z0-9_\s]", "", text)
    return cjk_count + latin_word_count + math.ceil(len(remaining) / 4)


def split_dense_spec_lines(content: str) -> List[str]:
    """Split dense key/value specification lists while preserving their heading."""
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) < 7:
        return [content]

    heading = lines[0] if re.match(r"^#{1,6}\s+", lines[0]) else ""
    body_lines = lines[1:] if heading else lines
    if len(body_lines) < 6:
        return [content]

    spec_lines = [
        line for line in body_lines
        if not re.match(r"^(?:[-*+]\s+|\d+[.)、]\s*)", line)
        and not line.startswith("<table")
        and len(line) <= 180
        and re.search(r"\S\s+\S", line)
    ]
    if len(spec_lines) / len(body_lines) < 0.75:
        return [content]

    group_size = max(2, DENSE_SPEC_GROUP_LINES)
    overlap = min(max(0, DENSE_SPEC_OVERLAP_LINES), group_size - 1)
    step = group_size - overlap
    groups = []
    for start in range(0, len(body_lines), step):
        selected = body_lines[start:start + group_size]
        if not selected:
            continue
        group_lines = ([heading] if heading else []) + selected
        groups.append("\n\n".join(group_lines))
        if start + group_size >= len(body_lines):
            break
    return groups if len(groups) > 1 else [content]


@step_log("step_1")
def step_1(state: ImportGraphState) -> Tuple[str, str]:
    md_content = state["md_content"]
    file_title = state["file_title"]
    md_path = state["md_path"]

    if not md_content:
        logger.warning("未从 state 读取到 md_content，尝试从 md_path 重新读取")
        if md_path:
            md_content = Path(md_path).read_text(encoding="utf-8")
            state["md_content"] = md_content
        if not md_content:
            raise ValueError("md_content 为空，且无法通过 md_path 读取到内容")

    if not file_title:
        logger.warning("未从 state 读取到 file_title，尝试自动补全")
        if md_path:
            file_title = Path(md_path).stem
        if not file_title:
            file_title = "default"
        state["file_title"] = file_title

    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
    state["md_content"] = md_content
    return md_content, file_title


@step_log("step_2")
def step_2(md_content: str, file_title: str) -> List[Dict[str, str]]:
    """
    第一次按 Markdown 标题做语义切分。
    如果没有识别到标题，则整篇文档作为一个块返回。
    """
    title_pattern = re.compile(r"^\s*#{1,6}\s+.+")
    lines = md_content.split("\n")

    chunks: List[Dict[str, str]] = []
    current_title: str | None = None
    current_lines: List[str] = []
    is_code_block = False

    for line in lines:
        if line.startswith("```") or line.startswith("~~~"):
            is_code_block = not is_code_block

        if title_pattern.match(line) and not is_code_block:
            if current_lines:
                chunks.append(
                    {
                        "content": "\n".join(current_lines).strip(),
                        "title": current_title or "default",
                        "file_title": file_title,
                    }
                )
            current_title = line.strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines and len(current_lines) > 1:
        chunks.append(
            {
                "content": "\n".join(current_lines).strip(),
                "title": current_title or "default",
                "file_title": file_title,
            }
        )

    if not chunks:
        chunks.append(
            {
                "content": md_content,
                "title": "default",
                "file_title": file_title,
            }
        )

    logger.info(f"语义切分完成，识别到 {len(chunks)} 个切分块")
    return chunks


@step_log("step_3")
def step_3(chunks: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    第二次切分：把过长的语义块拆成更小的片段。
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n", "</table>", "</tr>", "\n", "。", "；", ". ", "! ", "? ",
            "，", ";", " ", "",
        ],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=estimate_token_count,
        keep_separator=True,
    )

    final_chunks: List[Dict[str, Any]] = []

    for chunk in chunks:
        content = chunk["content"]
        title = chunk["title"]
        file_title = chunk["file_title"]

        dense_spec_chunks = split_dense_spec_lines(content)
        if len(dense_spec_chunks) > 1:
            for index, split_content in enumerate(dense_spec_chunks, start=1):
                final_chunks.append(
                    {
                        "content": split_content,
                        "title": f"{title}_{index}",
                        "parent_title": title,
                        "part": index,
                        "file_title": file_title,
                        "token_count": estimate_token_count(split_content),
                    }
                )
            continue

        if estimate_token_count(content) <= CHUNK_SIZE:
            final_chunks.append(
                {
                    "content": content,
                    "title": title,
                    "parent_title": title,
                    "part": 1,
                    "file_title": file_title,
                    "token_count": estimate_token_count(content),
                }
            )
            continue

        split_contents = splitter.split_text(content)
        for index, split_content in enumerate(split_contents, start=1):
            if not split_content.strip():
                continue
            if title != "default" and not split_content.lstrip().startswith(title):
                split_content = f"{title}\n\n{split_content}"
            final_chunks.append(
                {
                    "content": split_content,
                    "title": f"{title}_{index}",
                    "parent_title": title,
                    "part": index,
                    "file_title": file_title,
                    "token_count": estimate_token_count(split_content),
                }
            )

    return final_chunks


@step_log("step_5")
def step_5(chunks: List[Dict[str, Any]], path: str) -> None:
    """
    切分结果保存到本地 chunks.json。
    """
    chunk_json_path_obj = Path(path).parent / "chunks.json"
    chunk_json_path_obj.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )


@node_log("node_document_split")
def node_document_split(state: ImportGraphState) -> ImportGraphState:
    add_running_task(state["task_id"], "node_document_split")
    md_content, file_title = step_1(state)
    chunks = step_2(md_content, file_title)
    chunks = step_3(chunks)
    for chunk_index, chunk in enumerate(chunks, start=1):
        chunk["kb_id"] = state.get("kb_id", "")
        chunk["document_id"] = state.get("document_id", "")
        chunk["chunk_index"] = chunk_index
    step_5(chunks, state["md_path"])
    state["chunks"] = chunks
    add_done_task(state["task_id"], "node_document_split")
    return state
