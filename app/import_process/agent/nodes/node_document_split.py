import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_done_task, add_running_task

# 单个语义块的最大长度。超过后再做二次切分。
CHUNK_MAX_SIZE = 500
CHUNK_SIZE = 200
CHUNK_OVERLAP = 20


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
        separators=["\n\n", "\n", "。", "；", "，", ".", "!", "?", ";", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    final_chunks: List[Dict[str, Any]] = []

    for chunk in chunks:
        content = chunk["content"]
        title = chunk["title"]
        file_title = chunk["file_title"]

        if len(content) <= CHUNK_MAX_SIZE:
            final_chunks.append(
                {
                    "content": content,
                    "title": title,
                    "parent_title": title,
                    "part": 1,
                    "file_title": file_title,
                }
            )
            continue

        split_contents = splitter.split_text(content)
        for index, split_content in enumerate(split_contents, start=1):
            final_chunks.append(
                {
                    "content": split_content,
                    "title": f"{title}_{index}",
                    "parent_title": title,
                    "part": index,
                    "file_title": file_title,
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
    for chunk in chunks:
        chunk["kb_id"] = state.get("kb_id", "")
        chunk["document_id"] = state.get("document_id", "")
    step_5(chunks, state["md_path"])
    state["chunks"] = chunks
    add_done_task(state["task_id"], "node_document_split")
    return state
