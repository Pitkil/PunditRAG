import sys
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.import_process.agent.api import server as import_server
from app.query_process.agent.nodes import node_document_summary as summary_module
from app.query_process.agent.nodes.node_document_summary import is_document_summary_request
from app.query_process.agent.nodes.node_rerank import step_4_chunk_topk
from app.query_process.agent.source_utils import build_source_records, select_cited_sources
from app.query_process.api import server as query_server


def test_low_relevance_chunks_are_rejected():
    chunks = [
        {"title": "无关资料", "text": "不相关内容", "score": 0.01},
        {"title": "仍然无关", "text": "其他内容", "score": 0.14},
    ]
    assert step_4_chunk_topk(chunks) == []


def test_sources_are_deduplicated_and_claim_cited():
    documents = [
        {"title": "第一章", "file_title": "教材.pdf", "text": "相同证据", "score": 0.9},
        {"title": "第一章副本", "file_title": "教材.pdf", "text": "相同证据", "score": 0.8},
        {"title": "第二章", "file_title": "教材.pdf", "text": "另一条证据", "score": 0.7},
    ]
    candidates = build_source_records(documents)
    sources = select_cited_sources("结论来自第二条证据。[2]", candidates)

    assert len(candidates) == 2
    assert [source["index"] for source in sources] == [2]
    assert sources[0]["title"] == "第二章"


def test_summary_route_requires_whole_document_intent():
    assert is_document_summary_request(
        {"original_query": "请总结整份资料", "rewritten_query": ""}
    )


def test_document_summary_preserves_cited_sources():
    state = {
        "session_id": "session-1",
        "run_id": "run-1",
        "original_query": "请总结整份资料",
        "rewritten_query": "请总结整份资料",
        "is_stream": False,
    }
    documents = [
        {
            "title": "第一章",
            "file_title": "教材.pdf",
            "content": "第一章原文",
            "text": "第一章原文",
            "type": "milvus",
        },
        {
            "title": "第二章",
            "file_title": "教材.pdf",
            "content": "第二章原文",
            "text": "第二章原文",
            "type": "milvus",
        },
    ]
    with (
        patch.object(summary_module, "step_1_load_summary_chunks", return_value=documents),
        patch.object(summary_module, "step_2_map_summary", return_value=["局部摘要[1]", "局部摘要[2]"]),
        patch.object(summary_module, "step_3_reduce_summary", return_value="整份摘要。[1][2]"),
        patch.object(summary_module, "add_running_task"),
        patch.object(summary_module, "add_done_task"),
    ):
        result = summary_module.node_document_summary(state)

    assert result["answer"] == "整份摘要。[1][2]"
    assert [source["index"] for source in result["sources"]] == [1, 2]
    assert not is_document_summary_request(
        {"original_query": "总结这一段是什么意思", "rewritten_query": ""}
    )


def test_document_delete_cleans_all_storage_layers():
    document = {"document_id": "doc-1", "local_path": "demo.pdf"}
    with (
        patch.object(import_server, "get_document", return_value=document),
        patch.object(import_server, "update_document") as update_document,
        patch.object(import_server, "_delete_vectors") as delete_vectors,
        patch.object(import_server, "_delete_minio_artifacts") as delete_minio,
        patch.object(import_server, "_delete_local_artifacts") as delete_local,
        patch.object(import_server, "delete_document_record") as delete_record,
    ):
        result = import_server.remove_document("doc-1")

    assert result == {"deleted": True, "document_id": "doc-1"}
    update_document.assert_called_once_with("doc-1", status="deleting")
    delete_vectors.assert_called_once_with("doc-1")
    delete_minio.assert_called_once_with("doc-1")
    delete_local.assert_called_once_with(document)
    delete_record.assert_called_once_with("doc-1")


def test_running_session_cannot_be_deleted():
    query_server._register_run("session-1", "run-1")
    try:
        with patch.object(query_server, "delete_chat_session") as delete_session:
            try:
                query_server.remove_session("session-1")
            except HTTPException as exc:
                assert exc.status_code == 409
            else:
                raise AssertionError("正在运行的会话不应允许删除")
            delete_session.assert_not_called()
    finally:
        query_server._finish_run("session-1", "run-1")


if __name__ == "__main__":
    tests = [
        test_low_relevance_chunks_are_rejected,
        test_sources_are_deduplicated_and_claim_cited,
        test_summary_route_requires_whole_document_intent,
        test_document_summary_preserves_cited_sources,
        test_document_delete_cleans_all_storage_layers,
        test_running_session_cannot_be_deleted,
    ]
    failures = []
    for test_function in tests:
        try:
            test_function()
            print(f"[PASS] {test_function.__name__}")
        except Exception as exc:
            failures.append((test_function.__name__, exc))
            print(f"[FAIL] {test_function.__name__}: {exc}")
    if failures:
        sys.exit(1)
