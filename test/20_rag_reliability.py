import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.import_process.agent.api import server as import_server
from app.query_process.agent.nodes import node_document_summary as summary_module
from app.query_process.agent.nodes import node_item_name_confirm as item_name_module
from app.query_process.agent.nodes import node_search_embedding as search_module
from app.query_process.agent.main_graph import router
from app.query_process.agent.nodes.node_document_summary import is_document_summary_request
from app.query_process.agent.nodes.node_rerank import step_4_chunk_topk
from app.query_process.agent.retrieval_utils import search_chunks
from app.query_process.agent.source_utils import (
    build_source_records,
    compact_citations,
    deduplicate_documents,
    select_cited_sources,
)
from app.query_process.api import server as query_server


def test_low_relevance_chunks_are_rejected():
    chunks = [
        {"title": "无关资料", "text": "不相关内容", "score": 0.01},
        {"title": "仍然无关", "text": "其他内容", "score": 0.08},
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


def test_deep_explanation_route_is_document_type_agnostic():
    assert is_document_summary_request(
        {
            "original_query": "请详细讲解这份设备手册",
            "query_mode": "explain",
            "query_depth": "deep",
            "document_ids": ["manual-1"],
        }
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
        patch.object(summary_module, "step_2_direct_synthesis", return_value="整份摘要。[1][2]"),
        patch.object(summary_module, "add_running_task"),
        patch.object(summary_module, "add_done_task"),
    ):
        result = summary_module.node_document_summary(state)

    assert result["answer"] == "整份摘要。[1][2]"
    assert [source["index"] for source in result["sources"]] == [1, 2]
    assert not is_document_summary_request(
        {"original_query": "总结这一段是什么意思", "rewritten_query": ""}
    )


def test_scope_modes_are_resolved_explicitly():
    knowledge_bases = [{"kb_id": "kb-1"}, {"kb_id": "kb-2"}]
    documents = {
        "doc-1": {"document_id": "doc-1", "kb_id": "kb-2", "status": "completed"}
    }
    with (
        patch.object(query_server, "list_knowledge_bases", return_value=knowledge_bases),
        patch.object(query_server, "get_document", side_effect=documents.get),
    ):
        assert query_server.resolve_query_scope(
            query_server.QueryRequest(query="总结资料", scope_mode="all")
        ) == (["kb-1", "kb-2"], [])
        assert query_server.resolve_query_scope(
            query_server.QueryRequest(
                query="查询手册", scope_mode="knowledge_base", kb_ids=["kb-1"]
            )
        ) == (["kb-1"], [])
        assert query_server.resolve_query_scope(
            query_server.QueryRequest(
                query="详细讲解", scope_mode="documents", document_ids=["doc-1"]
            )
        ) == (["kb-2"], ["doc-1"])


def test_web_search_is_disabled_when_flag_is_absent():
    routes = router({"answer": "", "query_mode": "lookup"})
    assert "node_web_search_mcp" not in routes


def test_follow_up_inherits_recent_document_scope():
    history = [
        {
            "role": "assistant",
            "text": "上一轮回答",
            "sources": [
                {"document_id": "doc-1"},
                {"document_id": "doc-1"},
                {"document_id": "doc-2"},
            ],
        }
    ]

    assert item_name_module._recent_document_ids(history, "继续详细一点") == [
        "doc-1",
        "doc-2",
    ]
    assert item_name_module._recent_document_ids(history, "查询另一份手册") == []


def test_document_scope_builds_milvus_filter():
    with (
        patch(
            "app.query_process.agent.retrieval_utils.create_hybrid_search_requests",
            return_value=[],
        ) as create_requests,
        patch(
            "app.query_process.agent.retrieval_utils.hybrid_search",
            return_value=[[]],
        ),
    ):
        search_chunks(
            MagicMock(),
            [0.1],
            {1: 0.2},
            [],
            ["kb-1"],
            ["doc-1", "doc-2"],
        )

    assert create_requests.call_args.kwargs["expr"] == 'document_id in ["doc-1", "doc-2"]'


def test_deep_aspects_expand_document_search_and_merge_results():
    state = {
        "session_id": "session-1",
        "run_id": "run-1",
        "rewritten_query": "详细讲解设备手册",
        "item_names": [],
        "kb_ids": ["kb-1"],
        "document_ids": ["doc-1"],
        "query_depth": "deep",
        "query_aspects": ["安装", "故障处理"],
        "is_stream": False,
    }
    facet_embeddings = {
        "dense": [[0.2], [0.3]],
        "sparse": [{2: 0.2}, {3: 0.3}],
    }
    search_results = [
        [{"id": "base"}],
        [{"id": "base"}, {"id": "install"}],
        [{"id": "troubleshooting"}],
    ]

    with (
        patch.object(search_module, "step_2_rewritten_query_embedding", return_value=([0.1], {1: 0.1})),
        patch.object(search_module, "generate_embeddings", return_value=facet_embeddings) as embeddings,
        patch.object(search_module, "step_3_milvus_hybrid_search", side_effect=search_results) as search,
        patch.object(search_module, "add_running_task"),
        patch.object(search_module, "add_done_task"),
    ):
        result = search_module.node_search_embedding(state)

    embeddings.assert_called_once_with(
        [
            "详细讲解设备手册；重点检索：安装",
            "详细讲解设备手册；重点检索：故障处理",
        ]
    )
    assert search.call_count == 3
    assert all(call.args[-1] == ["doc-1"] for call in search.call_args_list)
    assert [hit["id"] for hit in result["embedding_chunks"]] == [
        "base",
        "install",
        "troubleshooting",
    ]


def test_cross_source_duplicates_prefer_local_and_citations_are_compact():
    abstract = "同一份文档摘要" * 80
    documents = [
        {
            "title": "示例文档",
            "file_title": "示例文档",
            "text": abstract,
            "score": 0.99,
            "type": "web",
            "url": "https://example.com/demo",
        },
        {
            "title": "摘要",
            "file_title": "示例文档",
            "text": abstract,
            "score": 0.92,
            "type": "milvus",
            "document_id": "doc-1",
        },
        {
            "title": "方法",
            "file_title": "示例文档",
            "text": "不同章节的有效证据",
            "score": 0.88,
            "type": "milvus",
            "document_id": "doc-1",
        },
    ]
    deduplicated = deduplicate_documents(documents)
    assert len(deduplicated) == 2
    assert deduplicated[0]["type"] == "milvus"
    candidates = build_source_records(deduplicated)
    answer, sources = compact_citations("方法结论。[2] 摘要结论。[1]", candidates)
    assert answer == "方法结论。[1] 摘要结论。[2]"
    assert [source["index"] for source in sources] == [1, 2]


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
        test_deep_explanation_route_is_document_type_agnostic,
        test_document_summary_preserves_cited_sources,
        test_scope_modes_are_resolved_explicitly,
        test_web_search_is_disabled_when_flag_is_absent,
        test_follow_up_inherits_recent_document_scope,
        test_document_scope_builds_milvus_filter,
        test_deep_aspects_expand_document_search_and_merge_results,
        test_cross_source_duplicates_prefer_local_and_citations_are_compact,
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
