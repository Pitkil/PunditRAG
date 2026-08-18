import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.import_process.agent.api import server as import_server
from app.query_process.agent.nodes import node_document_summary as summary_module
from app.query_process.agent.nodes import node_document_context as context_module
from app.query_process.agent.nodes import node_item_name_confirm as item_name_module
from app.query_process.agent.nodes import node_search_embedding as search_module
from app.query_process.agent.nodes import node_search_embedding_hyde as hyde_module
from app.query_process.agent.main_graph import router
from app.query_process.agent.nodes.node_document_summary import is_document_summary_request
from app.query_process.agent.nodes.node_rerank import step_4_chunk_topk
from app.query_process.agent.retrieval_utils import (
    build_retrieval_query,
    expand_reranked_neighbors,
    search_chunks,
)
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
    assert is_document_summary_request({"full_document": True})
    assert not is_document_summary_request({"full_document": False})


def test_deep_explanation_uses_retrieval_instead_of_summary():
    state = {
        "answer": "",
        "original_query": "请详细讲解这份设备手册",
        "full_document": False,
        "document_ids": ["manual-1"],
        "item_names": ["设备手册"],
        "enable_web_search": False,
    }

    assert not is_document_summary_request(state)
    assert router(state) == ("node_search_embedding", "node_search_embedding_hyde")


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


def test_document_summary_streams_final_synthesis():
    state = {"session_id": "session-1", "run_id": "run-1", "is_stream": True}
    fake_llm = MagicMock()
    fake_llm.stream.return_value = [
        MagicMock(content="整篇"),
        MagicMock(content="摘要。[1]"),
    ]
    with (
        patch.object(summary_module, "get_llm_client", return_value=fake_llm),
        patch.object(summary_module, "push_to_session") as push,
    ):
        answer = summary_module._invoke_text("prompt", state)

    assert answer == "整篇摘要。[1]"
    assert state["answer_streamed"] is True
    assert "".join(call.args[2]["delta"] for call in push.call_args_list) == answer


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
    routes = router({"answer": "", "full_document": False})
    assert "node_web_search_mcp" not in routes


def test_legacy_mode_fields_do_not_control_retrieval():
    state = {
        "original_query": "详细讲解论文",
        "kb_ids": ["kb-paper"],
        "document_ids": [],
    }
    query_plan = {
        "mode": "clarify",
        "depth": "deep",
        "aspects": [],
    }

    item_name_module.step_6_deal_state(
        state,
        {"confirmed_item_name_list": [], "options_item_name_list": []},
        query_plan,
    )

    assert state["answer"] == ""
    assert state["rewritten_query"] == state["original_query"]
    assert state["full_document"] is False
    assert router(state) == ("node_search_embedding", "node_search_embedding_hyde")


def test_full_document_is_the_only_planned_execution_strategy():
    state = {
        "original_query": "详细讲解这个论文",
        "scope_mode": "documents",
        "kb_ids": ["kb-paper"],
        "document_ids": ["doc-paper"],
    }
    empty_match = {"confirmed_item_name_list": [], "options_item_name_list": []}
    item_name_module.step_6_deal_state(
        state,
        empty_match,
        {"item_names": [], "full_document": False},
    )
    assert router(state) == ("node_search_embedding", "node_search_embedding_hyde")

    state["original_query"] = "总结整篇论文"
    item_name_module.step_6_deal_state(
        state,
        empty_match,
        {"item_names": [], "full_document": True},
    )
    assert router(state) == "node_document_summary"


def test_short_explicit_document_uses_complete_context():
    state = {
        "answer": "",
        "full_document": False,
        "enable_web_search": False,
        "document_ids": ["doc-1"],
    }
    rows = [
        {
            "chunk_id": 12,
            "chunk_index": 2,
            "document_id": "doc-1",
            "content": "第二段正文",
            "parent_title": "方法",
            "part": 1,
        },
        {
            "chunk_id": 11,
            "chunk_index": 1,
            "document_id": "doc-1",
            "content": "第一段正文",
            "parent_title": "摘要",
            "part": 1,
        },
    ]
    with (
        patch.object(
            context_module,
            "get_document",
            return_value={
                "document_id": "doc-1",
                "status": "completed",
                "chunk_count": 2,
                "content_chars": 10,
            },
        ),
        patch.object(context_module, "_load_document_chunks", return_value=rows),
        patch.object(context_module, "update_document"),
    ):
        result = context_module.prepare_document_context(state)

    assert result["document_context_complete"] is True
    assert result["evidence_quality"] == "full_context"
    assert [item["text"] for item in result["reranked_docs"]] == ["第一段正文", "第二段正文"]
    assert router(result) == "node_answer_output"


def test_long_document_and_summary_keep_existing_routes():
    long_state = {
        "answer": "",
        "full_document": False,
        "enable_web_search": False,
        "document_ids": ["doc-long"],
    }
    with (
        patch.object(
            context_module,
            "get_document",
            return_value={
                "document_id": "doc-long",
                "status": "completed",
                "chunk_count": 100,
                "content_chars": context_module.retrieval_config.direct_document_max_chars + 1,
            },
        ),
        patch.object(context_module, "_load_document_chunks") as load_chunks,
    ):
        result = context_module.prepare_document_context(long_state)

    assert result["document_context_complete"] is False
    assert router(result) == ("node_search_embedding", "node_search_embedding_hyde")
    load_chunks.assert_not_called()

    summary_state = {
        "answer": "",
        "full_document": True,
        "enable_web_search": False,
        "document_ids": ["doc-short"],
    }
    with patch.object(context_module, "get_document") as get_document:
        result = context_module.prepare_document_context(summary_state)
    assert router(result) == "node_document_summary"
    get_document.assert_not_called()


def test_reranked_anchor_expands_only_same_section_neighbors():
    anchor = {
        "type": "milvus",
        "document_id": "doc-1",
        "parent_title": "实验结果",
        "part": 2,
        "text": "锚点正文",
        "score": 0.9,
    }
    client = MagicMock()
    client.query.return_value = [
        {"document_id": "doc-1", "parent_title": "实验结果", "part": 1, "content": "前文"},
        {"document_id": "doc-1", "parent_title": "实验结果", "part": 2, "content": "锚点正文"},
        {"document_id": "doc-1", "parent_title": "实验结果", "part": 3, "content": "后文"},
        {"document_id": "doc-1", "parent_title": "参考文献", "part": 1, "content": "不应加入"},
    ]

    result = expand_reranked_neighbors(client, [anchor], 1)

    assert [item["part"] for item in result] == [1, 2, 3]
    assert result[1] is anchor
    assert result[0]["expanded_context"] is True
    assert result[2]["expanded_context"] is True
    assert all(item["parent_title"] == "实验结果" for item in result)


def test_query_planner_failure_falls_back_to_original_question():
    with patch.object(item_name_module, "get_llm_client", side_effect=TimeoutError("timeout")):
        result = item_name_module.step_3_llm_itemnames_and_rewrite(
            [],
            "详细讲解这个论文",
            "已选择论文",
        )

    assert result == {
        "rewritten_query": "详细讲解这个论文",
        "item_names": [],
        "full_document": False,
    }


def test_hyde_timeout_falls_back_without_blocking_normal_search():
    state = {
        "session_id": "session-1",
        "run_id": "run-1",
        "original_query": "详细讲解论文",
        "item_names": [],
        "kb_ids": ["kb-paper"],
        "document_ids": [],
        "is_stream": False,
    }
    with (
        patch.object(hyde_module, "step_2_call_llm", return_value=""),
        patch.object(hyde_module, "step_3_rewritten_hyde_vector") as vectorize,
        patch.object(hyde_module, "add_running_task"),
        patch.object(hyde_module, "add_done_task"),
    ):
        result = hyde_module.node_search_embedding_hyde(state)

    assert result == {"hyde_embedding_chunks": []}
    vectorize.assert_not_called()


def test_history_does_not_inherit_document_scope_or_replace_current_query():
    history = [
        {
            "role": "assistant",
            "text": "上一轮回答",
            "sources": [
                {"document_id": "doc-1", "kb_id": "kb-1"},
                {"document_id": "doc-1", "kb_id": "kb-1"},
                {"document_id": "doc-2", "kb_id": "kb-2"},
            ],
        }
    ]
    state = {
        "session_id": "new-session",
        "run_id": "new-run",
        "original_query": "你觉得男主是什么样的人",
        "kb_ids": ["kb-1"],
        "document_ids": [],
        "is_stream": False,
    }
    plan = {
        "rewritten_query": "选择一个更值得偏好的角色",
        "item_names": [],
        "full_document": False,
    }
    with (
        patch.object(item_name_module, "step_2_chat_history", return_value=history),
        patch.object(item_name_module, "build_scope_context", return_value="显式知识库范围"),
        patch.object(item_name_module, "step_3_llm_itemnames_and_rewrite", return_value=plan) as planner,
        patch.object(item_name_module, "step_4_vector_query_item_name") as topic_search,
        patch.object(item_name_module, "step_7_save_user_chat_message"),
        patch.object(item_name_module, "add_running_task"),
        patch.object(item_name_module, "add_done_task"),
    ):
        result = item_name_module.node_item_name_confirm(state)

    assert result["rewritten_query"] == state["original_query"]
    assert result["full_document"] is False
    assert result["item_names"] == []
    assert result["document_ids"] == []
    assert result["history"] == history
    planner.assert_called_once_with(history, state["original_query"], "显式知识库范围")
    topic_search.assert_not_called()


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


def test_retrieval_query_keeps_original_question_and_adds_explicit_document_name():
    state = {
        "original_query": "详细讲解这个论文",
        "scope_document_names": ["Collaborative Conversational Agent.pdf"],
        "item_names": ["不应替换原问题"],
    }
    query = build_retrieval_query(state, "研究方法")
    assert query.startswith(state["original_query"])
    assert "Collaborative Conversational Agent.pdf" in query
    assert "研究方法" in query
    assert "不应替换原问题" not in query


def test_all_regular_questions_use_one_vector_search():
    state = {
        "session_id": "session-1",
        "run_id": "run-1",
        "original_query": "详细讲解设备手册",
        "rewritten_query": "错误改写不得进入检索",
        "item_names": [],
        "kb_ids": ["kb-1"],
        "document_ids": ["doc-1"],
        "is_stream": False,
    }

    with (
        patch.object(search_module, "step_2_rewritten_query_embedding", return_value=([0.1], {1: 0.1})),
        patch.object(search_module, "step_3_milvus_hybrid_search", return_value=[{"id": "base"}]) as search,
        patch.object(search_module, "add_running_task"),
        patch.object(search_module, "add_done_task"),
    ):
        result = search_module.node_search_embedding(state)

    search.assert_called_once_with([0.1], {1: 0.1}, [], ["kb-1"], ["doc-1"])
    assert result["embedding_chunks"] == [{"id": "base"}]


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
        test_deep_explanation_uses_retrieval_instead_of_summary,
        test_document_summary_preserves_cited_sources,
        test_document_summary_streams_final_synthesis,
        test_scope_modes_are_resolved_explicitly,
        test_web_search_is_disabled_when_flag_is_absent,
        test_legacy_mode_fields_do_not_control_retrieval,
        test_full_document_is_the_only_planned_execution_strategy,
        test_short_explicit_document_uses_complete_context,
        test_long_document_and_summary_keep_existing_routes,
        test_reranked_anchor_expands_only_same_section_neighbors,
        test_query_planner_failure_falls_back_to_original_question,
        test_hyde_timeout_falls_back_without_blocking_normal_search,
        test_history_does_not_inherit_document_scope_or_replace_current_query,
        test_document_scope_builds_milvus_filter,
        test_retrieval_query_keeps_original_question_and_adds_explicit_document_name,
        test_all_regular_questions_use_one_vector_search,
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
