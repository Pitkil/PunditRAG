import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.logger import logger
from app.import_process.agent.api import server as import_server
from app.query_process.agent.nodes import node_answer_output as answer_module
from app.query_process.agent.nodes import node_search_embedding as search_module
from app.query_process.agent.nodes import node_search_embedding_hyde as hyde_module
from app.query_process.agent import retrieval_utils
from app.query_process.api import server as query_server


def test_knowledge_base_filter():
    for module, function_name in (
        (search_module, "step_3_milvus_hybrid_search"),
        (hyde_module, "step_4_mivlus_hybrid_search"),
    ):
        with (
            patch.object(module, "get_milvus_client", return_value=MagicMock()),
            patch.object(module, "search_chunks", return_value=[]) as search_chunks,
        ):
            getattr(module, function_name)([0.1], {1: 0.2}, ["学习资料"], ["kb-test"])

        assert search_chunks.call_args.args[-3:] == (["学习资料"], ["kb-test"], [])


def test_topic_search_is_an_expansion_not_a_hard_filter():
    captured_expressions = []

    def fake_requests(dense, sparse, expr=None, **kwargs):
        captured_expressions.append(expr)
        return ["request"]

    with (
        patch.object(retrieval_utils, "create_hybrid_search_requests", side_effect=fake_requests),
        patch.object(retrieval_utils, "hybrid_search", return_value=[[]]),
    ):
        retrieval_utils.search_chunks(
            MagicMock(), [0.1], {1: 0.2}, ["学习资料"], ["kb-test"]
        )

    assert captured_expressions[0] == 'kb_id in ["kb-test"]'
    assert 'item_name in ["学习资料"]' in captured_expressions[1]


def test_answer_sources():
    state = {"answer": "根据资料可确认该结论。[1]"}
    answer_module.step_5_build_sources(
        state,
        [
            {
                "title": "第一章",
                "file_title": "教材.pdf",
                "text": "引用原文",
                "score": 0.88,
                "type": "milvus",
                "document_id": "document-1",
            }
        ],
    )
    assert state["sources"][0]["file_title"] == "教材.pdf"
    assert state["sources"][0]["score"] == 0.88


def test_workspace_api_contract():
    knowledge_bases = [
        {
            "kb_id": "kb-test",
            "name": "课程知识库",
            "description": "",
            "document_count": 0,
        }
    ]
    final_state = {
        "answer": "测试答案",
        "image_urls": [],
        "sources": [{"index": 1, "title": "资料", "content": "原文", "score": 0.9}],
    }

    with (
        patch.object(query_server, "list_chat_sessions", return_value=[]),
        patch.object(query_server, "count_chat_sessions", return_value=125),
        patch.object(query_server, "list_knowledge_bases", return_value=knowledge_bases),
        patch.object(
            query_server,
            "ensure_chat_session",
            return_value={"session_id": "session-1", "title": "新对话"},
        ),
        patch.object(query_server.query_app, "invoke", return_value=final_state) as invoke,
        patch.object(import_server, "list_knowledge_bases", return_value=knowledge_bases),
        patch.object(import_server, "create_document", return_value={"document_id": "document-1"}),
        patch.object(import_server, "invoke_import_graph"),
    ):
        query_client = TestClient(query_server.app)
        import_client = TestClient(import_server.app)

        sessions = query_client.get("/sessions?limit=50&offset=100")
        assert sessions.status_code == 200
        assert sessions.json() == {"items": [], "total": 125, "has_more": True}
        query_server.list_chat_sessions.assert_called_once_with(limit=50, offset=100)

        response = query_client.post(
            "/query",
            json={
                "query": "测试问题",
                "session_id": "session-1",
                "kb_ids": ["kb-test"],
                "is_stream": False,
                "enable_web_search": False,
            },
        )
        assert response.status_code == 200
        assert response.json()["sources"][0]["title"] == "资料"
        assert response.json()["run_id"] != "session-1"
        query_state = invoke.call_args.args[0]
        assert query_state["kb_ids"] == ["kb-test"]
        assert query_state["run_id"] == response.json()["run_id"]
        assert query_state["enable_web_search"] is False

        upload = import_client.post(
            "/upload",
            data={"kb_id": "kb-test"},
            files={"files": ("demo.md", b"# demo", "text/markdown")},
        )
        assert upload.status_code == 200
        assert upload.json()["document_ids"] == ["document-1"]


if __name__ == "__main__":
    tests = [
        test_knowledge_base_filter,
        test_topic_search_is_an_expansion_not_a_hard_filter,
        test_answer_sources,
        test_workspace_api_contract,
    ]
    passed = 0
    logger.info("=== 开始执行工作台功能回归测试 ===")
    for test_function in tests:
        try:
            test_function()
            logger.success(f"[PASS] {test_function.__name__}")
            passed += 1
        except Exception as exc:
            logger.error(f"[FAIL] {test_function.__name__}: {exc}", exc_info=True)
    logger.info(f"=== 测试完成：通过 {passed}/{len(tests)} ===")
    if passed != len(tests):
        sys.exit(1)
