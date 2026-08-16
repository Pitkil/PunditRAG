import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.logger import logger
from app.query_process.agent.nodes import node_answer_output as answer_module
from app.query_process.agent.state import create_query_default_state


def build_state(**overrides):
    state = create_query_default_state(
        session_id=f"test_answer_output_{uuid4().hex}",
        original_query="RS-12数字万用表怎么测量交流电压？",
        rewritten_query="如何使用RS-12数字万用表测量交流电压？",
        item_names=["RS-12数字万用表"],
        history=[
            {
                "role": "user",
                "rewritten_query": "如何测量直流电压？",
                "text": "",
                "item_names": ["RS-12数字万用表"],
            }
        ],
        reranked_docs=[
            {
                "title": "交流电压测量",
                "score": 0.95,
                "type": "milvus",
                "text": "将功能转盘置于交流电压档位，再将表笔接触被测物。",
                "url": None,
            },
            {
                "title": "安全注意事项",
                "score": 0.82,
                "type": "web",
                "text": "测量时不要触碰表笔金属部分。",
                "url": "https://example.com/safety",
            },
        ],
        is_stream=False,
    )
    state.update(overrides)
    return state


def test_step_1_without_existing_answer():
    assert answer_module.step_1_data_validates(build_state(answer="")) is False


def test_step_1_with_existing_answer():
    state = build_state(answer="已有答案", is_stream=False)
    with patch.object(answer_module, "set_task_result") as mock_set_result:
        result = answer_module.step_1_data_validates(state)

    assert result is True
    mock_set_result.assert_called_once_with(state["session_id"], "answer", "已有答案")


def test_step_2_data_validates():
    state = build_state()
    history, docs, item_names, query = answer_module.step_2_data_validates(state)

    assert history == state["history"]
    assert docs == state["reranked_docs"]
    assert item_names == ["RS-12数字万用表"]
    assert query == state["rewritten_query"]


def test_step_2_rejects_empty_docs():
    try:
        answer_module.step_2_data_validates(build_state(reranked_docs=[]))
    except ValueError as exc:
        assert "reranked_docs为空" in str(exc)
    else:
        raise AssertionError("reranked_docs 为空时应抛出 ValueError")


def test_step_3_make_prompt():
    state = build_state()
    history, docs, item_names, query = answer_module.step_2_data_validates(state)

    with patch.object(answer_module, "load_prompt", return_value="组装后的提示词") as mock_load:
        prompt = answer_module.step_3_make_prompt(state, history, docs, item_names, query)

    assert prompt == "组装后的提示词"
    _, kwargs = mock_load.call_args
    assert "交流电压测量" in kwargs["context"]
    assert "知识库文档" in kwargs["context"]
    assert "网络搜索" in kwargs["context"]
    assert "RS-12数字万用表" in kwargs["item_names"]
    assert kwargs["question"] == state["rewritten_query"]


def test_step_4_generate_answer_non_stream():
    state = build_state(is_stream=False)
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = SimpleNamespace(content="非流式生成答案")

    with (
        patch.object(answer_module, "get_llm_client", return_value=fake_llm),
        patch.object(answer_module, "set_task_result") as mock_set_result,
    ):
        answer_module.step_4_generate_answer(state, "测试提示词")

    assert state["answer"] == "非流式生成答案"
    fake_llm.invoke.assert_called_once_with("测试提示词")
    mock_set_result.assert_called_once_with(state["session_id"], "answer", "非流式生成答案")


def test_step_4_generate_answer_stream():
    state = build_state(is_stream=True)
    fake_llm = MagicMock()
    fake_llm.stream.return_value = [
        SimpleNamespace(content="流式"),
        SimpleNamespace(content="答案"),
    ]

    with (
        patch.object(answer_module, "get_llm_client", return_value=fake_llm),
        patch.object(answer_module, "push_to_session") as mock_push,
        patch.object(answer_module, "set_task_result") as mock_set_result,
    ):
        answer_module.step_4_generate_answer(state, "测试提示词")

    assert state["answer"] == "流式答案"
    assert mock_push.call_count == 1
    assert [call.args[2]["delta"] for call in mock_push.call_args_list] == ["流式答案"]
    mock_set_result.assert_called_once_with(state["session_id"], "answer", "流式答案")


def test_node_answer_output_non_stream():
    state = build_state(answer="节点完整答案", is_stream=False)
    with (
        patch.object(answer_module, "add_running_task") as mock_running,
        patch.object(answer_module, "add_done_task") as mock_done,
        patch.object(answer_module, "push_to_session") as mock_push,
        patch.object(answer_module, "save_chat_message") as mock_save,
    ):
        result = answer_module.node_answer_output(state)

    assert result["answer"] == "节点完整答案"
    mock_running.assert_called_once()
    mock_done.assert_called_once()
    mock_push.assert_not_called()
    mock_save.assert_called_once()


def test_node_answer_output_stream():
    state = build_state(answer="流式", is_stream=True)
    with (
        patch.object(answer_module, "add_running_task"),
        patch.object(answer_module, "add_done_task"),
        patch.object(answer_module, "push_to_session") as mock_push,
        patch.object(answer_module, "save_chat_message") as mock_save,
        patch.object(answer_module.time, "sleep"),
    ):
        result = answer_module.node_answer_output(state)

    assert result["answer"] == "流式"
    assert [call.args[2]["delta"] for call in mock_push.call_args_list] == ["流式"]
    mock_save.assert_called_once()


if __name__ == "__main__":
    """node_answer_output 节点本地单元测试，无需真实大模型和 SSE 服务。"""
    tests = [
        test_step_1_without_existing_answer,
        test_step_1_with_existing_answer,
        test_step_2_data_validates,
        test_step_2_rejects_empty_docs,
        test_step_3_make_prompt,
        test_step_4_generate_answer_non_stream,
        test_step_4_generate_answer_stream,
        test_node_answer_output_non_stream,
        test_node_answer_output_stream,
    ]
    passed = 0
    logger.info("=== 开始执行 node_answer_output 节点单元测试 ===")
    for test_func in tests:
        try:
            test_func()
            logger.success(f"[PASS] {test_func.__name__}")
            passed += 1
        except Exception as exc:
            logger.error(f"[FAIL] {test_func.__name__}: {exc}", exc_info=True)
    logger.info(f"=== 测试完成：通过 {passed}/{len(tests)} ===")
    if passed != len(tests):
        sys.exit(1)
