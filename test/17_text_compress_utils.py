import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.logger import logger
from app.llm import text_compress_utils as tcu


def _mock_llm_returning(value: str):
    """构造一个 `llm | StrOutputParser()` 后 `.invoke()` 返回指定值的假 LLM。"""
    llm = MagicMock()
    chain = MagicMock()
    chain.invoke.return_value = value
    llm.__or__.return_value = chain
    return llm


def test_compress_by_truncate_short():
    short = "短文本"
    assert tcu.compress_by_truncate(short, max_chars=10) == short, "短文本不应被截断"


def test_compress_by_truncate_long():
    long_text = "长" * 100
    out = tcu.compress_by_truncate(long_text, max_chars=50)
    assert len(out) == 50, "超长文本应被截断到 max_chars"
    assert out == long_text[:50], "截断应保留开头"


def test_compress_by_llm_short_no_llm_call():
    """短文本不应调用 LLM，直接返回原文。"""
    short = "短" * 10
    with patch("app.llm.text_compress_utils.get_llm_client") as mock_get:
        result = tcu.compress_by_llm(short, max_chars=400)
    assert result == short
    mock_get.assert_not_called(), "短文本不应触发 LLM 调用"


def test_compress_by_llm_long():
    """超长文本应调用 LLM 并返回压缩结果。"""
    long_text = "长" * 500
    with patch(
        "app.llm.text_compress_utils.get_llm_client",
        return_value=_mock_llm_returning("压缩后的摘要内容"),
    ):
        result = tcu.compress_by_llm(long_text, max_chars=400)
    assert result == "压缩后的摘要内容", "应返回 LLM 压缩结果"


def test_compress_by_llm_long_result_truncated():
    """LLM 返回超长结果时，应二次截断到 max_chars。"""
    long_text = "长" * 500
    with patch(
        "app.llm.text_compress_utils.get_llm_client",
        return_value=_mock_llm_returning("压" * 1000),
    ):
        result = tcu.compress_by_llm(long_text, max_chars=400)
    assert len(result) == 400, "压缩结果超长应被截断"


def test_compress_by_llm_empty_fallback():
    """LLM 返回空串时，应回退为截断原文。"""
    long_text = "长" * 500
    with patch(
        "app.llm.text_compress_utils.get_llm_client",
        return_value=_mock_llm_returning(""),
    ):
        result = tcu.compress_by_llm(long_text, max_chars=400)
    assert result == long_text[:400], "空结果应回退为截断原文"


def test_compress_text_exception_fallback():
    """LLM 抛异常时，compress_text 应回退截断，不崩溃。"""
    long_text = "长" * 500
    with patch(
        "app.llm.text_compress_utils.get_llm_client",
        side_effect=RuntimeError("LLM 挂了"),
    ):
        result = tcu.compress_text(long_text, max_chars=400)
    assert result == long_text[:400], "异常时应回退为截断原文"


def test_compress_text_short_no_call():
    """compress_text 对短文本直接返回，不触发任何 LLM 调用。"""
    short = "短" * 10
    with patch("app.llm.text_compress_utils.get_llm_client") as mock_get:
        result = tcu.compress_text(short, max_chars=400)
    assert result == short
    mock_get.assert_not_called()


if __name__ == "__main__":
    """text_compress_utils 正文压缩工具本地单元测试（mock LLM，无需真实接口）。"""
    tests = [
        test_compress_by_truncate_short,
        test_compress_by_truncate_long,
        test_compress_by_llm_short_no_llm_call,
        test_compress_by_llm_long,
        test_compress_by_llm_long_result_truncated,
        test_compress_by_llm_empty_fallback,
        test_compress_text_exception_fallback,
        test_compress_text_short_no_call,
    ]
    passed = 0
    logger.info("=== 开始执行 text_compress_utils 压缩工具单元测试 ===")
    for test_func in tests:
        try:
            test_func()
            logger.success(f"[PASS] {test_func.__name__}")
            passed += 1
        except Exception as e:
            logger.error(f"[FAIL] {test_func.__name__}: {e}", exc_info=True)
    logger.info(f"=== 测试完成：通过 {passed}/{len(tests)} ===")
    if passed != len(tests):
        sys.exit(1)
