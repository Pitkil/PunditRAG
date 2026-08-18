import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.clients.mongo_history_utils import clear_history, get_recent_messages
from app.core.logger import logger
from app.query_process.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.query_process.agent.state import create_query_default_state


if __name__ == "__main__":
    """主题识别与资料范围确认节点本地集成测试。"""
    session_id = f"test_item_name_confirm_{uuid4().hex}"
    test_state = create_query_default_state(
        session_id=session_id,
        original_query="请根据学习资料解释牛顿第二定律，并给出公式。",
        is_stream=False,
    )

    logger.info("=== 开始执行主题识别与资料范围确认节点测试 ===")
    try:
        result_state = node_item_name_confirm(test_state)
        history_messages = get_recent_messages(session_id)

        assert result_state["rewritten_query"] == test_state["original_query"], "兼容字段必须保留用户原问题"
        assert isinstance(result_state["item_names"], list), "item_names 必须是列表"
        assert len(history_messages) == 1, "本次用户问题应保存一条 MongoDB 记录"
        assert history_messages[0]["text"] == test_state["original_query"]
        assert history_messages[0]["rewritten_query"] == result_state["rewritten_query"]
        assert history_messages[0]["item_names"] == result_state["item_names"]

        logger.info("=== 主题识别与资料范围确认节点测试通过 ===")
        logger.info(f"原始问题：{result_state['original_query']}")
        logger.info(f"原始问题：{result_state['rewritten_query']}")
        logger.info(f"匹配主题或实体：{result_state['item_names']}")
        logger.info(f"节点回复：{result_state['answer'] or '无，继续执行后续检索'}")
    finally:
        deleted_count = clear_history(session_id)
        logger.info(f"测试数据清理完成，共删除{deleted_count}条消息")
