import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.clients.mongo_history_utils import (
    clear_history,
    get_history_mongo_tool,
    save_chat_message,
    update_message_item_names,
)
from app.core.logger import logger


if __name__ == "__main__":
    """MongoDB 对话历史本地集成测试。"""
    session_id = f"test_textbook_history_{uuid4().hex}"

    logger.info("=== 开始执行 MongoDB 对话历史本地测试 ===")
    try:
        user_message_id = save_chat_message(
            session_id=session_id,
            role="user",
            text="请解释牛顿第二定律。",
            rewritten_query="高中物理教材中的牛顿第二定律是什么？",
            item_names=[],
        )
        assistant_message_id = save_chat_message(
            session_id=session_id,
            role="assistant",
            text="牛顿第二定律描述物体加速度与合外力、质量之间的关系。",
            item_names=["牛顿第二定律"],
            image_urls=[],
        )

        updated_count = update_message_item_names(
            [user_message_id],
            ["牛顿第二定律"],
        )

        mongo_tool = get_history_mongo_tool()
        messages = list(
            mongo_tool.chat_message.find({"session_id": session_id}).sort("ts", 1)
        )

        assert len(messages) == 2, f"期望保存2条消息，实际为{len(messages)}条"
        assert updated_count == 1, f"期望更新1条消息，实际为{updated_count}条"
        assert messages[0]["role"] == "user"
        assert messages[0]["item_names"] == ["牛顿第二定律"]
        assert messages[1]["role"] == "assistant"
        assert str(messages[1]["_id"]) == assistant_message_id

        logger.info("=== MongoDB 对话历史本地测试通过 ===")
        logger.info(f"测试会话ID：{session_id}")
        logger.info(f"保存消息数：{len(messages)} | 更新消息数：{updated_count}")
    finally:
        deleted_count = clear_history(session_id)
        logger.info(f"测试数据清理完成，共删除{deleted_count}条消息")
