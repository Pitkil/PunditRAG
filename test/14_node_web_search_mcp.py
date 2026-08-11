import asyncio
import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.logger import logger
from app.query_process.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.query_process.agent.state import create_query_default_state


async def run_node_inside_event_loop(test_state):
    """模拟 FastAPI 异步接口中已经存在事件循环的运行环境。"""
    return node_web_search_mcp(test_state)


if __name__ == "__main__":
    """百炼 MCP 网络搜索节点本地集成测试。"""
    test_state = create_query_default_state(
        session_id=f"test_web_search_mcp_{uuid4().hex}",
        original_query="新西兰人工智能教育有什么最新进展？",
        rewritten_query="新西兰人工智能教育最新进展",
        is_stream=False,
    )

    logger.info("=== 开始执行百炼 MCP 网络搜索节点测试 ===")
    result_state = asyncio.run(run_node_inside_event_loop(test_state))
    documents = result_state.get("web_search_docs", [])

    assert documents, "百炼 MCP 未返回任何网络搜索结果"
    assert all(isinstance(document, dict) for document in documents)
    assert all(document.get("title") for document in documents), "搜索结果缺少标题"
    assert all(document.get("url") for document in documents), "搜索结果缺少 URL"
    assert all(document.get("snippet") for document in documents), "搜索结果缺少摘要"

    logger.info("=== 百炼 MCP 网络搜索节点测试通过 ===")
    logger.info(f"改写问题：{test_state['rewritten_query']}")
    logger.info(f"返回网页数量：{len(documents)}")
    logger.info(f"首条标题：{documents[0]['title']}")
    logger.info(f"首条地址：{documents[0]['url']}")
