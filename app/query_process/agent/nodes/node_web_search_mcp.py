import asyncio
import json
import sys
from concurrent.futures import ThreadPoolExecutor

from agents.mcp import MCPServerStreamableHttp

from app.conf.bailian_mcp_config import mcp_config
from app.core.logger import logger, node_log, step_log
from app.utils.task_utils import add_done_task, add_running_task


DASHSCOPE_BASE_URL_STREAM_ABLE_HTTP = mcp_config.mcp_base_url
DASHSCOPE_API_KEY = mcp_config.api_key


@step_log("step_1_data_validate")
def step_1_data_validate(state):
    rewritten_query = state["rewritten_query"]
    if not rewritten_query:
        logger.error("rewritten_query不能为空!")
        raise ValueError("rewritten_query不能为空!")
    return rewritten_query


@step_log("node_web_search_mcp_async")
async def node_web_search_mcp_async(rewritten_query: str, count: int = 5):
    """使用 OpenAI Agents MCP 客户端调用百炼网络搜索工具。"""
    mcp_server = MCPServerStreamableHttp(
        name="search_mcp",
        params={
            "url": DASHSCOPE_BASE_URL_STREAM_ABLE_HTTP,
            "headers": {"Authorization": DASHSCOPE_API_KEY},
            "timeout": 300,
            "sse_read_timeout": 300,
        },
    )

    try:
        await mcp_server.connect()
        tool_list = await mcp_server.list_tools()
        logger.info(f"MCP工具列表：{tool_list}")
        return await mcp_server.call_tool(
            tool_name="bailian_web_search",
            arguments={"query": rewritten_query, "count": count},
        )
    finally:
        await mcp_server.cleanup()


def run_async_search(rewritten_query: str, count: int = 5):
    """兼容普通调用和已有事件循环的运行环境。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(node_web_search_mcp_async(rewritten_query, count))

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(
            asyncio.run,
            node_web_search_mcp_async(rewritten_query, count),
        ).result()


@node_log("node_web_search_mcp")
def node_web_search_mcp(state):
    """调用外部搜索引擎补充信息。"""
    session_id = state["session_id"]
    run_id = state.get("run_id") or session_id
    is_stream = state.get("is_stream", False)
    node_name = sys._getframe().f_code.co_name
    add_running_task(run_id, node_name, is_stream)

    rewritten_query = step_1_data_validate(state)
    try:
        mcp_result = run_async_search(rewritten_query, count=10)
    except Exception as exc:
        logger.warning(f"联网搜索不可用，继续执行本地检索分支：{exc}")
        add_done_task(run_id, node_name, is_stream)
        return {"web_search_docs": []}

    result_text = ""
    for content in mcp_result.content or []:
        text = getattr(content, "text", "")
        if text:
            result_text = text
            break

    try:
        result_dict = json.loads(result_text) if result_text else {}
    except json.JSONDecodeError:
        logger.warning("联网搜索返回了无法解析的内容，已忽略该分支")
        result_dict = {}
    pages = result_dict.get("pages", []) if isinstance(result_dict, dict) else []
    logger.info(f"网络搜索返回{len(pages)}条结果")

    add_done_task(run_id, node_name, is_stream)
    return {"web_search_docs": pages}
