import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.logger import logger
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.state import create_query_default_state


if __name__ == "__main__":
    """普通向量混合检索节点本地集成测试。"""
    test_state = create_query_default_state(
        session_id=f"test_search_embedding_{uuid4().hex}",
        original_query="RS-12数字万用表怎么测量电压？",
        rewritten_query="如何使用RS-12数字万用表测量电压？",
        item_names=["RS-12数字万用表"],
        is_stream=False,
    )

    logger.info("=== 开始执行普通向量混合检索节点测试 ===")
    result_state = node_search_embedding(test_state)
    chunks = result_state.get("embedding_chunks", [])

    assert chunks, "Milvus 未返回任何相关切片"
    assert all(
        chunk.get("entity", {}).get("item_name") == "RS-12数字万用表"
        for chunk in chunks
    ), "检索结果包含过滤范围之外的资料"

    logger.info("=== 普通向量混合检索节点测试通过 ===")
    logger.info(f"改写问题：{test_state['rewritten_query']}")
    logger.info(f"资料范围：{test_state['item_names']}")
    logger.info(f"返回切片数量：{len(chunks)}")
    logger.info(f"首条切片：{chunks[0].get('entity', {})}")
