import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.logger import logger
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.state import create_query_default_state


def create_search_chunk(chunk_id: int, title: str):
    return {
        "id": chunk_id,
        "distance": 0.9,
        "entity": {
            "chunk_id": chunk_id,
            "item_name": "RS-12数字万用表",
            "content": f"{title}的测试内容",
            "title": title,
            "parent_title": "RS-12数字万用表使用说明",
            "part": 1,
            "file_title": "万用表RS-12的使用",
        },
    }


if __name__ == "__main__":
    """RRF 多路召回融合排序节点本地单元测试。"""
    shared_chunk = create_search_chunk(103, "交流电压测量")
    embedding_chunks = [
        create_search_chunk(101, "直流电压测量"),
        create_search_chunk(102, "电阻测量"),
        shared_chunk,
    ]
    hyde_embedding_chunks = [
        create_search_chunk(104, "交流电流测量"),
        create_search_chunk(105, "安全注意事项"),
        shared_chunk,
    ]
    test_state = create_query_default_state(
        session_id=f"test_rrf_{uuid4().hex}",
        original_query="RS-12数字万用表怎么测量交流电压？",
        embedding_chunks=embedding_chunks,
        hyde_embedding_chunks=hyde_embedding_chunks,
        is_stream=False,
    )

    logger.info("=== 开始执行 RRF 多路召回融合排序节点测试 ===")
    result_state = node_rrf(test_state)
    chunks = result_state.get("rrf_chunks", [])
    chunk_ids = [chunk.get("chunk_id") for chunk in chunks]

    assert len(chunks) == 5, "RRF 融合结果应包含两路检索去重后的 5 个切片"
    assert set(chunk_ids) == {101, 102, 103, 104, 105}, "RRF 融合结果缺少某一路检索切片"
    assert chunk_ids[0] == 103, "两路检索共同召回的切片应在 RRF 融合后排在首位"

    logger.info("=== RRF 多路召回融合排序节点测试通过 ===")
    logger.info(f"普通向量检索切片数：{len(embedding_chunks)}")
    logger.info(f"HyDE 检索切片数：{len(hyde_embedding_chunks)}")
    logger.info(f"RRF 融合结果切片ID：{chunk_ids}")
    logger.info(f"首条融合结果：{chunks[0]}")
