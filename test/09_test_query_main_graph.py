import json
import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.logger import logger
from app.query_process.agent.main_graph import query_app
from app.query_process.agent.state import create_query_default_state

item_module = import_module("app.query_process.agent.nodes.node_item_name_confirm")
search_module = import_module("app.query_process.agent.nodes.node_search_embedding")
hyde_module = import_module("app.query_process.agent.nodes.node_search_embedding_hyde")
web_module = import_module("app.query_process.agent.nodes.node_web_search_mcp")
rerank_module = import_module("app.query_process.agent.nodes.node_rerank")
answer_module = import_module("app.query_process.agent.nodes.node_answer_output")


def search_hit(chunk_id: int, title: str):
    return {
        "id": chunk_id,
        "distance": 0.9,
        "entity": {
            "chunk_id": chunk_id,
            "item_name": "RS-12数字万用表",
            "content": f"{title}的知识库正文",
            "title": title,
            "parent_title": "电压测量",
            "part": 1,
            "file_title": "万用表RS-12的使用",
        },
    }


if __name__ == "__main__":
    """查询图端到端回归测试，外部模型、数据库和网络服务均使用 mock。"""
    state = create_query_default_state(
        session_id=f"test_query_graph_{uuid4().hex}",
        original_query="RS-12怎么测量交流电压？",
        is_stream=False,
    )
    fake_reranker = MagicMock()
    fake_reranker.compute_score.return_value = [0.95, 0.8, 0.7]
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = SimpleNamespace(content="请将转盘置于交流电压档位后测量。")
    web_result = SimpleNamespace(
        content=[
            SimpleNamespace(
                text=json.dumps(
                    {
                        "pages": [
                            {
                                "title": "交流电压测量补充资料",
                                "snippet": "测量时注意量程和用电安全。",
                                "url": "https://example.com/safety",
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            )
        ]
    )

    logger.info("=== 开始执行查询图端到端回归测试 ===")
    with (
        patch.object(item_module, "step_2_chat_history", return_value=[]),
        patch.object(
            item_module,
            "step_3_llm_itemnames_and_rewrite",
            return_value={
                "item_names": ["RS-12数字万用表"],
                "rewritten_query": "如何使用RS-12数字万用表测量交流电压？",
            },
        ),
        patch.object(
            item_module,
            "step_4_vector_query_item_name",
            return_value={
                "RS-12数字万用表": [
                    {"item_name": "RS-12数字万用表", "score": 0.9}
                ]
            },
        ),
        patch.object(item_module, "step_7_save_user_chat_message"),
        patch.object(search_module, "step_2_rewritten_query_embedding", return_value=([0.1], {1: 0.2})),
        patch.object(
            search_module,
            "step_3_milvus_hybrid_search",
            return_value=[search_hit(101, "交流电压测量"), search_hit(102, "安全事项")],
        ),
        patch.object(hyde_module, "step_2_call_llm", return_value="假设性参考文本"),
        patch.object(hyde_module, "step_3_rewritten_hyde_vector", return_value=([0.1], {1: 0.2})),
        patch.object(
            hyde_module,
            "step_4_mivlus_hybrid_search",
            return_value=[search_hit(101, "交流电压测量")],
        ),
        patch.object(web_module, "run_async_search", return_value=web_result),
        patch.object(rerank_module, "get_reranker_model", return_value=fake_reranker),
        patch.object(rerank_module, "compress_text", side_effect=lambda text: text),
        patch.object(answer_module, "get_llm_client", return_value=fake_llm),
        patch.object(answer_module, "save_chat_message"),
        patch.object(answer_module, "set_task_result"),
    ):
        result = query_app.invoke(state)

    assert result["embedding_chunks"], "普通向量检索结果未进入图状态"
    assert result["hyde_embedding_chunks"], "HyDE 检索结果未进入图状态"
    assert result["web_search_docs"], "网络搜索结果未进入图状态"
    assert result["rrf_chunks"], "RRF 融合结果未进入图状态"
    assert result["reranked_docs"], "重排序结果未进入图状态"
    assert result["answer"] == "请将转盘置于交流电压档位后测量。"
    assert result["reranked_docs"][0]["score"] == 0.95

    logger.info("=== 查询图端到端回归测试通过 ===")
    logger.info(f"最终答案：{result['answer']}")
    logger.info(f"重排序文档数：{len(result['reranked_docs'])}")
