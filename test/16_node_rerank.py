import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.logger import logger
from app.query_process.agent.nodes import node_rerank as node_rerank_module
from app.query_process.agent.state import create_query_default_state


def create_rrf_chunk(chunk_id: int, title: str):
    """模拟 node_rrf 输出的切片（entity 结构）。"""
    return {
        "chunk_id": chunk_id,
        "item_name": "RS-12数字万用表",
        "content": f"{title}的测试正文内容，用于rerank排序验证",
        "title": title,
        "parent_title": "RS-12数字万用表使用说明",
        "part": 1,
        "file_title": "万用表RS-12的使用",
    }


def create_web_doc(title: str, url: str):
    """模拟 node_web_search_mcp 输出的网页文档。"""
    return {
        "title": title,
        "content": f"来自网络的{title}补充说明内容",
        "url": url,
    }


def build_state():
    return create_query_default_state(
        session_id=f"test_rerank_{uuid4().hex}",
        original_query="怎么测量交流电压",
        rewritten_query="RS-12数字万用表怎么测量交流电压",
        rrf_chunks=[
            create_rrf_chunk(101, "交流电压测量"),
            create_rrf_chunk(102, "直流电压测量"),
            create_rrf_chunk(103, "安全注意事项"),
        ],
        web_search_docs=[
            create_web_doc("万用表测量技巧", "https://example.com/1"),
            create_web_doc("万用表选购指南", "https://example.com/2"),
        ],
        is_stream=False,
    )


def test_step_1_data_validates():
    state = build_state()
    rrf_chunks, web_search_docs = node_rerank_module.step_1_data_validates(state)
    assert len(rrf_chunks) == 3, "应读取到 3 个 rrf 切片"
    assert len(web_search_docs) == 2, "应读取到 2 个 web 文档"


def test_step_2_merged_rrf_and_mcp():
    rrf_chunks, web_search_docs = node_rerank_module.step_1_data_validates(build_state())
    final_list = node_rerank_module.step_2_merged_rrf_and_mcp(rrf_chunks, web_search_docs)

    assert len(final_list) == 5, "两路合并后应有 5 个候选"
    milvus_items = [c for c in final_list if c["type"] == "milvus"]
    web_items = [c for c in final_list if c["type"] == "web"]
    assert len(milvus_items) == 3, "milvus 候选数应为 3"
    assert len(web_items) == 2, "web 候选数应为 2"
    # milvus 来源的 url 应为 None，web 来源的 url 应有值
    assert all(c["url"] is None for c in milvus_items)
    assert all(c["url"] for c in web_items)
    # 文本取自 content，初始分数为 0.0
    assert milvus_items[0]["text"] == "交流电压测量的测试正文内容，用于rerank排序验证"
    assert all(c["score"] == 0.0 for c in final_list)


def test_step_3_rerank_score_and_sort():
    state = build_state()
    state["scope_document_names"] = ["不应加入重排问题.pdf"]
    rrf_chunks, web_search_docs = node_rerank_module.step_1_data_validates(state)
    final_list = node_rerank_module.step_2_merged_rrf_and_mcp(rrf_chunks, web_search_docs)

    # 5 个候选，预设 5 个分数（打乱顺序验证排序）
    fake_reranker = MagicMock()
    fake_reranker.compute_score.return_value = [0.5, 0.8, 0.7, 0.9, 0.6]

    with patch.object(node_rerank_module, "get_reranker_model", return_value=fake_reranker):
        result = node_rerank_module.step_3_rerank_score_and_sort(state, final_list)

    scores = [c["score"] for c in result]
    assert scores == sorted(scores, reverse=True), "打分后应按分数降序排列"
    assert scores[0] == 0.9, "最高分 0.9 的候选应排第一"
    assert scores[-1] == 0.5, "最低分 0.5 的候选应排最后"
    # 应调用一次 compute_score，且 normalize=True
    fake_reranker.compute_score.assert_called_once()
    _, kwargs = fake_reranker.compute_score.call_args
    assert kwargs.get("normalize") is True
    assert all(pair[0] == state["original_query"] for pair in fake_reranker.compute_score.call_args.args[0])
    assert all("文档：万用表RS-12的使用" in pair[1] for pair in fake_reranker.compute_score.call_args.args[0][:3])
    assert "章节：RS-12数字万用表使用说明" in fake_reranker.compute_score.call_args.args[0][0][1]


def test_step_2_limits_local_rerank_input_but_keeps_web_results():
    local_docs = [{"type": "milvus", "text": str(index)} for index in range(30)]
    web_docs = [{"type": "web", "text": "web-1"}, {"type": "web", "text": "web-2"}]
    result = node_rerank_module.step_2_limit_rerank_candidates(
        local_docs + web_docs,
    )

    assert len([item for item in result if item["type"] == "milvus"]) == node_rerank_module.RERANK_INPUT_TOPK
    assert result[-2:] == web_docs


def test_step_4_chunk_topk_no_gap():
    """无断崖时：保留全部候选（上限为 max_topk）。"""
    chunks = [{"score": s} for s in [0.95, 0.9, 0.85, 0.8, 0.75]]
    result = node_rerank_module.step_4_chunk_topk(chunks)
    assert len(result) == 5, "无断崖时应保留 5 个候选"


def test_step_4_chunk_topk_with_gap():
    """存在断崖时：在断崖处截断，且不少于 min_topk。"""
    # 调低绝对断崖阈值，让 0.95 -> 0.6 触发断崖
    with patch.object(node_rerank_module, "RERANK_GAP_ABS", 0.1):
        chunks = [{"score": s} for s in [0.95, 0.6, 0.59, 0.58, 0.57]]
        result = node_rerank_module.step_4_chunk_topk(chunks)
        assert len(result) == 2, "断崖出现在第1名后，仍应保留至少 2 个证据"


def test_step_4_keeps_secondary_evidence_above_min_score():
    chunks = [{"score": s} for s in [0.722, 0.096, 0.0456]]
    result = node_rerank_module.step_4_chunk_topk(chunks)
    assert [chunk["score"] for chunk in result] == [0.722, 0.096]


def test_low_score_candidates_reach_answer_model_as_fallback():
    """低于阈值不等于没有资料，回答模型仍应获得少量候选用于核验。"""
    state = build_state()
    fake_reranker = MagicMock()
    fake_reranker.compute_score.return_value = [0.04, 0.03, 0.02, 0.01, 0.0]

    with (
        patch.object(node_rerank_module, "get_reranker_model", return_value=fake_reranker),
        patch.object(node_rerank_module, "add_running_task"),
        patch.object(node_rerank_module, "add_done_task"),
    ):
        result_state = node_rerank_module.node_rerank(state)

    assert result_state["answer"] == ""
    assert result_state["evidence_quality"] == "low"
    assert len(result_state["reranked_docs"]) == 4
    assert all(chunk.get("low_confidence") is True for chunk in result_state["reranked_docs"])
    assert [chunk["score"] for chunk in result_state["reranked_docs"]] == [0.04, 0.03, 0.02, 0.01]


def test_node_rerank_full():
    """主函数全流程：mock reranker 后应能完整跑通并返回 state。"""
    fake_reranker = MagicMock()
    fake_reranker.compute_score.return_value = [0.5, 0.8, 0.7, 0.9, 0.6]

    with patch.object(node_rerank_module, "get_reranker_model", return_value=fake_reranker):
        result_state = node_rerank_module.node_rerank(build_state())

    assert isinstance(result_state, dict), "主函数应返回 state 字典"
    assert result_state["session_id"], "state 中应保留 session_id"
    assert result_state["reranked_docs"], "重排序结果应写回 reranked_docs"
    assert result_state["reranked_docs"][0]["score"] == 0.9
    logger.info(f"rerank 全流程跑通，候选输入 5 条，返回 state 正常")


if __name__ == "__main__":
    """node_rerank 节点本地单元测试（mock reranker，无需加载本地模型）。"""
    tests = [
        test_step_1_data_validates,
        test_step_2_merged_rrf_and_mcp,
        test_step_3_rerank_score_and_sort,
        test_step_2_limits_local_rerank_input_but_keeps_web_results,
        test_step_4_chunk_topk_no_gap,
        test_step_4_chunk_topk_with_gap,
        test_step_4_keeps_secondary_evidence_above_min_score,
        test_low_score_candidates_reach_answer_model_as_fallback,
        test_node_rerank_full,
    ]
    passed = 0
    logger.info("=== 开始执行 node_rerank 节点单元测试 ===")
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
