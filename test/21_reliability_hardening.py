import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from fastapi import HTTPException

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "eval") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from app.import_process.agent.api import server as import_server
from app.import_process.agent.nodes import node_item_name_recognition as item_recognition_module
from app.import_process.agent.nodes.node_document_split import split_dense_spec_lines
from app.core.load_prompt import load_prompt
from app.query_process.agent.nodes import node_item_name_confirm as item_name_module
from app.query_process.agent.nodes import node_document_summary as summary_module
from app.query_process.agent.retrieval_utils import search_chunks
from app.llm import embedding_utils
from eval_utils import latency_metrics, stable_case_id
from eval_workspace import EvalWorkspace


def test_stable_case_ids_prevent_prefix_collisions():
    assert stable_case_id("64fa9b2ca") != stable_case_id("64fa9b2cb")


def test_latency_p95_is_not_the_maximum_for_small_samples():
    metrics = latency_metrics([1, 2, 3, 100])
    assert metrics["latency_p95_s"] == 85.45
    assert metrics["latency_p95_s"] != 100


def test_empty_kb_scope_does_not_search_all_documents():
    with patch("app.query_process.agent.retrieval_utils.create_hybrid_search_requests") as requests:
        result = search_chunks(MagicMock(), [0.1], {1: 0.2}, [], [])
    assert result == []
    requests.assert_not_called()


def test_summary_requires_explicit_kb_scope():
    with patch.object(summary_module, "get_milvus_client") as get_client:
        assert summary_module.step_1_load_summary_chunks({"kb_ids": [], "item_names": []}) == []
    get_client.assert_not_called()


def test_upload_filename_is_normalized_and_blocks_empty_names():
    assert import_server._safe_upload_filename("../../report.md") == "report.md"
    assert import_server._safe_upload_filename("..\\report.md") == "report.md"
    try:
        import_server._safe_upload_filename("..")
    except HTTPException as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("无效文件名必须被拒绝")


def test_dense_spec_lines_are_grouped_with_heading_and_overlap():
    content = """## 技术指标说明

二极管测试 测试电流最大值1mA
短路蜂鸣测试 电阻小于30Ω
电池测试电流 9V (6mA)
输入阻抗 >1MΩ
交流电压频宽 45Hz～450Hz
DCA电压跌落测试 200mV
显示 2000位液晶显示
超量程提示 以1表示
低电池提示 显示BAT符号
电池 一粒9V (NEDA 1604)
操作环境 0°C～50°C
储存温度 -20°C～60°C"""

    chunks = split_dense_spec_lines(content)

    assert len(chunks) == 3
    assert all(chunk.startswith("## 技术指标说明") for chunk in chunks)
    assert "交流电压频宽 45Hz～450Hz" in chunks[0]
    assert "交流电压频宽 45Hz～450Hz" in chunks[1]
    assert "电池 一粒9V (NEDA 1604)" in chunks[2]


def test_embedding_calls_are_serialized_for_shared_fp16_model():
    class Sparse:
        indices = np.array([1, 2], dtype=np.int32)
        indptr = np.array([0, 1, 2], dtype=np.int32)
        data = np.array([0.5, 0.25], dtype=np.float16)

    class FakeEmbeddingModel:
        active = 0
        max_active = 0
        guard = threading.Lock()

        def encode_documents(self, texts):
            with self.guard:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.02)
            with self.guard:
                self.active -= 1
            return {
                "dense": np.array([[1.0, 2.0]], dtype=np.float16).repeat(len(texts), axis=0),
                "sparse": Sparse(),
            }

    model = FakeEmbeddingModel()
    with patch.object(embedding_utils, "get_bge_m3_ef", return_value=model):
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(embedding_utils.generate_embeddings, [["a"], ["b"]]))

    assert model.max_active == 1
    assert all(result["dense"][0] == [1.0, 2.0] for result in results)
    assert all(result["sparse"][0] == {1: 0.5} for result in results)


def test_item_name_vectors_match_float16_collection_schema():
    insert_client = MagicMock()
    insert_client.has_collection.return_value = True
    with (
        patch.object(item_recognition_module, "get_milvus_client", return_value=insert_client),
        patch.object(item_recognition_module, "ensure_document_active"),
    ):
        item_recognition_module.step_4(
            "示例主题",
            "示例文档",
            [0.1, 0.2],
            {1: 0.5},
            "kb-1",
            "doc-1",
        )

    inserted_vector = insert_client.insert.call_args.kwargs["data"][0]["dense_vector"]
    assert isinstance(inserted_vector, np.ndarray)
    assert inserted_vector.dtype == np.float16

    search_client = MagicMock()
    search_client.has_collection.return_value = True
    with (
        patch.object(item_name_module, "get_milvus_client", return_value=search_client),
        patch.object(item_name_module.milvus_config, "item_name_collection", "item_names"),
        patch.object(
            item_name_module,
            "generate_embeddings",
            return_value={"dense": [[0.1, 0.2]], "sparse": [{1: 0.5}]},
        ),
        patch.object(item_name_module, "create_hybrid_search_requests", return_value=[]) as requests,
        patch.object(item_name_module, "hybrid_search", return_value=[[]]),
    ):
        item_name_module.step_4_vector_query_item_name(["示例主题"], ["kb-1"])

    queried_vector = requests.call_args.args[0]
    assert isinstance(queried_vector, np.ndarray)
    assert queried_vector.dtype == np.float16


def test_query_resolution_prompt_preserves_task_type():
    prompt = load_prompt(
        "rewritten_query_and_itemnames",
        history_text="",
        scope_text="当前选择单篇文档",
        query="你最喜欢败犬女主里的谁",
    )

    assert "不得把人物评价识别成偏好选择" in prompt
    assert "full_document 只表示执行策略" in prompt
    assert "原问题 + 检索内容" in prompt

    state = {"original_query": "你觉得男主是什么样的人"}
    item_name_module.step_6_deal_state(
        state,
        {"confirmed_item_name_list": ["败犬女主太多了"], "options_item_name_list": []},
        {
            "rewritten_query": "主要角色有哪些；基于资料推荐其中一名并说明理由",
            "full_document": False,
        },
    )
    assert state["item_names"] == ["败犬女主太多了"]
    assert state["rewritten_query"] == "你觉得男主是什么样的人"
    assert state["full_document"] is False


def test_all_prompt_contracts_render_and_keep_critical_rules():
    rendered = {
        "answer_out": load_prompt(
            "answer_out",
            evidence_notice="低置信候选",
            image_urls="无可用图片",
            context='<source id="1">正文</source>',
            history="没有历史聊天记录",
            item_names="主题A",
            question="问题A",
        ),
        "compress": load_prompt("compress", max_chars=200, text="待压缩正文"),
        "document_synthesis": load_prompt(
            "document_synthesis",
            question="详细讲解资料",
            context='<source id="1">正文</source>',
        ),
        "hyde_prompt": load_prompt("hyde_prompt", rewritten_query="检索问题"),
        "image_summary": load_prompt(
            "image_summary",
            root_folder="文档目录",
            image_content=("图片上文", "图片下文"),
        ),
        "item_name_recognition": load_prompt(
            "item_name_recognition",
            file_title="示例文档.md",
            context="示例正文",
        ),
        "product_recognition_system": load_prompt("product_recognition_system"),
        "rewritten_query_and_itemnames": load_prompt(
            "rewritten_query_and_itemnames",
            history_text="历史",
            scope_text="当前选择知识库",
            query="当前问题",
        ),
        "summary_map": load_prompt("summary_map", question="总结问题", context="片段"),
        "summary_reduce": load_prompt(
            "summary_reduce",
            question="总结问题",
            summaries="分段摘要",
        ),
    }

    assert len(rendered) == 10
    assert all(isinstance(prompt, str) and prompt.strip() for prompt in rendered.values())
    assert "候选资料正文" in rendered["answer_out"]
    assert "AI 通识回答" in rendered["answer_out"]
    assert "AI 通识补充" in rendered["answer_out"]
    assert "每个来自资料的事实性结论" in rendered["answer_out"]
    assert "不可信数据" in rendered["answer_out"]
    assert "当前显式资料范围" in rendered["rewritten_query_and_itemnames"]
    assert "字段与取值及单位的对应关系" in rendered["compress"]
    assert "不得把“分配到不同组”扩写成“随机分组”" in rendered["document_synthesis"]
    assert "不是回答问题" in rendered["hyde_prompt"]
    assert "图片内容无法清晰识别" in rendered["image_summary"]
    assert "只输出一行名称" in rendered["item_name_recognition"]
    assert '"item_names"' in rendered["rewritten_query_and_itemnames"]
    assert "本批次无相关信息" in rendered["summary_map"]
    assert "不要提及分批、Map、Reduce" in rendered["summary_reduce"]


def test_eval_workspace_cleanup_preserves_reused_knowledge_base():
    http_json = MagicMock()
    workspace = EvalWorkspace(http_json, "http://import", "http://query", "run-1")
    workspace.use_knowledge_base("shared-kb", owns_kb=False)
    session_id = workspace.session_id("rgb", "q1")
    workspace.cleanup()

    http_json.assert_called_once_with(
        "DELETE",
        f"http://query/sessions/{session_id}",
        timeout=120,
    )


if __name__ == "__main__":
    tests = [
        test_stable_case_ids_prevent_prefix_collisions,
        test_latency_p95_is_not_the_maximum_for_small_samples,
        test_empty_kb_scope_does_not_search_all_documents,
        test_summary_requires_explicit_kb_scope,
        test_upload_filename_is_normalized_and_blocks_empty_names,
        test_dense_spec_lines_are_grouped_with_heading_and_overlap,
        test_embedding_calls_are_serialized_for_shared_fp16_model,
        test_item_name_vectors_match_float16_collection_schema,
        test_query_resolution_prompt_preserves_task_type,
        test_all_prompt_contracts_render_and_keep_critical_rules,
        test_eval_workspace_cleanup_preserves_reused_knowledge_base,
    ]
    failures = []
    for test in tests:
        try:
            test()
            print(f"[PASS] {test.__name__}")
        except Exception as exc:
            failures.append((test.__name__, exc))
            print(f"[FAIL] {test.__name__}: {exc}")
    if failures:
        sys.exit(1)
