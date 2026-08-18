"""Reproducible end-to-end evaluation for the local Chinese QA set."""

import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from eval_workspace import EvalWorkspace, new_eval_run_id
from eval_utils import latency_metrics, rate


EVAL_ROOT = Path(__file__).resolve().parent
DATASET_PATH = EVAL_ROOT / "datasets" / "selfbuilt_qa.json"
RESULT_PATH = EVAL_ROOT / "results" / "result_selfbuilt_qa.json"
IMPORT_API = "http://127.0.0.1:8000"
QUERY_API = "http://127.0.0.1:8001"

REFUSAL_KEYWORDS = (
    "无法", "没有找到", "未找到", "没有相关资料", "没有相关信息",
    "没有足够信息", "无法回答", "不能回答", "无法确认", "暂未", "不能确定", "无法确定",
)
GENERAL_KNOWLEDGE_NOTICE = "> **AI 通识回答**：当前知识库未提供直接依据。"


def http_json(method, url, body=None, timeout=120, retries=0):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 502, 503, 504} or attempt >= retries:
                raise
            time.sleep(2 ** attempt)


def multipart_upload(url, kb_id, files):
    boundary = "----selfbuiltboundary"
    body = f"--{boundary}\r\nContent-Disposition: form-data; name=\"kb_id\"\r\n\r\n{kb_id}\r\n".encode()
    for file_path in files:
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="files"; filename="{file_path.name}"\r\n'.encode("utf-8")
        body += b"Content-Type: application/octet-stream\r\n\r\n"
        body += file_path.read_bytes() + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_import(task_ids, timeout=1800):
    deadline = time.time() + timeout
    statuses = {}
    while time.time() < deadline:
        all_done = True
        for task_id in task_ids:
            try:
                status = http_json("GET", f"{IMPORT_API}/status/{task_id}", timeout=120)
            except Exception as exc:
                print(f"导入状态暂时不可用，稍后重试: task={task_id} error={exc}")
                all_done = False
                continue
            statuses[task_id] = status
            if status.get("status") not in {"completed", "failed"}:
                all_done = False
        if all_done:
            return statuses
        time.sleep(10)
    return statuses


def wait_documents(kb_id, timeout=1800):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            payload = http_json("GET", f"{IMPORT_API}/knowledge-bases/{kb_id}/documents", timeout=120)
        except Exception as exc:
            print(f"知识库文档状态暂时不可用，稍后重试: {exc}")
            time.sleep(10)
            continue
        documents = payload.get("items", [])
        if documents and all(document.get("status") in {"completed", "failed"} for document in documents):
            return {
                document.get("task_id", document.get("document_id", "")): {
                    "status": document.get("status"),
                    "error": document.get("error", ""),
                }
                for document in documents
            }
        time.sleep(10)
    return {}


def normalize(value):
    return re.sub(r"\s+", "", str(value or "")).lower()


def main():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    cases = dataset.get("cases") or []
    document_paths = [(DATASET_PATH.parent / path).resolve() for path in dataset.get("documents", [])]
    missing = [str(path) for path in document_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"评测文档不存在: {missing}")
    if not cases:
        raise ValueError("自建评测集没有 cases，无法运行")

    eval_run_id = new_eval_run_id()
    workspace = EvalWorkspace(http_json, IMPORT_API, QUERY_API, eval_run_id)

    import os
    kb_id = os.getenv("EVAL_KB_ID", "").strip()
    if kb_id:
        workspace.use_knowledge_base(kb_id, owns_kb=False)
        print(f"复用已有评测知识库: {kb_id}")
        statuses = wait_documents(kb_id)
    else:
        kb = http_json(
            "POST",
            f"{IMPORT_API}/knowledge-bases",
            {"name": f"eval-{dataset.get('name', 'selfbuilt')}", "description": "可复现自建问答评测"},
        )
        kb_id = workspace.use_knowledge_base(kb["kb_id"], owns_kb=True)
        upload = multipart_upload(f"{IMPORT_API}/upload", kb_id, document_paths)
        statuses = wait_import(upload.get("task_ids", []))
    import_failed = not statuses or any(
        status.get("status") != "completed" for status in statuses.values()
    )
    if import_failed:
        raise RuntimeError(f"评测文档导入未全部完成: {statuses}")

    results = []
    for case in cases:
        qid = str(case["qid"])
        start = time.monotonic()
        try:
            response = http_json(
                "POST",
                f"{QUERY_API}/query",
                {
                    "query": case["query"],
                    "session_id": workspace.session_id("selfbuilt", qid),
                    "scope_mode": "knowledge_base",
                    "kb_ids": [kb_id],
                    "document_ids": [],
                    "is_stream": False,
                    "enable_web_search": False,
                },
                timeout=300,
                retries=2,
            )
            latency = time.monotonic() - start
            error = None
        except Exception as exc:
            response = {}
            latency = None
            error = str(exc)

        answer = response.get("answer", "")
        sources = response.get("sources", [])
        answerable = bool(case.get("answerable", True))
        expected_terms = [normalize(term) for term in case.get("expected_terms", []) if normalize(term)]
        forbidden_terms = [normalize(term) for term in case.get("forbidden_terms", []) if normalize(term)]
        answer_normalized = normalize(answer)
        general_knowledge_disclosed = answer.lstrip().startswith(GENERAL_KNOWLEDGE_NOTICE)
        # “尚未发现生命证据”等正常通识结论不等于系统拒答。
        refused = not general_knowledge_disclosed and any(
            keyword in answer for keyword in REFUSAL_KEYWORDS
        )
        knowledge_gap_handled = refused or general_knowledge_disclosed
        answer_correct = (
            all(term in answer_normalized for term in expected_terms)
            and not any(term in answer_normalized for term in forbidden_terms)
            if answerable
            else refused
        )
        expected_doc = normalize(case.get("expected_doc"))
        source_hit = (
            any(expected_doc in normalize(source.get("file_title")) for source in sources)
            if answerable and expected_doc
            else None
        )
        results.append(
            {
                "qid": qid,
                "query": case["query"],
                "answerable": answerable,
                "expected_terms": case.get("expected_terms", []),
                "forbidden_terms": case.get("forbidden_terms", []),
                "expected_doc": case.get("expected_doc", ""),
                "answer": answer,
                "sources": sources,
                "source_hit": source_hit,
                "answer_correct": answer_correct,
                "refused": refused,
                "general_knowledge_disclosed": general_knowledge_disclosed,
                "knowledge_gap_handled": knowledge_gap_handled,
                "latency": round(latency, 3) if latency is not None else None,
                "error": error,
            }
        )

    answerable_results = [result for result in results if result["answerable"]]
    unanswerable_results = [result for result in results if not result["answerable"]]
    metrics = {
        "dataset": dataset.get("name", "selfbuilt_zh_qa"),
        "samples": len(results),
        "source_hit_rate": rate(result["source_hit"] for result in answerable_results),
        "answer_accuracy": rate(result["answer_correct"] for result in answerable_results),
        "unanswerable_refusal_rate": rate(result["refused"] for result in unanswerable_results),
        "unanswerable_disclosure_rate": rate(
            result["general_knowledge_disclosed"] for result in unanswerable_results
        ),
        "unanswerable_handled_rate": rate(
            result["knowledge_gap_handled"] for result in unanswerable_results
        ),
        "failure_rate": rate(bool(result["error"]) for result in results),
        **latency_metrics(result["latency"] for result in results),
        "config": {
            "is_stream": False,
            "enable_web_search": False,
            "kb_id": kb_id,
            "eval_run_id": eval_run_id,
            "dataset_file": str(DATASET_PATH),
        },
    }
    output = {
        "metrics": metrics,
        "results": results,
        "import_status": {task_id: status.get("status") for task_id, status in statuses.items()},
    }
    RESULT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"结果已保存: {RESULT_PATH}")
    workspace.cleanup()
    return 1 if import_failed else 0


if __name__ == "__main__":
    sys.exit(main())
