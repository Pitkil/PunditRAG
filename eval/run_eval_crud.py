"""CRUD-RAG 官方中文问答子集（questanswer_1doc）适配器 + 端到端评测脚本。

数据来源：CRUD-RAG (IAAR-Shanghai/CRUD_RAG) 官方 data/crud_split/split_merged.json
已下载至 eval/raw/crudrag/split_merged.json。

流程：
1. 从官方 questanswer_1doc 抽样 N 条
2. 每条 news1 原文写入独立 .md 文档
3. 上传到独立知识库
4. 逐条查询（关闭联网搜索）
5. 对照官方 answers 计算指标
"""
import json
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path

from eval_workspace import EvalWorkspace, new_eval_run_id

EVAL_ROOT = Path(__file__).resolve().parent
CRUD_JSON = EVAL_ROOT / "raw" / "crudrag" / "split_merged.json"
RESULT_DIR = EVAL_ROOT / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

IMPORT_API = "http://127.0.0.1:8000"
QUERY_API = "http://127.0.0.1:8001"

SAMPLE_N = 10  # 抽样条数


def http_json(method, url, body=None, timeout=120):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def multipart_upload(url, kb_id, files):
    boundary = "----crudboundary"
    body = b""
    body += f"--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="kb_id"\r\n\r\n'
    body += kb_id.encode() + b"\r\n"
    for file_path in files:
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="files"; filename="{file_path.name}"\r\n'.encode()
        body += b"Content-Type: text/markdown\r\n\r\n"
        body += file_path.read_bytes() + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_import(task_ids, timeout=900):
    deadline = time.time() + timeout
    statuses = {}
    while time.time() < deadline:
        all_done = True
        for tid in task_ids:
            st = http_json("GET", f"{IMPORT_API}/status/{tid}", timeout=30)
            statuses[tid] = st
            if st.get("status") != "completed":
                all_done = False
        if all_done:
            return statuses
        time.sleep(10)
    return statuses


def main():
    print("=" * 70)
    print("CRUD-RAG 官方中文问答子集(questanswer_1doc) 端到端评测")
    print("=" * 70)

    data = json.loads(CRUD_JSON.read_text(encoding="utf-8"))
    items = data.get("questanswer_1doc", [])
    print(f"官方 questanswer_1doc 共 {len(items)} 条，抽样 {SAMPLE_N} 条")

    step = max(1, len(items) // SAMPLE_N)
    sample = [items[i] for i in range(0, len(items), step)][:SAMPLE_N]

    workspace = EvalWorkspace(http_json, IMPORT_API, QUERY_API, new_eval_run_id())
    reused_kb_id = os.getenv("EVAL_KB_ID", "").strip()
    if reused_kb_id:
        kb_id = workspace.use_knowledge_base(reused_kb_id, owns_kb=False)
        print(f"[1/4] 复用评测知识库: {kb_id}")
    else:
        kb = http_json("POST", f"{IMPORT_API}/knowledge-bases", {"name": "eval-crudrag-qa1", "description": "CRUD-RAG官方中文问答子集"})
        kb_id = workspace.use_knowledge_base(kb["kb_id"], owns_kb=True)
        print(f"[1/4] 已创建独立知识库: {kb_id}")

    # 2. news1 写入 .md 并上传
    print(f"[2/4] 导入 {len(sample)} 条官方 news1 文档...")
    doc_dir = EVAL_ROOT / "tmp" / "crud_docs"
    doc_dir.mkdir(parents=True, exist_ok=True)
    doc_files = []
    for idx, item in enumerate(sample):
        fid = f"crud_qa1_{item['ID'][:8]}"
        content = str(item.get("news1", ""))
        path = doc_dir / f"{fid}.md"
        path.write_text(content, encoding="utf-8")
        doc_files.append(path)
    if reused_kb_id:
        statuses = {"reused": {"status": "completed"}}
    else:
        upload = multipart_upload(f"{IMPORT_API}/upload", kb_id, doc_files)
        task_ids = upload.get("task_ids", [])
        statuses = wait_import(task_ids)
    failed = any(st.get("status") != "completed" for st in statuses.values())
    for tid, st in statuses.items():
        print(f"      导入任务 {tid}: {st.get('status')} err={st.get('error')}")
    if failed:
        print("!!! 部分文档导入失败")

    # 3. 逐条查询
    print(f"[3/4] 运行 {len(sample)} 条官方 query...")
    results = []
    latencies = []
    for i, item in enumerate(sample):
        qid = item["ID"][:8]
        fid = f"crud_qa1_{qid}"
        query = item["questions"]
        gold = item["answers"]
        try:
            payload = {
                "query": query,
                "session_id": workspace.session_id("crud", qid),
                "kb_ids": [kb_id],
                "is_stream": False,
                "enable_web_search": False,
            }
            start = time.monotonic()
            result = http_json("POST", f"{QUERY_API}/query", payload, timeout=300)
            latency = time.monotonic() - start
            latencies.append(latency)
            error = None
        except Exception as exc:
            result = {}
            latency = None
            error = str(exc)

        answer = result.get("answer", "")
        sources = result.get("sources", [])
        source_files = {s.get("file_title", "") for s in sources}
        source_hit = any(fid in f for f in source_files)

        # 回答正确：官方答案中提取的关键词是否命中（答案与问题文档同源）
        gold_compact = "".join(str(gold).split())
        ans_compact = "".join(answer.split())
        # 用官方答案的前 20 个字符做包含判断（宽松）
        key_part = gold_compact[:20]
        answer_correct = bool(key_part) and key_part in ans_compact

        record = {
            "qid": qid,
            "query": query,
            "gold_answer": gold,
            "answer": answer,
            "source_hit": source_hit,
            "answer_correct": answer_correct,
            "latency": round(latency, 3) if latency else None,
            "error": error,
        }
        results.append(record)
        tag = "OK" if answer_correct else "FAIL"
        print(f"      {fid} [{tag}] correct={answer_correct} source_hit={source_hit} lat={latency:.1f}s" if latency else f"      {fid} [ERR] {error}")

    # 4. 汇总
    print("[4/4] 汇总指标...")
    valid_lat = [r["latency"] for r in results if r["latency"]]
    metrics = {
        "dataset": "CRUD-RAG (IAAR-Shanghai/CRUD_RAG) questanswer_1doc",
        "samples": len(results),
        "sample_size": len(items),
        "source_hit_rate": round(sum(r["source_hit"] for r in results) / len(results), 4),
        "answer_accuracy": round(sum(r["answer_correct"] for r in results) / len(results), 4),
        "failure_rate": round(sum(1 for r in results if r["error"]) / len(results), 4),
        "latency_avg_s": round(statistics.mean(valid_lat), 2) if valid_lat else None,
        "latency_p50_s": round(statistics.median(valid_lat), 2) if valid_lat else None,
        "latency_p95_s": round(sorted(valid_lat)[-1], 2) if valid_lat else None,
        "config": {"is_stream": False, "enable_web_search": False, "kb_id": kb_id},
    }
    output = {"metrics": metrics, "results": results, "import_status": {k: v.get("status") for k, v in statuses.items()}}
    result_file = RESULT_DIR / "result_crudrag_qa1.json"
    result_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\n结果已保存: {result_file}")
    workspace.cleanup()
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
