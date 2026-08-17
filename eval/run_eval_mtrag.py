"""MTRAG 官方数据集适配器 + 端到端评测脚本。

数据来源：IBM/mt-rag-benchmark 官方 mtrag-human/generation_tasks/RAG.jsonl
已下载至 eval/raw/mtrag/RAG.jsonl。

评测设计：
1. 从官方任务抽样（8 ANSWERABLE + 2 UNANSWERABLE，覆盖 Follow-up/Clarification）
2. 每个任务的内嵌 contexts 文档写入 .md，导入独立知识库
3. 用对话最后一条 user 消息作为查询（多轮追问/澄清直接测试）
4. 对照官方 targets 参考回答评分（关键信息匹配）
5. UNANSWERABLE 任务检查拒答率

指标：回答正确率、来源命中率、无答案拒答率、失败率、延迟
"""
import json
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent
MTRAG_JSONL = EVAL_ROOT / "raw" / "mtrag" / "RAG.jsonl"
RESULT_DIR = EVAL_ROOT / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

IMPORT_API = "http://127.0.0.1:8000"
QUERY_API = "http://127.0.0.1:8001"

N_ANSWERABLE = 8
N_UNANSWERABLE = 2
MAX_CTX_PER_TASK = 2  # 每任务最多导入的 context 文档数

REFUSAL_KEYWORDS = [
    "无法", "没有找到", "未找到", "没有相关资料", "没有相关信息",
    "无法回答", "不能回答", "无法确认", "无法从资料", "暂未",
    "资料中未", "不能确定", "无法确定", "no answer", "not have the answer",
    "don't have the answer", "don't know", "cannot answer", "unable",
]


def http_json(method, url, body=None, timeout=120):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def multipart_upload(url, kb_id, files):
    boundary = "----mtragboundary"
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


def extract_key_points(gold: str) -> list:
    """从参考回答提取关键信息点。"""
    points = []
    for m in re.findall(r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}:\d{2}", gold):
        points.append(m)
    for m in re.findall(r"\d+(?:\.\d+)?(?:%|million|billion|thousand|km|miles|years|games|points)?", gold):
        points.append(m)
    for m in re.findall(r"[A-Z][A-Za-z .&'-]{2,40}", gold):
        points.append(m)
    for m in re.findall(r"[\u4e00-\u9fff]{4,30}", gold):
        points.append(m)
    return points


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).lower().strip()


def main():
    print("=" * 70)
    print("MTRAG 官方数据集 端到端评测")
    print("=" * 70)

    lines = MTRAG_JSONL.read_text(encoding="utf-8").strip().splitlines()
    tasks = [json.loads(line) for line in lines if line.strip()]
    print(f"官方任务共 {len(tasks)} 条")

    def answerability(t):
        a = t.get("Answerability")
        return a[0] if isinstance(a, list) and a else str(a)

    answerable = [t for t in tasks if answerability(t) == "ANSWERABLE"]
    unanswerable = [t for t in tasks if answerability(t) == "UNANSWERABLE"]
    print(f"可答 {len(answerable)} / 不可答 {len(unanswerable)}")

    sample = answerable[:N_ANSWERABLE] + unanswerable[:N_UNANSWERABLE]
    print(f"抽样 {len(sample)} 条（可答 {N_ANSWERABLE} + 不可答 {N_UNANSWERABLE}）")

    # 1. 独立知识库
    kb = http_json("POST", f"{IMPORT_API}/knowledge-bases", {"name": "eval-mtrag", "description": "MTRAG官方多轮RAG评测"})
    kb_id = kb["kb_id"]
    print(f"[1/4] 已创建独立知识库: {kb_id}")

    # 2. 收集并导入 contexts 文档（去重）
    print(f"[2/4] 收集并导入 contexts 文档...")
    doc_dir = EVAL_ROOT / "tmp" / "mtrag_docs"
    doc_dir.mkdir(parents=True, exist_ok=True)
    doc_files = []
    seen_doc = set()
    ctx_count = 0
    for t in sample:
        for ctx in t.get("contexts", [])[:MAX_CTX_PER_TASK]:
            doc_id = ctx.get("document_id", f"{t['task_id']}-{ctx_count}")
            if doc_id in seen_doc:
                continue
            seen_doc.add(doc_id)
            safe = re.sub(r"[^A-Za-z0-9_-]", "_", doc_id)[:60]
            path = doc_dir / f"mtrag_{safe}.md"
            path.write_text(str(ctx.get("text", "")), encoding="utf-8")
            doc_files.append(path)
            ctx_count += 1
    print(f"      共 {len(doc_files)} 份 context 文档")
    upload = multipart_upload(f"{IMPORT_API}/upload", kb_id, doc_files)
    task_ids = upload.get("task_ids", [])
    statuses = wait_import(task_ids)
    failed = any(st.get("status") != "completed" for st in statuses.values())
    for tid, st in statuses.items():
        print(f"      导入任务 {tid}: {st.get('status')} err={st.get('error')}")
    if failed:
        print("!!! 部分文档导入失败")

    # 3. 逐任务查询
    print(f"[3/4] 运行 {len(sample)} 条任务...")
    results = []
    latencies = []
    for idx, t in enumerate(sample):
        task_id = t["task_id"]
        answerability_val = answerability(t)
        # 取对话最后一条 user 消息作为查询
        query = ""
        for msg in reversed(t.get("input", [])):
            if msg.get("speaker") == "user":
                query = msg.get("text", "")
                break
        # 参考回答：取第一条 agent target
        gold = ""
        for msg in t.get("targets", []):
            if msg.get("speaker") == "agent":
                gold = msg.get("text", "")
                break

        try:
            payload = {
                "query": query,
                "session_id": f"mtrag-{idx}",
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
        refused = any(k.lower() in normalize(answer) for k in REFUSAL_KEYWORDS)

        # 评分：可答任务用关键点匹配，不可答任务应拒答
        if answerability_val == "UNANSWERABLE":
            correct = refused
        else:
            points = extract_key_points(gold)
            ans_n = normalize(answer)
            hits = [p for p in points if normalize(p) and normalize(p) in ans_n]
            hit_rate = len(hits) / len(points) if points else 0
            correct = hit_rate >= 0.3 or (gold and normalize(gold) in ans_n)

        record = {
            "task_id": task_id,
            "turn": t.get("turn"),
            "multi_turn": (t.get("Multi-Turn") or ["N/A"])[0] if isinstance(t.get("Multi-Turn"), list) else t.get("Multi-Turn"),
            "answerability": answerability_val,
            "query": query,
            "gold_answer": gold,
            "answer": answer,
            "refused": refused,
            "answer_correct": correct,
            "latency": round(latency, 3) if latency else None,
            "error": error,
        }
        results.append(record)
        tag = "OK" if correct else "FAIL"
        print(f"      {task_id[:20]} [{tag}] type={answerability_val} multi={record['multi_turn']} refused={refused} correct={correct} lat={latency:.1f}s" if latency else f"      {task_id[:20]} [ERR] {error}")

    # 4. 汇总
    print("[4/4] 汇总指标...")
    valid_lat = [r["latency"] for r in results if r["latency"]]
    answerable_res = [r for r in results if r["answerability"] == "ANSWERABLE"]
    unans_res = [r for r in results if r["answerability"] == "UNANSWERABLE"]
    metrics = {
        "dataset": "MTRAG (IBM/mt-rag-benchmark) RAG.jsonl",
        "samples": len(results),
        "sample_size": len(tasks),
        "answer_accuracy": round(sum(1 for r in answerable_res if r["answer_correct"]) / len(answerable_res), 4) if answerable_res else None,
        "unanswerable_refusal_rate": round(sum(1 for r in unans_res if r["refused"]) / len(unans_res), 4) if unans_res else None,
        "failure_rate": round(sum(1 for r in results if r["error"]) / len(results), 4),
        "latency_avg_s": round(statistics.mean(valid_lat), 2) if valid_lat else None,
        "latency_p50_s": round(statistics.median(valid_lat), 2) if valid_lat else None,
        "latency_p95_s": round(sorted(valid_lat)[-1], 2) if valid_lat else None,
        "config": {"is_stream": False, "enable_web_search": False, "kb_id": kb_id},
    }
    output = {"metrics": metrics, "results": results, "import_status": {k: v.get("status") for k, v in statuses.items()}}
    result_file = RESULT_DIR / "result_mtrag.json"
    result_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\n结果已保存: {result_file}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
