"""下载 MTRAG 任务数据并查看结构。"""
import json
import requests
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent
RAW_DIR = EVAL_ROOT / "raw" / "mtrag"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0"}

FILES = {
    "conversations.json": "https://raw.githubusercontent.com/IBM/mt-rag-benchmark/main/mtrag-human/conversations/conversations.json",
    "RAG.jsonl": "https://raw.githubusercontent.com/IBM/mt-rag-benchmark/main/mtrag-human/generation_tasks/RAG.jsonl",
}

for name, url in FILES.items():
    target = RAW_DIR / name
    if target.exists() and target.stat().st_size > 0:
        print(f"[skip] {name} 已存在 {target.stat().st_size}B")
        continue
    print(f"[下载] {name} <- {url}")
    r = requests.get(url, headers=HEADERS, timeout=300)
    r.raise_for_status()
    target.write_bytes(r.content)
    print(f"[完成] {name} {len(r.content)}B")

# 查看结构
print("\n=== conversations.json ===")
data = json.loads((RAW_DIR / "conversations.json").read_text(encoding="utf-8"))
print(f"类型: {type(data)}, 数量: {len(data) if hasattr(data,'__len__') else '?'}")
if isinstance(data, dict):
    print(f"顶层键: {list(data.keys())[:10]}")
    for k in list(data.keys())[:1]:
        v = data[k]
        print(f"  {k}: {type(v)}")
        if isinstance(v, list) and v:
            print(f"    元素字段: {list(v[0].keys()) if isinstance(v[0],dict) else v[0]}")
elif isinstance(data, list):
    print(f"元素字段: {list(data[0].keys()) if data and isinstance(data[0],dict) else data[:1]}")
    if data and isinstance(data[0], dict):
        for k, val in data[0].items():
            print(f"  {k}: {str(val)[:150]}")

print("\n=== RAG.jsonl 前2条 ===")
lines = (RAW_DIR / "RAG.jsonl").read_text(encoding="utf-8").strip().splitlines()
for line in lines[:2]:
    obj = json.loads(line)
    print("字段:", list(obj.keys()))
    for k, v in obj.items():
        print(f"  {k}: {str(v)[:180]}")
    print("---")
