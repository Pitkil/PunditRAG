"""下载 CRUD-RAG split_merged.json 并查看结构。"""
import requests
import json
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent
RAW_DIR = EVAL_ROOT / "raw" / "crudrag"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0"}

url = "https://raw.githubusercontent.com/IAAR-Shanghai/CRUD_RAG/master/data/crud_split/split_merged.json"
target = RAW_DIR / "split_merged.json"

if target.exists() and target.stat().st_size > 0:
    print(f"[skip] 已存在 {target.stat().st_size}B")
else:
    print(f"[下载] {url}")
    r = requests.get(url, headers=HEADERS, timeout=300)
    r.raise_for_status()
    target.write_bytes(r.content)
    print(f"[完成] {len(r.content)}B")

# 查看结构
data = json.loads(target.read_text(encoding="utf-8"))
print(f"类型: {type(data)}")
if isinstance(data, dict):
    print(f"顶层键: {list(data.keys())[:20]}")
    for k in list(data.keys())[:3]:
        v = data[k]
        print(f"  {k}: {type(v)} len={len(v) if hasattr(v, '__len__') else '?'}")
        if isinstance(v, list) and v:
            print(f"    元素字段: {list(v[0].keys()) if isinstance(v[0], dict) else v[0]}")
elif isinstance(data, list):
    print(f"列表长度: {len(data)}")
    print(f"元素字段: {list(data[0].keys()) if data and isinstance(data[0], dict) else '?'}")
    if data and isinstance(data[0], dict):
        for k, val in data[0].items():
            s = str(val)
            print(f"  {k}: {s[:150]}")
