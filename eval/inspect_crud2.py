"""查看 CRUD-RAG 各问答子集结构。"""
import json
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent
path = EVAL_ROOT / "raw" / "crudrag" / "split_merged.json"
data = json.loads(path.read_text(encoding="utf-8"))

for key in ["questanswer_1doc", "questanswer_2docs", "questanswer_3docs"]:
    items = data.get(key, [])
    print(f"=== {key}: {len(items)} 条 ===")
    if items:
        print(f"  字段: {list(items[0].keys())}")
        for k, v in items[0].items():
            print(f"    {k}: {str(v)[:150]}")
    print()
