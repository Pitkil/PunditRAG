"""查看 RGB 数据集 JSON 结构。"""
import json
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw" / "rgb"

for name in ["zh.json", "zh_fact.json", "zh_int.json", "zh_refine.json"]:
    path = RAW_DIR / name
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    data = [json.loads(line) for line in lines if line.strip()]
    print(f"=== {name}: {len(data)} 条 ===")
    if data:
        sample = data[0]
        print("  字段:", list(sample.keys()))
        for k, v in sample.items():
            s = str(v)
            print(f"    {k}: {s[:200]}")
    print()
