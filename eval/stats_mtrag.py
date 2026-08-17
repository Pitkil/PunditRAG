"""统计 MTRAG RAG.jsonl 任务分布，确定抽样策略。"""
import json
from collections import Counter
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent
path = EVAL_ROOT / "raw" / "mtrag" / "RAG.jsonl"

lines = path.read_text(encoding="utf-8").strip().splitlines()
tasks = [json.loads(line) for line in lines if line.strip()]
print(f"总任务数: {len(tasks)}")

print("\nAnswerability 分布:")
for k, v in Counter(t.get("Answerability", ["?"])[0] if isinstance(t.get("Answerability"), list) else t.get("Answerability") for t in tasks).most_common():
    print(f"  {k}: {v}")

print("\nMulti-Turn 分布:")
cnt = Counter()
for t in tasks:
    mt = t.get("Multi-Turn", ["N/A"])
    if isinstance(mt, list):
        cnt[mt[0] if mt else "N/A"] += 1
    else:
        cnt[str(mt)] += 1
for k, v in cnt.most_common():
    print(f"  {k}: {v}")

print("\n数据集(Collection)分布:")
for k, v in Counter(t.get("Collection", "?") for t in tasks).most_common():
    print(f"  {k}: {v}")

# 看一个 ANSWERABLE 任务的完整结构
print("\n=== 第一个 ANSWERABLE 任务 ===")
for t in tasks:
    a = t.get("Answerability")
    if isinstance(a, list) and a and a[0] == "ANSWERABLE":
        print("字段:", list(t.keys()))
        print("input:", json.dumps(t["input"], ensure_ascii=False)[:500])
        print("targets:", json.dumps(t["targets"], ensure_ascii=False)[:300])
        print("contexts 数量:", len(t.get("contexts", [])))
        if t.get("contexts"):
            print("context[0] text 前200:", t["contexts"][0]["text"][:200])
        break
