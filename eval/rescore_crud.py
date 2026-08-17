"""CRUD-RAG 结果重新评分：用关键信息点匹配代替前20字包含判定。

从官方答案提取关键信息（数字/日期/引号内容/书名号内容/关键短语），
判断生成回答是否包含这些要点（多数命中即正确），更接近官方语义评估。
"""
import json
import re
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent
RESULT_FILE = EVAL_ROOT / "results" / "result_crudrag_qa1.json"


def extract_key_points(gold: str) -> list:
    """从官方答案提取关键信息点。"""
    points = []
    # 日期时间：2023年7月29日 / 2023年8月9日15时34分
    for m in re.findall(r"\d{4}年\d{1,2}月\d{1,2}日(?:\d{1,2}时\d{1,2}分)?", gold):
        points.append(m)
    # 数字（含小数、百分比）
    for m in re.findall(r"\d+(?:\.\d+)?(?:%|亿|万|公里|个|项|人|元|吨|名|辆|套)?", gold):
        if m and m.strip("0123456789.%") in ("", "个", "项", "人", "元", "吨", "名", "辆", "套", "亿", "万", "公里"):
            points.append(m)
    # 书名号内容
    for m in re.findall(r"[《〈「「]([^》〉」」]{1,40})[》〉」」]", gold):
        points.append(m)
    # 引号内容
    for m in re.findall(r"“([^”]{1,40})”|‘([^’]{1,40})’", gold):
        points.append(m[0] or m[1])
    # 关键短语（含中文引号或特殊术语）
    for m in re.findall(r"[\u4e00-\u9fff]{4,20}(?:提取法|技术|协议|工程|项目|行动|计划|方案|机制|特点|方式|方法|标准|措施)", gold):
        points.append(m)
    return points


def normalize(s: str) -> str:
    return "".join(s.split()).lower()


def rescore():
    data = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
    results = data["results"]
    new_results = []
    for r in results:
        gold = r["gold_answer"]
        answer = r["answer"]
        points = extract_key_points(gold)
        # 去重
        seen = set()
        unique = []
        for p in points:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        ans_n = normalize(answer)
        gold_n = normalize(gold)
        hits = [p for p in unique if normalize(p) in ans_n]
        # 判定：官方答案整体包含 或 关键点命中率>=60%
        overall = gold_n in ans_n or ans_n in gold_n
        hit_rate = len(hits) / len(unique) if unique else 0
        correct = overall or hit_rate >= 0.6
        new_results.append(
            {
                **r,
                "key_points": unique,
                "key_point_hits": hits,
                "key_point_hit_rate": round(hit_rate, 2),
                "answer_correct": correct,
            }
        )
        print(f"qid={r['qid']} correct={correct} hit_rate={hit_rate:.2f} points={unique} hits={hits}")
        print(f"    answer: {answer[:100]}")

    n = len(new_results)
    acc = sum(1 for r in new_results if r["answer_correct"]) / n
    src = sum(1 for r in new_results if r["source_hit"]) / n
    print("=" * 70)
    print(f"重新评分：回答正确率 {acc:.2%} ({int(acc * n)}/{n})，来源命中率 {src:.2%}")
    data["metrics"]["answer_accuracy"] = round(acc, 4)
    data["results"] = new_results
    data["metrics"]["rescore_note"] = "已用关键信息点匹配重新评分（命中率>=60%或整体包含判定为正确）"
    RESULT_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已更新: {RESULT_FILE}")


if __name__ == "__main__":
    rescore()
