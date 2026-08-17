"""查看 MTRAG 官方数据结构，确定评测内容。"""
import requests

HEADERS = {"User-Agent": "Mozilla/5.0"}
repo = "IBM/mt-rag-benchmark"

paths = [
    "mtrag-human/conversations",
    "mtrag-human/generation_tasks",
    "mtrag-human/retrieval_tasks",
    "mtrag-human/evaluations",
    "mtrag-synthetic/conversations",
    "mtrag-synthetic/generation_tasks",
    "corpora/document_level",
    "corpora/passage_level",
]

for p in paths:
    url = f"https://api.github.com/repos/{repo}/contents/{p}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code == 200:
            items = r.json()
            names = [i["name"] + ("/" if i["type"] == "dir" else f" ({i.get('size',0)}B)") for i in items]
            print(f"{p}: {names}")
        else:
            print(f"{p}: HTTP {r.status_code}")
    except Exception as e:
        print(f"{p}: FAIL {e}")
