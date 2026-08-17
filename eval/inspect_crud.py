"""查看 CRUD-RAG 官方数据结构，评估可接入性。"""
import requests

HEADERS = {"User-Agent": "Mozilla/5.0"}
repo = "IAAR-Shanghai/CRUD_RAG"

paths = [
    "https://api.github.com/repos/IAAR-Shanghai/CRUD_RAG/contents/data/crud_split",
    "https://api.github.com/repos/IAAR-Shanghai/CRUD_RAG/contents/data/crud",
]

for url in paths:
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code == 200:
            items = r.json()
            names = [
                i["name"] + ("/" if i["type"] == "dir" else f" ({i.get('size', 0)}B)")
                for i in items
            ]
            print(f"{url.split('/contents/')[-1]}: {names}")
        else:
            print(f"{url.split('/contents/')[-1]}: HTTP {r.status_code} {r.text[:120]}")
    except Exception as e:
        print(f"{url}: FAIL {e}")
