"""浏览官方评测集仓库结构，定位数据集文件。"""
import requests

headers = {"User-Agent": "Mozilla/5.0"}

paths = {
    "RGB/data": "https://api.github.com/repos/chen700564/RGB/contents/data",
    "CRUD_RAG/data": "https://api.github.com/repos/IAAR-Shanghai/CRUD_RAG/contents/data",
    "MTRAG/corpora": "https://api.github.com/repos/IBM/mt-rag-benchmark/contents/corpora",
    "MTRAG/mtrag-human": "https://api.github.com/repos/IBM/mt-rag-benchmark/contents/mtrag-human",
    "MTRAG/mtrag-synthetic": "https://api.github.com/repos/IBM/mt-rag-benchmark/contents/mtrag-synthetic",
}

for name, url in paths.items():
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            items = r.json()
            names = []
            for i in items:
                if i["type"] == "dir":
                    names.append(i["name"] + "/")
                else:
                    names.append(f'{i["name"]} ({i.get("size", 0)}B)')
            print(f"{name}: {names}")
        else:
            print(f"{name}: HTTP {r.status_code} {r.text[:120]}")
    except Exception as e:
        print(f"{name}: FAIL {e}")
