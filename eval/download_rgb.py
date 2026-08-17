"""通过 GitHub API 获取 RGB 数据集文件的真实下载地址并下载。"""
import requests
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent
RAW_DIR = EVAL_ROOT / "raw" / "rgb"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0"}

# 先查默认分支
repo = "chen700564/RGB"
r = requests.get(f"https://api.github.com/repos/{repo}", headers=HEADERS, timeout=15)
default_branch = r.json().get("default_branch", "main")
print(f"默认分支: {default_branch}")

# 列出 data 目录文件及 download_url
r = requests.get(f"https://api.github.com/repos/{repo}/contents/data?ref={default_branch}", headers=HEADERS, timeout=15)
if r.status_code != 200:
    print(f"列出 data 目录失败: HTTP {r.status_code} {r.text[:200]}")
    raise SystemExit(1)

files = r.json()
for f in files:
    print(f"{f['name']} -> {f['download_url']}")

# 下载 zh 系列（中文评测）
wanted = ["zh.json", "zh_fact.json", "zh_int.json", "zh_refine.json"]
for f in files:
    name = f["name"]
    if name not in wanted:
        continue
    target = RAW_DIR / name
    if target.exists() and target.stat().st_size > 0:
        print(f"[skip] {name} 已存在")
        continue
    url = f["download_url"]
    print(f"[下载] {name} <- {url}")
    rr = requests.get(url, headers=HEADERS, timeout=180)
    rr.raise_for_status()
    target.write_bytes(rr.content)
    print(f"[完成] {name} {len(rr.content)}B")
