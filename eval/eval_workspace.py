"""评测运行的临时工作区标识与正常退出清理。"""

import os
from datetime import datetime, timezone


def new_eval_run_id() -> str:
    return os.getenv("EVAL_RUN_ID", "").strip() or datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S%fZ")


class EvalWorkspace:
    def __init__(self, http_json, import_api: str, query_api: str, run_id: str):
        self.http_json = http_json
        self.import_api = import_api
        self.query_api = query_api
        self.run_id = run_id
        self.kb_id = ""
        self.owns_kb = False
        self.session_ids: list[str] = []

    def use_knowledge_base(self, kb_id: str, owns_kb: bool) -> str:
        self.kb_id = kb_id
        self.owns_kb = owns_kb
        return kb_id

    def session_id(self, dataset: str, case_id: str) -> str:
        session_id = f"{dataset}-{self.run_id}-{case_id}"
        self.session_ids.append(session_id)
        return session_id

    def cleanup(self) -> None:
        if os.getenv("EVAL_KEEP_ARTIFACTS", "").strip().lower() in {"1", "true", "yes"}:
            print("已按 EVAL_KEEP_ARTIFACTS 保留本轮评测知识库和会话")
            return

        for session_id in dict.fromkeys(self.session_ids):
            try:
                self.http_json(
                    "DELETE",
                    f"{self.query_api}/sessions/{session_id}",
                    timeout=120,
                )
            except Exception as exc:
                print(f"清理评测会话失败: session={session_id} error={exc}")
        if self.owns_kb and self.kb_id:
            try:
                self.http_json(
                    "DELETE",
                    f"{self.import_api}/knowledge-bases/{self.kb_id}",
                    timeout=300,
                )
            except Exception as exc:
                print(f"清理评测知识库失败: kb={self.kb_id} error={exc}")
