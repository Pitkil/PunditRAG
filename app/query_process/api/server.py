import os
import uuid
from mimetypes import guess_type
from threading import Lock
from typing import Literal

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from app.clients.mongo_history_utils import get_recent_messages
from app.clients.mongo_workspace_utils import (
    count_chat_sessions,
    delete_chat_session,
    ensure_chat_session,
    get_chat_session,
    get_document,
    list_chat_sessions,
    rename_chat_session,
    list_knowledge_bases,
)
from app.clients.milvus_utils import get_milvus_client
from app.core.logger import PROJECT_ROOT, logger
from app.query_process.agent.main_graph import query_app
from app.query_process.agent.state import create_query_default_state
from app.utils.sse_utils import SSEEvent, create_sse_queue, push_to_session, sse_generator
from app.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    clear_task,
    get_done_task_list,
    get_running_task_list,
    get_task_result,
    get_task_status,
    get_task_trace,
    set_task_result,
    update_task_status,
)
from app.utils.api_utils import get_cors_origins


app = FastAPI(title="PunditRAG query service", description="知识库查询与会话服务")
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)

_active_runs: dict[str, set[str]] = {}
_active_runs_lock = Lock()


def _register_run(session_id: str, run_id: str) -> None:
    with _active_runs_lock:
        _active_runs.setdefault(session_id, set()).add(run_id)


def _finish_run(session_id: str, run_id: str) -> None:
    with _active_runs_lock:
        runs = _active_runs.get(session_id)
        if not runs:
            return
        runs.discard(run_id)
        if not runs:
            _active_runs.pop(session_id, None)


def _session_is_running(session_id: str) -> bool:
    with _active_runs_lock:
        return bool(_active_runs.get(session_id))


@app.get("/")
def index():
    return RedirectResponse(url="/query/html")


@app.get("/query/html")
def return_query_html():
    html_path = PROJECT_ROOT / "app" / "query_process" / "page" / "chat.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="未找到问答页面")
    return FileResponse(path=html_path, media_type=guess_type(html_path.name)[0])


@app.get("/health")
def health():
    checks = {}
    try:
        knowledge_bases = list_knowledge_bases()
        checks["mongo"] = {"status": "ok", "knowledge_bases": len(knowledge_bases)}
    except Exception as exc:
        checks["mongo"] = {"status": "error", "detail": str(exc)}
    try:
        milvus = get_milvus_client()
        if not milvus:
            raise RuntimeError("Milvus client unavailable")
        checks["milvus"] = {"status": "ok", "collections": len(milvus.list_collections())}
    except Exception as exc:
        checks["milvus"] = {"status": "error", "detail": str(exc)}

    healthy = all(check["status"] == "ok" for check in checks.values())
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={"status": "ok" if healthy else "degraded", "checks": checks},
    )


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="查询内容")
    session_id: str | None = Field(None, description="会话 ID")
    scope_mode: Literal["all", "knowledge_base", "documents"] = Field(
        "knowledge_base", description="资料范围模式"
    )
    kb_ids: list[str] = Field(default_factory=list, description="知识库 ID 列表")
    document_ids: list[str] = Field(default_factory=list, description="文档 ID 列表")
    is_stream: bool = Field(False, description="是否流式返回")
    enable_web_search: bool = Field(False, description="是否启用联网补充")


def resolve_query_scope(req: QueryRequest) -> tuple[list[str], list[str]]:
    """把用户选择解析成明确范围，空数组不再同时表示“全部”和“无范围”。"""
    knowledge_bases = list_knowledge_bases()
    known_kb_ids = {item["kb_id"] for item in knowledge_bases}

    if req.scope_mode == "all":
        return sorted(known_kb_ids), []

    if req.scope_mode == "documents":
        if not req.document_ids:
            raise HTTPException(status_code=400, detail="文档范围不能为空")
        documents = [get_document(document_id) for document_id in dict.fromkeys(req.document_ids)]
        unknown_document_ids = [
            document_id
            for document_id, document in zip(dict.fromkeys(req.document_ids), documents)
            if not document or document.get("status") in {"deleting", "deleted"}
        ]
        if unknown_document_ids:
            raise HTTPException(status_code=404, detail={"unknown_document_ids": unknown_document_ids})
        document_ids = [document["document_id"] for document in documents]
        kb_ids = sorted({document["kb_id"] for document in documents})
        return kb_ids, document_ids

    unknown_kb_ids = sorted(set(req.kb_ids) - known_kb_ids)
    if unknown_kb_ids:
        raise HTTPException(status_code=404, detail={"unknown_kb_ids": unknown_kb_ids})
    return list(dict.fromkeys(req.kb_ids)), []


def run_query_graph(
    session_id: str,
    run_id: str,
    query: str,
    scope_mode: str,
    kb_ids: list[str],
    document_ids: list[str],
    is_stream: bool,
    enable_web_search: bool,
):
    try:
        clear_task(run_id)
        ensure_chat_session(session_id, query)
        update_task_status(run_id, TASK_STATUS_PROCESSING, is_stream)
        initial_state = create_query_default_state(
            session_id=session_id,
            run_id=run_id,
            original_query=query,
            scope_mode=scope_mode,
            kb_ids=kb_ids,
            document_ids=document_ids,
            is_stream=is_stream,
            enable_web_search=enable_web_search,
        )
        final_state = query_app.invoke(initial_state)
        update_task_status(run_id, TASK_STATUS_COMPLETED, is_stream)
        if is_stream:
            push_to_session(
                run_id,
                SSEEvent.FINAL,
                {
                    "answer": final_state.get("answer", ""),
                    "status": "completed",
                    "image_urls": final_state.get("image_urls", []),
                    "sources": final_state.get("sources", []),
                },
            )
        return final_state
    except Exception as exc:
        logger.exception(f"session_id={session_id} run_id={run_id} 查询失败：{exc}")
        set_task_result(run_id, "error", str(exc))
        update_task_status(run_id, TASK_STATUS_FAILED, is_stream)
        if is_stream:
            push_to_session(run_id, SSEEvent.ERROR, {"error": str(exc)})
        return None
    finally:
        _finish_run(session_id, run_id)


@app.post("/query")
async def query(req: QueryRequest, background_tasks: BackgroundTasks):
    resolved_kb_ids, resolved_document_ids = resolve_query_scope(req)
    session_id = req.session_id or str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    _register_run(session_id, run_id)
    if req.is_stream:
        create_sse_queue(run_id)
        background_tasks.add_task(
            run_query_graph,
            session_id,
            run_id,
            req.query,
            req.scope_mode,
            resolved_kb_ids,
            resolved_document_ids,
            True,
            req.enable_web_search,
        )
        return {"message": "结果正在输出", "session_id": session_id, "run_id": run_id}

    final_state = run_query_graph(
        session_id,
        run_id,
        req.query,
        req.scope_mode,
        resolved_kb_ids,
        resolved_document_ids,
        False,
        req.enable_web_search,
    )
    if final_state is None:
        raise HTTPException(
            status_code=500,
            detail=get_task_result(run_id, "error") or "查询处理失败",
        )
    return {
        "message": "处理完成",
        "session_id": session_id,
        "run_id": run_id,
        "answer": final_state.get("answer", ""),
        "error": get_task_result(run_id, "error"),
        "image_urls": final_state.get("image_urls", []),
        "sources": final_state.get("sources", []),
        "done_list": get_done_task_list(run_id),
    }


@app.get("/status/{task_id}")
def get_query_status(task_id: str):
    return {
        "task_id": task_id,
        "status": get_task_status(task_id),
        "answer": get_task_result(task_id, "answer"),
        "error": get_task_result(task_id, "error"),
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id),
        "trace": get_task_trace(task_id),
    }


@app.get("/query/stream/{run_id}")
async def stream_query_result(run_id: str, request: Request):
    return StreamingResponse(
        sse_generator(run_id, request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/history/{session_id}")
def get_history(session_id: str, limit: int = 50):
    messages = []
    for item in get_recent_messages(session_id, limit):
        messages.append(
            {
                "_id": str(item.get("_id", "")),
                "session_id": item.get("session_id", ""),
                "role": item.get("role", ""),
                "text": item.get("text", ""),
                "rewritten_query": item.get("rewritten_query", ""),
                "item_names": item.get("item_names", []),
                "image_urls": item.get("image_urls", []),
                "sources": item.get("sources", []),
                "kb_ids": item.get("kb_ids", []),
                "document_ids": item.get("document_ids", []),
                "ts": item.get("ts"),
            }
        )
    return {"session_id": session_id, "messages": messages}


class SessionPayload(BaseModel):
    title: str = Field("新对话", min_length=1, max_length=80)


@app.get("/sessions")
def get_sessions(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    total = count_chat_sessions()
    items = list_chat_sessions(limit=limit, offset=offset)
    return {
        "items": items,
        "total": total,
        "has_more": offset + len(items) < total,
    }


@app.post("/sessions")
def create_session(payload: SessionPayload):
    return ensure_chat_session(str(uuid.uuid4()), payload.title)


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    result = get_chat_session(session_id)
    if not result:
        raise HTTPException(status_code=404, detail="会话不存在")
    return result


@app.patch("/sessions/{session_id}")
def edit_session(session_id: str, payload: SessionPayload):
    result = rename_chat_session(session_id, payload.title)
    if not result:
        raise HTTPException(status_code=404, detail="会话不存在")
    return result


@app.delete("/sessions/{session_id}")
def remove_session(session_id: str):
    if _session_is_running(session_id):
        raise HTTPException(status_code=409, detail="该会话正在生成答案，请完成后再删除")
    delete_chat_session(session_id)
    return {"deleted": True, "session_id": session_id}


@app.get("/settings/status")
def settings_status():
    return {
        "llm": bool(os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_BASE_URL")),
        "milvus": bool(os.getenv("MILVUS_URL")),
        "mongo": bool(os.getenv("MONGO_URL")),
        "minio": bool(os.getenv("MINIO_ENDPOINT")),
        "web_search": bool(
            os.getenv("MCP_DASHSCOPE_BASE_URL") and os.getenv("OPENAI_API_KEY")
        ),
        "model": os.getenv("LLM_DEFAULT_MODEL", "未配置"),
        "embedding_model": os.getenv("BGE_M3_PATH", os.getenv("BGE_M3", "BGE-M3")),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
