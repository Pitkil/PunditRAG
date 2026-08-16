import json
import os
import shutil
import uuid
from datetime import datetime
from mimetypes import guess_type
from pathlib import Path
from typing import Any, Dict, List

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from minio.deleteobjects import DeleteObject
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from app.clients.milvus_utils import get_milvus_client
from app.clients.minio_utils import get_minio_client
from app.clients.mongo_workspace_utils import (
    create_document,
    create_knowledge_base,
    delete_document_record,
    delete_knowledge_base_record,
    ensure_document_active,
    get_document,
    list_documents,
    list_knowledge_bases,
    update_document,
    update_knowledge_base,
)
from app.conf.milvus_config import milvus_config
from app.conf.minio_config import minio_config
from app.core.logger import PROJECT_ROOT, logger
from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.nodes.node_file_to_md import SUPPORTED_CONVERT_EXTENSIONS
from app.import_process.agent.state import get_default_state
from app.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    get_done_task_list,
    get_running_task_list,
    get_task_result,
    get_task_status,
    set_task_result,
    update_task_status,
)


app = FastAPI(title="PunditRAG import service", description="知识库与文档导入服务")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_UPLOAD_EXTENSIONS = {".pdf", ".md", *SUPPORTED_CONVERT_EXTENSIONS}


@app.get("/")
def index():
    return RedirectResponse(url="/import/html")


@app.get("/import/html")
def return_import_html():
    html_path = PROJECT_ROOT / "app" / "import_process" / "page" / "import.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="未找到导入页面")
    return FileResponse(path=html_path, media_type=guess_type(html_path.name)[0])


@app.get("/health")
def health():
    return {"status": "ok"}


def invoke_import_graph(
    task_id: str,
    local_dir: str,
    local_file_path: str,
    kb_id: str,
    document_id: str,
):
    try:
        ensure_document_active(document_id)
        suffix = Path(local_file_path).suffix.lower()
        total_steps = 7 if suffix in SUPPORTED_CONVERT_EXTENSIONS else 6
        set_task_result(task_id, "total_steps", str(total_steps))
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        update_document(document_id, status="processing", error="")
        state = get_default_state()
        state.update(
            task_id=task_id,
            local_dir=local_dir,
            local_file_path=local_file_path,
            kb_id=kb_id,
            document_id=document_id,
        )
        final_state = kb_import_app.invoke(state)
        ensure_document_active(document_id)
        update_document(
            document_id,
            status="completed",
            chunk_count=len(final_state.get("chunks", [])),
            file_title=final_state.get("file_title", ""),
        )
        update_task_status(task_id, TASK_STATUS_COMPLETED)
    except Exception as exc:
        set_task_result(task_id, "error", str(exc))
        if get_document(document_id):
            update_document(document_id, status="failed", error=str(exc))
        update_task_status(task_id, TASK_STATUS_FAILED)
        if "导入任务已取消" in str(exc):
            logger.warning(f"task_id={task_id} 导入已取消：{exc}")
        else:
            logger.exception(f"task_id={task_id} 导入失败：{exc}")


@app.post("/upload")
async def upload(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    kb_id: str = Form(...),
):
    knowledge_bases = {item["kb_id"] for item in list_knowledge_bases()}
    if kb_id not in knowledge_bases:
        raise HTTPException(status_code=404, detail="知识库不存在")

    task_ids = []
    document_ids = []
    base_dir = PROJECT_ROOT / "temp-files" / "imports" / datetime.now().strftime("%Y%m%d")
    for file in files:
        filename = file.filename or "unknown_file"
        if Path(filename).suffix.lower() not in SUPPORTED_UPLOAD_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"暂不支持该文件类型：{filename}")

        task_id = str(uuid.uuid4())
        task_dir = base_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        local_path = task_dir / filename
        local_path.write_bytes(await file.read())
        document = create_document(kb_id, filename, str(local_path), task_id)

        task_ids.append(task_id)
        document_ids.append(document["document_id"])
        background_tasks.add_task(
            invoke_import_graph,
            task_id,
            str(task_dir),
            str(local_path),
            kb_id,
            document["document_id"],
        )

    return {
        "code": 200,
        "message": "文件已上传，正在解析",
        "task_ids": task_ids,
        "document_ids": document_ids,
        "kb_id": kb_id,
    }


@app.get("/status/{task_id}")
def get_task_progress(task_id: str):
    status = get_task_status(task_id)
    done_list = get_done_task_list(task_id)
    running_list = get_running_task_list(task_id)
    total_steps = int(get_task_result(task_id, "total_steps", "6"))
    if status == TASK_STATUS_COMPLETED:
        progress = 100
    else:
        running_progress = 0.35 if running_list else 0
        progress = min(99, round((len(done_list) + running_progress) / total_steps * 100))
    task_status: Dict[str, Any] = {
        "code": 200,
        "task_id": task_id,
        "status": status,
        "done_list": done_list,
        "running_list": running_list,
        "completed_steps": len(done_list),
        "total_steps": total_steps,
        "progress": progress,
        "error": get_task_result(task_id, "error"),
    }
    return task_status


class KnowledgeBasePayload(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    description: str = Field("", max_length=300)


@app.get("/knowledge-bases")
def get_knowledge_bases():
    return {"items": list_knowledge_bases()}


@app.post("/knowledge-bases")
def add_knowledge_base(payload: KnowledgeBasePayload):
    return create_knowledge_base(payload.name, payload.description)


@app.patch("/knowledge-bases/{kb_id}")
def edit_knowledge_base(kb_id: str, payload: KnowledgeBasePayload):
    result = update_knowledge_base(kb_id, payload.model_dump())
    if not result:
        raise HTTPException(status_code=404, detail="知识库不存在")
    return result


def _delete_vectors(document_id: str) -> None:
    client = get_milvus_client()
    if not client:
        return
    expr = f"document_id == {json.dumps(document_id, ensure_ascii=False)}"
    for collection in (milvus_config.chunks_collection, milvus_config.item_name_collection):
        if collection and client.has_collection(collection):
            client.delete(collection, filter=expr)


def _delete_local_artifacts(document: Dict[str, Any]) -> None:
    local_path = document.get("local_path")
    if not local_path:
        return
    path = Path(local_path).resolve()
    imports_root = (PROJECT_ROOT / "temp-files" / "imports").resolve()
    if imports_root in path.parents and path.parent != imports_root:
        shutil.rmtree(path.parent, ignore_errors=True)
    elif path.is_file():
        path.unlink(missing_ok=True)


def _delete_minio_artifacts(document_id: str) -> None:
    image_root = str(minio_config.minio_img_dir or "").strip("/")
    prefix = f"{image_root}/{document_id}/".lstrip("/")
    client = get_minio_client()
    objects = client.list_objects(
        bucket_name=str(minio_config.bucket_name),
        prefix=prefix,
        recursive=True,
    )
    errors = client.remove_objects(
        bucket_name=str(minio_config.bucket_name),
        delete_object_list=(DeleteObject(str(item.object_name)) for item in objects),
    )
    for error in errors:
        logger.warning(f"MinIO对象删除失败：{error}")


@app.get("/knowledge-bases/{kb_id}/documents")
def get_knowledge_documents(kb_id: str):
    return {"items": list_documents(kb_id)}


@app.delete("/documents/{document_id}")
def remove_document(document_id: str):
    document = get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")
    update_document(document_id, status="deleting")
    _delete_vectors(document_id)
    _delete_minio_artifacts(document_id)
    _delete_local_artifacts(document)
    delete_document_record(document_id)
    return {"deleted": True, "document_id": document_id}


@app.delete("/knowledge-bases/{kb_id}")
def remove_knowledge_base(kb_id: str):
    documents = list_documents(kb_id)
    for document in documents:
        update_document(document["document_id"], status="deleting")
    for document in documents:
        _delete_vectors(document["document_id"])
        _delete_minio_artifacts(document["document_id"])
        _delete_local_artifacts(document)
    if not delete_knowledge_base_record(kb_id):
        raise HTTPException(status_code=404, detail="知识库不存在")
    return {"deleted": True, "kb_id": kb_id}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
