from datetime import datetime
from mimetypes import guess_type
from pathlib import Path
from typing import Any, Dict, List
import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from app.core.logger import logger, PROJECT_ROOT
from app.import_process.agent.state import get_default_state

from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
from app.import_process.agent.main_graph import kb_import_app

app = FastAPI(title="import service",description = "导入文件处理")

#跨域请求（CORS）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],#无条件允许任何人、任何域名、任何 IP 来访问本接口
    allow_methods=["*"],#允许所有的 HTTP 请求动作
    allow_headers=["*"],#允许请求头里带任何信息
)

'''
返回import_html文件
'''
@app.get("/import/html")
def return_import_html():
    html_file_obj = PROJECT_ROOT / "app" / "import_process" / "page" / "import.html"
    if not html_file_obj.exists():
        logger.error(f"没有找到对应的前端文件,返回404异常!")
        raise HTTPException(status_code=404,detail="没有找到对应的前端文件")
    return FileResponse(
        path=html_file_obj,
        #拿到文件名import.html 猜测出这属于网页，返回一个结果：('text/html', None) 取第一个
        media_type=guess_type(html_file_obj.name)[0]
    )

'''
执行图
'''
def invoke_import_graph(task_id:str,local_dir:str,local_file_path:str):
    try:
        update_task_status(task_id,TASK_STATUS_PROCESSING)
        state = get_default_state()
        state["task_id"] = task_id
        state["local_dir"] = local_dir
        state["local_file_path"] = local_file_path
        kb_import_app.invoke(state)
        update_task_status(task_id,TASK_STATUS_COMPLETED)
    except Exception as e:
       update_task_status(task_id,TASK_STATUS_FAILED)     # 任务失败状态 解析失败
       logger.exception(f"task_id={task_id}任务的导入流程发生异常!{str(e)}")

'''
接收上传文件
'''
@app.post("/upload")
async def upload(backgroundtasks:BackgroundTasks,files:List[UploadFile] = File(...)):
    task_ids = []
    local_dir_obj = PROJECT_ROOT / "temp-files" / "imports" / datetime.now().strftime("%Y%m%d")
    for file in files:
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        task_dir_obj = local_dir_obj / task_id
        task_dir_obj.mkdir(parents=True,exist_ok=True)
        filename = file.filename or "unknown_file"
        local_file_path_obj = task_dir_obj / filename
        content = await file.read()
        local_file_path_obj.write_bytes(content)

         #异步调用 import_graph_app
        backgroundtasks.add_task(invoke_import_graph,
                                 task_id = task_id,
                                 local_dir = str(task_dir_obj),
                                 local_file_path=str(local_file_path_obj)
                                 )

    return {
        "code":200,
        "message":"文件已经上传，正在解析！",
        "task_ids":task_ids
    }

'''
向后端查询任务状态接口 
'''
@app.get("/status/{task_id}")
async def get_task_progress(task_id: str):
    task_status_info: Dict[str, Any] = {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),  # 任务全局状态：pending/processing/completed/failed
        "done_list": get_done_task_list(task_id),  # 已完成的节点/阶段列表
        "running_list": get_running_task_list(task_id)  # 正在运行的节点/阶段列表
    }
    logger.info(
        f"[{task_id}] 任务状态查询，当前状态：{task_status_info['status']}，已完成节点：{task_status_info['done_list']}")
    return task_status_info

 
if __name__ == "__main__":
    uvicorn.run(
        app,
        host = "0.0.0.0",
        port = 8000,
    )

    

