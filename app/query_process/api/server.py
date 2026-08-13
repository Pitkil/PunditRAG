from mimetypes import guess_type
from pathlib import Path
import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from app.clients.mongo_history_utils import clear_history as clear_chat_history
from app.clients.mongo_history_utils import get_recent_messages
from app.core.logger import PROJECT_ROOT, logger

from app.query_process.agent.state import create_query_default_state
from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
from app.query_process.agent.main_graph import query_app


# 定义fastapi对象
app = FastAPI(title="query service", description="掌柜智库查询服务！")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def index():
    return RedirectResponse(url="/query/html")

@app.get("/query/html")
def return_query_html():
    html_path_obj = PROJECT_ROOT / "app" / "query_process" / "page" / "chat.html"
    if not html_path_obj.exists():
        logger.error(f"没有找到对应的前端文件,返回404异常!")
        raise HTTPException(status_code=404, detail="没有找到对应的前端文件")
    return FileResponse(
        path=html_path_obj,
        media_type = guess_type(html_path_obj.name)[0],
    )

@app.get("/health")
def health():
    return {"status": "ok"}

# 定义接口接收的数据结构
class QueryRequest(BaseModel):
    """查询请求数据结构"""
    query: str = Field(..., description="查询内容")
    session_id: str | None = Field(None, description="会话ID")
    is_stream: bool = Field(False, description="是否流式返回")

def run_query_graph(session_id: str, query: str, is_stream: bool):
    try:
        #清空缓存
        clear_task(session_id)
        
        update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)
        initial_state = create_query_default_state(session_id=session_id, original_query=query, is_stream=is_stream)
        final_state = query_app.invoke(initial_state)
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream) 
        #final事件一定最后推送，因为会关闭本次连接流
        push_to_session(
                    session_id,
                    SSEEvent.FINAL,
                    {
                        "answer": get_task_result(session_id, "answer"),
                        "status": "completed",
                        "image_urls": final_state.get("image_urls", [])
                    }
                )
        return final_state
    except Exception as e:
        logger.exception(f"task_id={session_id}任务的查询流程发生异常!{str(e)}")
        error_message = str(e)
        set_task_result(session_id, "error", error_message)
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        if is_stream:
            push_to_session(session_id, SSEEvent.ERROR, {"error": error_message})
        return None

#触发查询流程
@app.post("/query")
async def query(req: QueryRequest,background_tasks: BackgroundTasks):
    """
    /query
    作用：触发查询流程的执行  query_graph的执行.... (意图确认 / 多路召回 / 粗排序 / 精排序 / 结果输出...)
    形式：is_stream = true 异步执行以上过程    is_stream = false 同步执行以上过程
    过程：
        1. 获取了核心参数 is_stream | session_id | query
        2. 判断是否是流式
           是
               [也]要执行main_graph [2]
               接口 backgroundtask.add_task(执行main_graph)  提取一个函数
               组装json
               return
           否
               直接执行main_graph [1]   直接写
               并获取结果
               组装json即可
               return
        3.因为异步和同步都需要执行main_grap,所以我们将执行过程提取出去
"""
    is_stream  =   req.is_stream
    session_id = req.session_id or str(uuid.uuid4())
    query = req.query

    if is_stream:
        create_sse_queue(session_id)
        background_tasks.add_task(run_query_graph, session_id, query, is_stream)
        return  {"messages":"结果正在输出...","session_id":session_id}
    else:
        final_state = run_query_graph(session_id, query, is_stream)
        answer = get_task_result(session_id,"answer")
        return {
            "messages": "处理完成!" if final_state is not None else "处理失败!",
            "session_id": session_id,
            "answer": answer,
            "error": get_task_result(session_id, "error"),
            "image_urls": final_state.get("image_urls", []) if final_state else [],
            "done_list": get_done_task_list(session_id),
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
    }

#流式获取接口
@app.get("/query/stream/{session_id}")
async def stream_query_result(session_id: str, request: Request):
    """
    /query/stream/{session_id}
    作用：流式获取查询结果
    过程：
        1. 获取session_id
        2. 调用sse_generator(session_id) 生成器
        3. StreamingResponse返回
    """
    return StreamingResponse(
        sse_generator(session_id, request), 
        media_type="text/event-stream")

#查询历史聊天记录
@app.get("/history/{session_id}")
def get_history(session_id:str,limit = 10):
   message_list =  get_recent_messages(session_id, limit)
   formatted_messages = []
  # MongoDB 的主键 _id 在数据库内部是一个特殊的 ObjectId 对象，而不是字符串。
   for m in message_list:
       formatted_messages.append({
           "_id": str(m.get("_id")) if m.get("_id") is not None else "",
            "session_id": m.get("session_id", ""),
            "role": m.get("role", ""),
            "text": m.get("text", ""),
            "rewritten_query": m.get("rewritten_query", ""),
            "item_names": m.get("item_names", []),
            "ts": m.get("ts")
       })
       
   return {"session_id": session_id,"messages":formatted_messages}

#清空历史聊天记录
@app.delete("/history/{session_id}")
@app.get("/history/{session_id}/clear")
def clear_history_endpoint(session_id:str):
    deleted_count = clear_chat_history(session_id)
    return {
        "messages": f"已清空会话 {session_id} 的历史记录，共删除 {deleted_count} 条记录。",
        "deleted_count": deleted_count
    }

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0",
        port=8001,
    )
