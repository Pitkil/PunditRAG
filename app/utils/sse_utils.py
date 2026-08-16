'''
实现了轻量级、基于内存的会话级 SSE 推送框架, 。
'''
import json
import queue
import asyncio
from typing import Dict, Any, Optional, AsyncGenerator
from fastapi import Request


class SSEEvent:
    READY = "ready"         # 连接建立
    PROGRESS = "progress"   # 任务节点进度
    DELTA = "delta"         # LLM 流式输出增量
    FINAL = "final"         # 最终完整答案
    ERROR = "error"         # 错误信息
    CLOSE = "__close__"     # 关闭连接信号

# 全局 SSE 会话队列存储
# Key: session_id, Value: queue.Queue
_session_stream: Dict[str, queue.Queue] = {}
SSE_DEBUG = False


def _debug(message: str) -> None:
    if SSE_DEBUG:
        print(message)

def get_sse_queue(session_id:str)->Optional[queue.Queue]:
    """
    获取指定 session_id 的 SSE 队列。
    如果不存在，则返回 None。
    """
    return _session_stream.get(session_id)\

def create_sse_queue(session_id:str)->queue.Queue:
    """
    创建一个新的 SSE 队列，并存储在全局字典中。
    """
    _debug(f"[SSE] Creating queue for session: {session_id}")
    q = queue.Queue()
    _session_stream[session_id] = q
    return q

def remove_sse_queue(session_id:str):
    """
    ：从全局字典里删除指定会话的队列，即注销该 SSE 会话。
    """
    if session_id in _session_stream:
        _debug(f"[SSE] Removing queue for session: {session_id}")
        _session_stream.pop(session_id, None)

def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    """
    把事件名和数据字典序列化成标准 SSE 文本格式.
    """
    payload = json.dumps(data, ensure_ascii=False)
    _debug(f"[SSE] Packing event: {event}, payload: {payload[:50]}...")
    return f"event: {event}\ndata: {payload}\n\n"

def push_to_session(session_id:str, event_type:str, data:Any):
    """
    向指定 session_id 的 SSE 队列中推送事件。
    """
    q = get_sse_queue(session_id)
    if q:
        event = {
            "event": event_type,
            "data": data
        }
        _debug(f"[SSE] Pushing to session {session_id}: {event}")
        q.put(event)
    else:
        _debug(f"[SSE] No queue found for session: {session_id}")

async def sse_generator(session_id: str, request: Request):
    """
    SSE 生成器，用于 FastAPI 的 StreamingResponse
    """
    _debug(f"[SSE] Generator started for session: {session_id}")
    stream_queue = get_sse_queue(session_id)
    if stream_queue is None:
        # 如果没有对应的队列，直接结束
        _debug(f"[SSE] Error: Queue not found for session {session_id}. Available sessions: {list(_session_stream.keys())}")
        return

    loop = asyncio.get_running_loop()
    try:
        # 发送连接建立信号
        _debug(f"[SSE] Sending ready signal for {session_id}")
        yield _sse_pack("ready", {})

        while True:
            # 若客户端断开，尽快退出
            if await request.is_disconnected():
                _debug(f"[SSE] Client disconnected: {session_id}")
                _debug("-----------------------断开连接--------------------")
                break

            try:
                # 使用 run_in_executor 避免阻塞 async 事件循环
                msg = await loop.run_in_executor(None, stream_queue.get, True, 1.0)
            except queue.Empty:
                # print(f"[SSE] Queue empty for {session_id}, waiting...")
                continue

            event = msg.get("event")
            data = msg.get("data")
            
            _debug(f"[SSE] Yielding event {event} for {session_id}")

            # 特殊关闭事件
            if event == "__close__":
                _debug(f"[SSE] Closing signal received for {session_id}")
                break

            yield _sse_pack(event, data)
    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
        _debug(f"[SSE] Client disconnected (Cancelled/Reset/Pipe): {session_id}")
        # 生成器被取消/对端断开：静默退出
        return
    except Exception as e:
        _debug(f"[SSE] Exception in generator for {session_id}: {e}")
    finally:
        _debug(f"[SSE] Generator finished for {session_id}")
        # 清理资源
        remove_sse_queue(session_id)
