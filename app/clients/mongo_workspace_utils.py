import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.clients.mongo_history_utils import get_history_mongo_tool


def _now() -> float:
    return datetime.now().timestamp()


def _db():
    tool = get_history_mongo_tool()
    tool.db["knowledge_base"].create_index("kb_id", unique=True)
    tool.db["knowledge_document"].create_index("document_id", unique=True)
    tool.db["knowledge_document"].create_index([("kb_id", 1), ("created_at", -1)])
    tool.db["chat_session"].create_index("session_id", unique=True)
    return tool.db


def serialize_document(document: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not document:
        return {}
    result = dict(document)
    result.pop("_id", None)
    return result


def list_knowledge_bases() -> List[Dict[str, Any]]:
    db = _db()
    result = []
    for kb in db.knowledge_base.find().sort("created_at", 1):
        item = serialize_document(kb)
        item["document_count"] = db.knowledge_document.count_documents(
            {"kb_id": item["kb_id"], "status": {"$ne": "deleted"}}
        )
        result.append(item)
    return result


def create_knowledge_base(name: str, description: str = "") -> Dict[str, Any]:
    kb = {
        "kb_id": uuid.uuid4().hex,
        "name": name.strip(),
        "description": description.strip(),
        "created_at": _now(),
        "updated_at": _now(),
    }
    _db().knowledge_base.insert_one(kb)
    return serialize_document(kb)


def update_knowledge_base(kb_id: str, values: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {key: value.strip() for key, value in values.items() if key in {"name", "description"} and isinstance(value, str)}
    if allowed:
        allowed["updated_at"] = _now()
        _db().knowledge_base.update_one({"kb_id": kb_id}, {"$set": allowed})
    return serialize_document(_db().knowledge_base.find_one({"kb_id": kb_id}))


def delete_knowledge_base_record(kb_id: str) -> int:
    db = _db()
    db.knowledge_document.delete_many({"kb_id": kb_id})
    return db.knowledge_base.delete_one({"kb_id": kb_id}).deleted_count


def create_document(kb_id: str, filename: str, local_path: str, task_id: str) -> Dict[str, Any]:
    document = {
        "document_id": uuid.uuid4().hex,
        "kb_id": kb_id,
        "filename": filename,
        "file_title": filename.rsplit(".", 1)[0],
        "local_path": local_path,
        "task_id": task_id,
        "status": "pending",
        "chunk_count": 0,
        "error": "",
        "created_at": _now(),
        "updated_at": _now(),
    }
    _db().knowledge_document.insert_one(document)
    return serialize_document(document)


def update_document(document_id: str, **values: Any) -> Dict[str, Any]:
    values["updated_at"] = _now()
    db = _db()
    db.knowledge_document.update_one({"document_id": document_id}, {"$set": values})
    return serialize_document(db.knowledge_document.find_one({"document_id": document_id}))


def list_documents(kb_id: Optional[str] = None) -> List[Dict[str, Any]]:
    query: Dict[str, Any] = {"status": {"$nin": ["deleting", "deleted"]}}
    if kb_id:
        query["kb_id"] = kb_id
    return [serialize_document(item) for item in _db().knowledge_document.find(query).sort("created_at", -1)]


def get_document(document_id: str) -> Dict[str, Any]:
    return serialize_document(_db().knowledge_document.find_one({"document_id": document_id}))


def ensure_document_active(document_id: str) -> Dict[str, Any]:
    """阻止已经删除的文档被后台导入任务重新写回。"""
    document = get_document(document_id)
    if not document or document.get("status") in {"deleting", "deleted"}:
        raise RuntimeError("文档已删除，导入任务已取消")
    return document


def delete_document_record(document_id: str) -> int:
    return _db().knowledge_document.delete_one({"document_id": document_id}).deleted_count


def ensure_chat_session(session_id: str, title: str = "新对话") -> Dict[str, Any]:
    db = _db()
    now = _now()
    db.chat_session.update_one(
        {"session_id": session_id},
        {
            "$setOnInsert": {"session_id": session_id, "title": title[:40] or "新对话", "created_at": now},
            "$set": {"updated_at": now},
        },
        upsert=True,
    )
    if title != "新对话":
        db.chat_session.update_one(
            {"session_id": session_id, "title": "新对话"},
            {"$set": {"title": title[:40], "updated_at": now}},
        )
    return serialize_document(db.chat_session.find_one({"session_id": session_id}))


def list_chat_sessions(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    cursor = (
        _db().chat_session.find()
        .sort("updated_at", -1)
        .skip(max(0, int(offset)))
        .limit(max(1, min(int(limit), 100)))
    )
    return [serialize_document(item) for item in cursor]


def count_chat_sessions() -> int:
    return _db().chat_session.count_documents({})


def get_chat_session(session_id: str) -> Dict[str, Any]:
    return serialize_document(_db().chat_session.find_one({"session_id": session_id}))


def rename_chat_session(session_id: str, title: str) -> Dict[str, Any]:
    _db().chat_session.update_one(
        {"session_id": session_id},
        {"$set": {"title": title.strip()[:80], "updated_at": _now()}},
        upsert=False,
    )
    return serialize_document(_db().chat_session.find_one({"session_id": session_id}))


def delete_chat_session(session_id: str) -> int:
    db = _db()
    db.chat_message.delete_many({"session_id": session_id})
    return db.chat_session.delete_one({"session_id": session_id}).deleted_count
