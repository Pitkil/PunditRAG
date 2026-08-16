import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pymongo import MongoClient, ASCENDING
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()

history_mongo_tool = None

class HistoryMongoTool:
    def __init__(self):
        try:
            self.mongo_url = os.getenv("MONGO_URL")
            self.db_name = os.getenv("MONGO_DB_NAME")
            if not self.db_name:
                raise ValueError("MONGO_DB_NAME is not set")
            self.client = MongoClient(self.mongo_url)
            self.db = self.client[self.db_name]
            #获取对话记录
            self.chat_message = self.db["chat_message"]
            #创建索引
            self.chat_message.create_index([("session_id", 1), ("ts", -1)])

            logging.info("成功连接到MongoDB数据库: %s", self.db_name)
        except Exception as e:
            logging.error("连接到MongoDB数据库失败: %s", str(e))
            raise

def get_history_mongo_tool() -> HistoryMongoTool:
    """
    获取HistoryMongoTool的单例实例
    :return: HistoryMongoTool实例
    """
    global history_mongo_tool
    if history_mongo_tool is None:
        history_mongo_tool = HistoryMongoTool()
    return history_mongo_tool

def save_chat_message(
    session_id:str,
    role:str,
    text:str,
    rewritten_query:str="",
    item_names:Optional[List[str]] = None,
    image_urls:Optional[List[str]] = None,
    sources:Optional[List[Dict[str, Any]]] = None,
    message_id:Optional[str] = None,
)->str:
    ts = datetime.now().timestamp()

    document = {
        "session_id": session_id,
        "role": role,
        "text": text,
        "rewritten_query": rewritten_query,
        "item_names": item_names,
        "image_urls": image_urls,
        "sources": sources or [],
        "ts": ts,
    }

    mongo_tool = get_history_mongo_tool()

    if message_id:
        result = mongo_tool.chat_message.update_one(
            {"_id": ObjectId(message_id)},#，每一条存入的数据（称为 Document）都必须有一个唯一的主键（Primary Key）。MongoDB 官方在设计时，把这个系统保留的主键字段强制命名为 _id。
            {"$set": document}#更新指定字段
        )
        return message_id
    else:
        result = mongo_tool.chat_message.insert_one(document)
        return str(result.inserted_id)

def clear_history(session_id:str)->int:
    mongo_tool = get_history_mongo_tool()
    try:
        result = mongo_tool.chat_message.delete_many({"session_id": session_id})
        logging.info("已删除 %d 条会话 %s 的历史记录", result.deleted_count, session_id)
        return result.deleted_count
    except Exception as e:
        logging.error("清空会话 %s 历史记录时出错: %s", session_id, str(e))
        return 0

def update_message_item_names(ids: List[str], item_names: List[str]) -> int:
     """
    批量更新历史会话记录的关键词名称
    仅更新满足条件的记录：主键在指定列表中，且item_names为空/不存在/None
    :param ids: 要更新的记录主键ID列表（字符串类型）
    :param item_names: 要设置的新关键词列表
    :return: 实际更新的文档数量，更新失败返回0
    """
     mongo_tool = get_history_mongo_tool()
     try:
        # 将字符串类型的主键列表转为MongoDB的ObjectId类型（数据库中主键是ObjectId类型）
        object_ids = [ObjectId(id_str) for id_str in ids]

        # 执行批量更新操作，使用$in匹配主键，$or匹配item_names为空/不存在/None的记录
        result = mongo_tool.chat_message.update_many(
            {
                "_id": {"$in": object_ids},  # 主键在指定的ID列表中（批量筛选）
                "$or": [  # 满足以下任一条件
                    {"item_names": {"$exists": False}},  # item_names字段不存在
                    {"item_names": []},  # item_names是空列表
                    {"item_names": None}  # item_names是None
                ]
            },
            {"$set": {"item_names": item_names}}  # 更新操作：设置新的关键词列表
        )

        logging.info("已更新 %d 条记录的关键词名称为: %s", result.modified_count, item_names)
        return result.modified_count
     except Exception as e:
        logging.error("批量更新历史会话记录的关键词名称时出错: %s", str(e))
        return 0

def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    获取指定会话的最近聊天记录
    :param session_id: 会话ID
    :param limit: 返回的记录条数，默认10条
    :return: 最近聊天记录列表，按时间倒序排列
    """
    mongo_tool = get_history_mongo_tool()
    try:
        #查询的是当前会话
        query = {"session_id":session_id}
        cursor = mongo_tool.chat_message.find(query).sort("ts", -1).limit(limit)
        messages = list(reversed(list(cursor)))
        return messages
    except Exception as e:
        logging.error(f"Error getting recent messages: {e}")
        return []
