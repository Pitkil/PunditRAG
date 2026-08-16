import json
import sys
from pymilvus import DataType
from app.clients.milvus_utils import get_milvus_client
from app.clients.mongo_workspace_utils import ensure_document_active
from app.conf.milvus_config import milvus_config
from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_done_task, add_running_task

@step_log("step_1")
def step_1(state):
    chunks = state['chunks']
    if not chunks or len(chunks) == 0:
        logger.error("chunks为空，无法导入向量库！")
        raise ValueError(f"chunks为空,无法继续业务!!")
    return chunks

@step_log("step_2")
def step_2():
    milvus_client = get_milvus_client() 
    if not milvus_client:
            logger.error("无法连接到 Milvus 数据库，获取 client 失败！")
            raise ValueError("无法连接到 Milvus 数据库，获取 client 失败！")
    
    collection_name = milvus_config.chunks_collection
    if milvus_client.has_collection(collection_name):
         collection = milvus_client.describe_collection(collection_name)
         part_field = next((field for field in collection.get("fields", []) if field.get("name") == "part"), None)
         if part_field and part_field.get("type") != DataType.INT32:
              row_count = int(milvus_client.get_collection_stats(collection_name).get("row_count", 0))
              if row_count:
                   raise RuntimeError(
                        f"Milvus集合{collection_name}的part字段仍为旧类型，且包含{row_count}条数据，"
                        "请先迁移或清空该集合后重试"
                   )
              logger.warning(f"Milvus集合{collection_name}使用旧版part字段，正在重建空集合")
              milvus_client.drop_collection(collection_name)

    if not milvus_client.has_collection(collection_name):
         #创建schema
         schema = milvus_client.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
         )
         #添加字段
         schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
         schema.add_field(field_name="file_title", datatype=DataType.VARCHAR,max_length=512)
         schema.add_field(field_name="item_name", datatype=DataType.VARCHAR,max_length=512)
         schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR,max_length=512)
         # 切片序号可能超过127，不能使用INT8。
         schema.add_field(field_name="part", datatype=DataType.INT32)
         schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
         schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
         schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
         #索引
         index_params= milvus_client.prepare_index_params()
         index_params.add_index(
            field_name="dense_vector",
            index_type="AUTOINDEX", 
            index_name="dense_vector_index",
            metric_type="IP"
        )
         index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",  
            index_name="sparse_vector_index",
            metric_type="IP",  
            params={"inverted_index_algo": "DAAT_MAXSCORE"}
        )
         #创建collection
         milvus_client.create_collection(
              collection_name=collection_name,
              schema = schema,
              index_params=index_params
         )

@step_log("step_3")
def step_3(state):
     '''
     删除旧数据
     同一个文档按 file_title 幂等更新，避免同主题的不同文档互相覆盖
     '''
     milvus_client = get_milvus_client()
     if not milvus_client:
                 logger.error("无法连接到 Milvus 数据库，获取 client 失败！")
                 raise ValueError("无法连接到 Milvus 数据库，获取 client 失败！")
     file_title = state['file_title']
     document_id = state.get("document_id", "")
     delete_filter = (
         f"document_id == {json.dumps(document_id, ensure_ascii=False)}"
         if document_id
         else f"file_title == {json.dumps(file_title, ensure_ascii=False)}"
     )
     milvus_client.delete(milvus_config.chunks_collection, filter=delete_filter)

@step_log("step_4")
def step_4(chunks, document_id=""):
      '''
      插入数据
      '''
      if document_id:
          ensure_document_active(document_id)
      milvus_client = get_milvus_client()
      if not milvus_client:
                       logger.error("无法连接到 Milvus 数据库，获取 client 失败！")
                       raise ValueError("无法连接到 Milvus 数据库，获取 client 失败！")
      result = milvus_client.insert(
        collection_name=milvus_config.chunks_collection,
        data=chunks
    )
      insert_count = result.get("insert_count",0)
      logger.info(f"插入数据成功! 总条数:{insert_count}")
      if document_id:
          try:
              ensure_document_active(document_id)
          except RuntimeError:
              milvus_client.delete(
                  milvus_config.chunks_collection,
                  filter=f"document_id == {json.dumps(document_id, ensure_ascii=False)}",
              )
              raise
      
@node_log("node_import_milvus")
def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    实现:
    1. 连接 Milvus。
    2. 根据 file_title 删除同一文档的旧数据。
    3. 批量插入新的向量数据。
    """ 
    add_running_task(state['task_id'],"node_import_milvus")
    chunks = step_1(state)
    step_2()
    step_3(state)
    step_4(chunks, state.get("document_id", ""))
    add_done_task(state['task_id'],"node_import_milvus")
    return state
