import sys

import numpy as np

from langchain.messages import HumanMessage, SystemMessage
from pymilvus import DataType
from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.llm.embedding_utils import generate_embeddings
from app.llm.llm_util import get_llm_client
from langchain_core.output_parsers import StrOutputParser

from app.utils.task_utils import add_done_task, add_running_task

# 大模型识别item_name的上下文切片数：取前5个切片，避免上下文过长导致大模型输入超限
DEFAULT_ITEM_NAME_CHUNK_K = 5
# 大模型上下文总字符数上限：适配主流大模型输入限制，默认2500
CONTEXT_TOTAL_MAX_CHARS = 10000

@step_log("step_1")
def step_1(state):
    chunks = state['chunks']
    file_title = state['file_title']

    if not chunks:
        logger.error(f"chunks没有内容,无法继续业务!")
        raise ValueError("chunks没有内容,无法继续业务!")
    
    if not file_title:
        logger.warning(f"file_title为空给与默认值处理!")
        file_title = "default_title"
    return chunks,file_title

@step_log("step_2")
def step_2(chunks)->str:
    '''
    获得某个文档的上下文
    文档的每个切块提供一点
    '''
    current_chunks = chunks[:DEFAULT_ITEM_NAME_CHUNK_K]
    chunk_str_list = []
    for index,item in enumerate(current_chunks,start=1):
        chunk_str_list.append(f"切片:{index},标题:{item['title']},内容:{item['content']}")
    chunk_str = "\n".join(chunk_str_list)
    final_chunk_str = chunk_str[:CONTEXT_TOTAL_MAX_CHARS]
    return final_chunk_str

@step_log("step_3")
def step_3(context,file_title)->str:
    '''
    调用大模型
    获得整个文档的item_name
    '''
    llm = get_llm_client()
    #处理提示词
    system_prompt = load_prompt("product_recognition_system")
    user_prompt =  load_prompt("item_name_recognition",file_title = file_title , context = context)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    #链式组装
    chains = llm | StrOutputParser()
    item_name = chains.invoke(messages)

    #兜底处理
    if not item_name:
         item_name = file_title

    return item_name  

@step_log("step_4")
def step_4(item_name, file_title, dense_vector, sparse_vector):
    '''
    存入milvus
    '''
    milvus_client = get_milvus_client()
    if not milvus_client:
        logger.error("无法连接到 Milvus 数据库，获取 client 失败！")
        raise ValueError("无法连接到 Milvus 数据库，获取 client 失败！")

    #判断集合是否已经被创建
    if not milvus_client.has_collection(milvus_config.item_name_collection):
        #创建表schema
        schema = milvus_client.create_schema(
            auto_id = True,#主键自增
            enable_dynamic_field = True,#可以传入没有申明的字段
        )

        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True,auto_id=True)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=512)
        #设置的GPU 半精度 当前嵌入模型维度为1024
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT16_VECTOR, dim = 1024)
        #dim不固定，有几个算几个
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        #创建索引
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_type="AUTOINDEX",  # 根据我们的数据量自动切换索引类型 [只支持稠密向量]
            index_name="dense_vector_index",
            metric_type = "IP", #向量没做归一化用COSINE，做了直接用内积IP
        )
        index_params.add_index(
                field_name="sparse_vector",
                index_type="SPARSE_INVERTED_INDEX",  # 只有一种索引
                index_name="sparse_vector_index",
                metric_type = "IP",
                params={"inverted_index_algo": "DAAT_MAXSCORE"}  #跳过0 比较有值的位置
            )

        #创建集合
        milvus_client.create_collection(
            collection_name=milvus_config.item_name_collection,
            schema=schema,
            index_params = index_params,
        )
    #删除旧数据
    milvus_client.delete(milvus_config.item_name_collection ,filter=f"item_name == '{item_name}'")
    #存储数据
    normalized_dense_vector = np.asarray(dense_vector, dtype=np.float16)

    data = [
            {
            "file_title":file_title,
            "item_name":item_name,
            "dense_vector":normalized_dense_vector,
            "sparse_vector":sparse_vector
            }
        ]
    milvus_client.insert(collection_name=milvus_config.item_name_collection , data=data)

@node_log("node_item_name_recognition")
def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 主体识别 (node_item_name_recognition)
    为什么叫这个名字: 识别文档核心的内容名称 (Item Name)。
    未来要实现:
    1. 取文档前几段内容。
    2. 调用 LLM 识别这篇文档讲的是什么东西。
    3. 存入 state["item_name"] 
    """ 
    add_running_task(state['task_id'],"node_item_name_recognition")
    
    chunks,file_title = step_1(state)
    #拼接上下文
    context = step_2(chunks)
    #获取item_name
    item_name = step_3(context,file_title)
    state["item_name"] = item_name

    for chunk in chunks:
        chunk['item_name'] = item_name
    state["chunks"] = chunks

    result = generate_embeddings([item_name])
    dense_vector = result['dense'][0]
    sparse_vector = result['sparse'][0]

    step_4(item_name,file_title,dense_vector,sparse_vector)

    add_done_task(state['task_id'],"node_item_name_recognition")
    return state
