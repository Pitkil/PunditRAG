import sys
from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.llm.embedding_utils import generate_embeddings
from app.utils.task_utils import add_done_task, add_running_task

@step_log("step_1")
def step_1(state):
    chunks = state['chunks']
    if not chunks:
        logger.error("chunks为komg,无法继续！")
        raise ValueError("chunks为komg,无法继续！")
    return chunks

@step_log("step_2")
def step_2(chunks):
    '''
    给chunks生成向量（稠密/稀疏）
    '''
    chunks_vector = []
    total = len(chunks)
    step = 5
    for index in range(0,total,step):
        #步长为5，0->5->10....
        try:
            step_chunks = chunks[index:index+step]
            vector_str_list = []
            for item in step_chunks:
                metadata = [
                    f"文档：{item.get('file_title', '')}",
                    f"章节：{item.get('parent_title') or item.get('title', '')}",
                ]
                if item.get("item_name"):
                    metadata.append(f"主题：{item['item_name']}")
                metadata.append(f"内容：{item['content']}")
                text = "\n".join(metadata)
                vector_str_list.append(text)
                #最多5个一次，防止本地模型崩溃
            vectors = generate_embeddings(vector_str_list)
            for i,chunk in enumerate(step_chunks,start=0):
                chunk_new = chunk.copy()
                chunk_new['dense_vector'] = vectors['dense'][i]
                chunk_new['sparse_vector'] = vectors['sparse'][i]
                chunks_vector.append(chunk_new)
        except Exception as e:
            logger.exception(f"index = {index} 批次向量生成失败，终止导入，避免部分数据入库")
            raise RuntimeError(f"第 {index // step + 1} 批切片向量生成失败") from e

    return chunks_vector 

@node_log("node_bge_embedding")
def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    实现:
    1. 加载 BGE-M3 模型。
    2. 对每个 Chunk 的文本进行 Dense (稠密) 和 Sparse (稀疏) 向量化。
    3. 准备好写入 Milvus 的数据格式。
    """
    add_running_task(state['task_id'],"node_bge_embedding")
    chunks = step_1(state)
    chunks_vector = step_2(chunks)
    state['chunks'] = chunks_vector
    add_done_task(state['task_id'],"node_bge_embedding")
    return state
