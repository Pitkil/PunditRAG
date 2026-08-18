from typing_extensions import TypedDict
from typing import List
import copy

class QueryGraphState(TypedDict):
    """
    QueryGraphState 定义了整个查询流程中流转的数据结构。
    """
    session_id: str  # 会话唯一标识
    run_id: str  # 单次查询运行标识，用于隔离任务状态和 SSE
    original_query: str  # 用户原始问题
    scope_mode: str  # all / knowledge_base / documents
    kb_ids: List[str]  # 本轮允许检索的知识库
    document_ids: List[str]  # 本轮明确绑定的文档
    scope_document_names: List[str]  # 当前显式范围内的文档名，仅用于辅助召回
    enable_web_search: bool  # 是否启用联网搜索

    # 检索过程中的中间数据
    embedding_chunks: list  # 普通向量检索回来的切片
    hyde_embedding_chunks: list  # HyDE 检索回来的切片
    web_search_docs: list  # 网络搜索回来的文档

    # 排序过程中的数据
    rrf_chunks: list  # RRF 融合排序后的切片
    reranked_docs: list  # 重排序后的最终 Top-K 文档
    evidence_quality: str  # qualified / low / unscored / none

    # 生成过程中的数据
    prompt: str  # 组装好的 Prompt
    answer: str  # 最终生成的答案
    answer_streamed: bool  # 已在上游节点发送过答案增量，避免重复推送
    answer_basis: str  # sources / general / refused
    image_urls: list  # 最终答案关联的图片地址
    sources: list  # 最终答案引用来源
    user_message_id: str  # 本轮用户消息的 MongoDB ID
    assistant_message_id: str  # 本轮助手消息的 MongoDB ID

    # 辅助信息
    item_names: List[str]  # 提取出的商品名称
    rewritten_query: str  # 兼容字段，始终保存用户原问题
    full_document: bool  # 是否读取当前显式范围的全部正文
    document_context_complete: bool  # 显式文档完整正文是否已装入回答上下文
    history: list  # 历史对话记录
    is_stream: bool  # 是否流式输出标记


# 默认状态（全部为空）
query_graph_default_state: QueryGraphState = {
    "session_id": "",
    "run_id": "",
    "original_query": "",
    "scope_mode": "knowledge_base",
    "kb_ids": [],
    "document_ids": [],
    "scope_document_names": [],
    "enable_web_search": True,
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "web_search_docs": [],
    "rrf_chunks": [],
    "reranked_docs": [],
    "evidence_quality": "none",
    "prompt": "",
    "answer": "",
    "answer_streamed": False,
    "answer_basis": "",
    "image_urls": [],
    "sources": [],
    "user_message_id": "",
    "assistant_message_id": "",
    "item_names": [],
    "rewritten_query": "",
    "full_document": False,
    "document_context_complete": False,
    "history": [],
    "is_stream": False
}


def create_query_default_state(**overrides) -> QueryGraphState:
    """
    创建查询流程的默认状态，支持覆盖字段
    """
    state = copy.deepcopy(query_graph_default_state)
    state.update(**overrides)
    return state


# 获取干净状态
def get_query_default_state() -> QueryGraphState:
    return copy.deepcopy(query_graph_default_state)


def copy_query_state(state: QueryGraphState, **overrides) -> QueryGraphState:
    """
    复制现有状态并可覆盖字段，深拷贝，不污染原数据
    """
    new_state = copy.deepcopy(state)
    new_state.update(**overrides)
    return new_state
