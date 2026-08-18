from langgraph.graph import StateGraph, END
from loguru import logger

from app.query_process.agent.nodes.node_answer_output import node_answer_output
from app.query_process.agent.nodes.node_document_context import node_document_context
from app.query_process.agent.nodes.node_document_summary import (
    is_document_summary_request,
    node_document_summary,
)
from app.query_process.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.query_process.agent.nodes.node_rerank import node_rerank
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.query_process.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.query_process.agent.state import QueryGraphState

#定义状态图对象
query_graph = StateGraph(QueryGraphState)

#添加节点信息
query_graph.add_node("node_item_name_confirm", node_item_name_confirm)
query_graph.add_node("node_search_embedding", node_search_embedding)
query_graph.add_node("node_search_embedding_hyde", node_search_embedding_hyde)
query_graph.add_node("node_web_search_mcp", node_web_search_mcp)
query_graph.add_node("node_rrf", node_rrf)
query_graph.add_node("node_rerank", node_rerank)
query_graph.add_node("node_answer_output", node_answer_output)
query_graph.add_node("node_document_summary", node_document_summary)
query_graph.add_node("node_document_context", node_document_context)

#指定入口节点（条件边）
query_graph.set_entry_point("node_item_name_confirm")

def router(state:QueryGraphState):
    #不为空，说明这个节点已经判断出“后面没法继续检索了”，那就直接跳到 node_answer_output
    if state['answer']:
        logger.warning(f"{state['answer']}")
        return "node_answer_output"
    if is_document_summary_request(state):
        return "node_document_summary"
    if state.get("document_context_complete"):
        return "node_answer_output"
    else:
        #并发执行多路检索
        #1.向量数据库检索 2.HyDE(假设性文档嵌入)检索 3.mcp搜索
        routes = ["node_search_embedding", "node_search_embedding_hyde"]
        if state.get("enable_web_search", False):
            routes.append("node_web_search_mcp")
        return tuple(routes)

query_graph.add_edge("node_item_name_confirm", "node_document_context")

query_graph.add_conditional_edges(
    "node_document_context",
    router,
    {
        "node_search_embedding":"node_search_embedding",
        "node_search_embedding_hyde":"node_search_embedding_hyde",
        "node_web_search_mcp":"node_web_search_mcp",
        "node_answer_output":"node_answer_output",
        "node_document_summary":"node_document_summary",
    }
)

#静态边
query_graph.add_edge("node_search_embedding","node_rrf")
query_graph.add_edge("node_search_embedding_hyde","node_rrf")
query_graph.add_edge("node_web_search_mcp","node_rrf")
query_graph.add_edge("node_rrf","node_rerank")
query_graph.add_edge("node_rerank","node_answer_output")
query_graph.add_edge("node_document_summary", "node_answer_output")

#编译
query_app = query_graph.compile()
