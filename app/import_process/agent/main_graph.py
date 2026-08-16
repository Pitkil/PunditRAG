from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.import_process.agent.nodes.node_entry import node_entry  
from app.import_process.agent.nodes.node_pdf_to_md import node_pdf_to_md 
from app.import_process.agent.nodes.node_file_to_md import node_file_to_md
from app.import_process.agent.nodes.node_md_img import node_md_img  
from app.import_process.agent.nodes.node_document_split import node_document_split  
from app.import_process.agent.nodes.node_item_name_recognition import node_item_name_recognition  
from app.import_process.agent.nodes.node_bge_embedding import node_bge_embedding  
from app.import_process.agent.nodes.node_import_milvus import node_import_milvus 

load_dotenv()

#1.定义状态图对象
workflow = StateGraph(ImportGraphState)

#2.添加节点
workflow.add_node("node_entry", node_entry)
workflow.add_node("node_pdf_to_md", node_pdf_to_md)
workflow.add_node("node_file_to_md", node_file_to_md)
workflow.add_node("node_md_img", node_md_img)
workflow.add_node("node_document_split", node_document_split)
workflow.add_node("node_item_name_recognition", node_item_name_recognition)
workflow.add_node("node_bge_embedding", node_bge_embedding)
workflow.add_node("node_import_milvus", node_import_milvus)

#3.指定入口节点
workflow.set_entry_point("node_entry")

#3.设置入口节点后的条件边
def after_entry_node(state: ImportGraphState):
    if state['is_md_read_enabled']:
        return "node_md_img"
    elif state['is_pdf_read_enabled']:
        return "node_pdf_to_md"
    elif state['is_file_convert_enabled']:
        return "node_file_to_md"
    else:
        return END
'''
添加条件判断边
'''
workflow.add_conditional_edges(
    "node_entry",
    after_entry_node,
    {
        "node_md_img": "node_md_img",
        "node_pdf_to_md": "node_pdf_to_md",
        "node_file_to_md": "node_file_to_md",
        END: END,
    },
)

#4.设置静态条件边
workflow.add_edge("node_md_img","node_document_split")
workflow.add_edge("node_pdf_to_md","node_document_split")
workflow.add_edge("node_file_to_md","node_md_img")
workflow.add_edge("node_document_split","node_item_name_recognition")
workflow.add_edge("node_item_name_recognition","node_bge_embedding")
workflow.add_edge("node_bge_embedding","node_import_milvus")
workflow.add_edge("node_import_milvus",END)

#5.编译图对象(Knowledge Base->知识库)
kb_import_app = workflow.compile()


