import sys
import time
from app.utils.task_utils import add_running_task, add_done_task, set_task_result
from app.utils.sse_utils import push_to_session, SSEEvent
from app.query_process.agent.state import QueryGraphState
from app.core.logger import logger, step_log, node_log
from app.core.load_prompt import load_prompt
from app.llm.llm_util import get_llm_client
from app.clients.mongo_history_utils import save_chat_message
import re

@step_log("step_1_data_validates")
def step_1_data_validates(state):
    """
    验证输入数据的有效性
    """
    answer = state.get("answer")
    is_stream = state.get("is_stream", True)
    session_id = state.get("session_id")
    #完整答案就已经存在
    #说明：1.搜索关键词不确定 2.未找到关键词
    if answer:
        if is_stream:
            #模拟这个答案流式输出
            for ch in answer:
                push_to_session(session_id, SSEEvent.DELTA, {"delta": ch})
                time.sleep(0.3)
        set_task_result(session_id, "answer", answer)
        return True #返回 True：已有答案，本轮处理完毕，不用走后续流程
    return False #返回 False：无答案，需要继续走完整检索流程

@step_log("step_2_data_validates")
def step_2_data_validates(state):
    history = state.get("history", [])
    reranked_docs = state.get("reranked_docs", [])
    item_names = state.get("item_names", [])
    rewritten_query = state.get("rewritten_query", "")
    if not reranked_docs or len(reranked_docs) == 0 or not rewritten_query:
        logger.warning("reranked_docs为空或rewritten_query为空，无法继续处理。")
        raise ValueError("reranked_docs为空或rewritten_query为空，无法继续处理。")

    return history, reranked_docs, item_names, rewritten_query

@ step_log("step_3_make_prompt")
def step_3_make_prompt(state, history, reranked_docs, item_names, rewritten_query):
    context_chunk_list = []
    for number, chunk in enumerate(reranked_docs,start=1):
        context_chunk_list.append(
            f"第{number}块: 标题:{chunk['title']} 匹配度得分:{chunk['score']} 来源:{'网络搜索' if chunk['type'] == 'web' else '向量查询'}"
            f"\n"
            f"内容:{chunk['text']}"
        )
    context_chunk_str = "\n\n".join(context_chunk_list)

    history_text = "没有历史聊天记录!"
    if history and len(history) > 0:
        history_text = ""
        for msg in history:
            history_text += \
                (f"角色:{msg['role']},内容:{msg['rewritten_query'] if msg['role'] == 'user' else msg['text']}"
                 f",关联主体: {'、'.join(msg.get('item_names',[]))}\n")

    item_name = "本次关联主体:" + ",".join(item_names) if item_names and len(item_names) > 0 else '没有关联主体'
    # 加载提示词
    prompt = load_prompt("answer_out",context = context_chunk_str,
                history = history_text ,item_names = item_name,question = rewritten_query)

    return prompt

@step_log("step_4_generate_answer")
def step_4_generate_answer(state, prompt):
    """
    最终答案生成
    """
    is_stream = state.get("is_stream", True)
    session_id = state.get("session_id")
    llm_client = get_llm_client()
    final_answer = ""
    if is_stream:
        for chunk in llm_client.stream(prompt):
            delta_content = str(chunk.content)
            final_answer += delta_content
            push_to_session(session_id, SSEEvent.DELTA, {"delta": delta_content})
    else:
        response = llm_client.invoke(prompt)
        final_answer = str(response.content)
    set_task_result(session_id,"answer",final_answer)
    state['answer'] = final_answer

@step_log("step_5_extract_chunk_images")
def step_5_extract_chunk_images(state, reranked_docs):
    """
    提取切片中图片数据
    """
    image_urls = []
    #编译一个正则表达式，用来匹配 Markdown 格式的图片语法，例如：![描述](图片链接)
    reg = re.compile(r"\!\[.*?\]\((.*?)\)") 

    for chunk in reranked_docs:
        url = chunk.get("url")
        text = chunk.get("text", "")
        #web搜索字段里可能有图片url
        if url:
            if url.endswith((".png", ".jpg",".gif",".jpeg",".svg")):
                if url not in image_urls:
                    image_urls.append(url)

        if text:
            image_list = reg.findall(text)
            for image_url in image_list:
                if image_url not in image_urls:
                    image_urls.append(image_url)

    state['image_urls'] = image_urls

@step_log("step_6_save_chat_history")
def step_6_save_chat_history(state):
    """
    保存聊天记录
    """
    save_chat_message(
        session_id= state['session_id'],
        role="assistant",
        text=state.get("answer"),
        rewritten_query=state.get("rewritten_query") or state.get("original_query"),
        item_names=state.get("item_names",[]),
        image_urls = state.get("image_urls",[])
    )

@node_log("node_answer_output")
def node_answer_output(state):
    """
    节点功能：进行过处理可以是流式输出可以整体输出
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream", False))
    has_answer = step_1_data_validates(state)
    if not has_answer:
        history, reranked_docs, item_names, rewritten_query = step_2_data_validates(state)
        prompt = step_3_make_prompt(state, history, reranked_docs, item_names, rewritten_query)
        step_4_generate_answer(state, prompt)
        step_5_extract_chunk_images(state, reranked_docs)
    step_6_save_chat_history(state)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream", False))   
    return state
