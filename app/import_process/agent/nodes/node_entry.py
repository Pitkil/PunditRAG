from pathlib import Path
from app.core.logger import logger, node_log
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.task_utils import add_running_task, add_done_task

@node_log("node_entry")#一个自定义装饰器，用来给这个节点自动记录日志
def node_entry(state: ImportGraphState) -> ImportGraphState:
     """
     节点作用: 接收传入的文件地址(local_file_path)识别文件类型,修改对应的state
  入参:  local_file_path / task_id
  出参:  is_md_read_enabled is_pdf_read_enabled  md_path  pdf_path  file_title 
  步骤:
       0. 日志动作  @node_log + 任务列表记录 (进行中,已完成)
       1. 获取state中数据 local_file_path task_id
       2. 进行文件校验 local_file_path 是否为空
       3. 根据地址判断文件类型,修改对应的state参数即可
       4. 识别文件地址对应的文件名称
       5. 返回结果和状态 
    """
     #设置成进行中任务
     add_running_task(state['task_id'],"node_entry")

     local_file_path = state["local_file_path"]

     if not local_file_path:
          #文件为空，降级处理
          logger.warning(f"节点:node_entry,获取文件输入地址,发现地址为空!直接跳转到END节点")
          return state
     elif local_file_path.endswith(".md"):
          state["md_path"] = local_file_path
          state["is_md_read_enabled"] = True
     elif local_file_path.endswith(".pdf"):
          state["pdf_path"] = local_file_path
          state["is_pdf_read_enabled"] = True
     else:
          logger.warning(f"虽然local_file_path有值{local_file_path},但是无法识别类型，跳转到END节点")
          return state

     # 将字符串路径转换为 pathlib.Path 对象，方便进行路径与文件名操作
     local_file_path_obj = Path(local_file_path)
     # 获取剥离后缀后的纯文件名（例如 'C:/docs/report.pdf' -> 'report'）
     file_name = local_file_path_obj.stem 

     state["file_title"] = file_name

     #执行完成了
     add_done_task(state['task_id'],"node_entry")

     return state


     
          
