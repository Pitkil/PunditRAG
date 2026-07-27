import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.state import create_default_state
from app.core.logger import logger

#1.创建state
state = create_default_state(task_id="001",local_file_path="xxx.pdf")

#2.执行编译后的图对象
result = kb_import_app.invoke(state)
logger.info(f"执行结果: {json.dumps(result)}")

#3.查看编译的图结构
print(kb_import_app.get_graph().print_ascii())

