from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
#自动为仅用来保存数据的类生成常用的模板代码
# （如初始化函数 __init__、字符串打印 __repr__、比较函数 __eq__ 等）。
class LLMConfig:
    base_url: str
    api_key : str
    vl_model: str
    llm_model: str
    llm_temperature: float

llm_config = LLMConfig(
    base_url=os.getenv("OPENAI_BASE_URL", ""),
    api_key=os.getenv("OPENAI_API_KEY", ""),
    vl_model=os.getenv("VL_MODEL", ""),
    llm_model=os.getenv("LLM_DEFAULT_MODEL", ""),
    llm_temperature=float(os.getenv("LLM_DEFAULT_TEMPERATURE", 0.1))
)