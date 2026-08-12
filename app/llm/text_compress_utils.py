from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.llm.llm_util import get_llm_client

# 正文压缩目标长度（字符）：reranker 输入上限 512 token，query 预留空间，正文控制在 400 字内
COMPRESS_MAX_CHARS: int = 400


def compress_by_truncate(text: str, max_chars: int = COMPRESS_MAX_CHARS) -> str:
    """正文超长截断，保留开头（兜底方案，零成本）。"""
    return text if len(text) <= max_chars else text[:max_chars]


def compress_by_llm(text: str, max_chars: int = COMPRESS_MAX_CHARS) -> str:
    """正文超长时用 LLM 压缩，保留语义要点（prompts/compress.prompt）。"""
    if len(text) <= max_chars:
        return text

    prompt = load_prompt("compress", text=text, max_chars=max_chars)
    llm_client = get_llm_client()
    compressed = (llm_client | StrOutputParser()).invoke([HumanMessage(content=prompt)])
    # 兜底：压缩结果异常/超长时再截断一次，保证不超模型上限
    return compressed[:max_chars] if compressed else text[:max_chars]


def compress_text(text: str, max_chars: int = COMPRESS_MAX_CHARS) -> str:
    """LLM 压缩优先，异常时回退截断，保证链路不崩。"""
    try:
        return compress_by_llm(text, max_chars)
    except Exception as e:
        logger.warning(f"LLM压缩失败，回退截断：{str(e)}")
        return compress_by_truncate(text, max_chars)
