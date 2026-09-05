import json
import os
from urllib.parse import urlparse
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.exceptions import LangChainException
from typing import Optional

from pydantic import SecretStr
from app.conf.llm_config import llm_config
from app.core.logger import logger

llm_client_cache = {}


def _normalise_base_url(value: str) -> str:
    """Return a stable OpenAI-compatible base URL for the client and cache."""
    return (value or "").strip().rstrip("/")


def _is_dashscope_url(value: str) -> bool:
    host = (urlparse(value).hostname or "").lower()
    return host.endswith("dashscope.aliyuncs.com")


def _read_extra_body() -> dict:
    """Read optional provider-specific JSON without baking in a vendor."""
    raw = os.getenv("OPENAI_EXTRA_BODY_JSON", "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("OPENAI_EXTRA_BODY_JSON 必须是合法 JSON 对象") from exc
    if not isinstance(value, dict):
        raise ValueError("OPENAI_EXTRA_BODY_JSON 必须是 JSON 对象")
    return value

def get_llm_client(
    model: Optional[str] = None,
    json_mode: bool = False,
    timeout: Optional[float] = None,
    max_retries: int = 2,
) -> ChatOpenAI:
    '''
     获取带全局缓存的LangChain ChatOpenAI客户端实例

     model:模型名称

     json_model:是否开启JSON输出模式
    '''
    target_model = (model or llm_config.llm_model or "").strip()
    base_url = _normalise_base_url(llm_config.base_url)
    cache_key = (target_model, base_url, json_mode, timeout, max_retries)
    if cache_key in llm_client_cache:
        logger.debug(f"[LLM客户端] 缓存命中，直接返回实例：模型={target_model}，JSON模式={json_mode}")
        return llm_client_cache[cache_key]

    if not llm_config.api_key:
         raise ValueError("[LLM客户端] 配置缺失：请在.env中配置OPENAI_API_KEY（大模型API密钥）")
    if not base_url:
        raise ValueError("[LLM客户端] 配置缺失：请在.env中配置OPENAI_BASE_URL（API接口基础地址）")

    logger.info(f"[LLM客户端] 开始初始化新实例：模型={target_model}，JSON模式={json_mode}")

    #参数配置
    extra_params = _read_extra_body()
    # enable_thinking 是 DashScope 的扩展字段，不能仅凭模型名判断。
    # OrcaRouter/OpenRouter 等网关也可能代理 qwen 模型，但不一定接受该字段。
    if _is_dashscope_url(base_url):
        extra_params.setdefault("enable_thinking", False)
    model_kwargs = {}

    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}
        logger.debug(f"[LLM客户端] 已开启JSON输出模式，模型将返回标准JSON结构")

    try:
        llm_client = ChatOpenAI(
            model = target_model,
            temperature=llm_config.llm_temperature or 0.1,
            api_key = SecretStr(llm_config.api_key),
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            extra_body=extra_params or None,
            model_kwargs=model_kwargs,
        )
    except LangChainException as e:
        raise Exception(f"[LLM客户端] 模型【{target_model}】初始化失败（LangChain层）：{str(e)}") from e

    #新实例存入全局缓存，供后续复用
    llm_client_cache[cache_key] = llm_client
    logger.info(f"[LLM客户端] 实例初始化成功并缓存：模型={target_model}，JSON模式={json_mode}")

    return llm_client

