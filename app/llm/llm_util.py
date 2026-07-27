import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.exceptions import LangChainException
from typing import Optional

from pydantic import SecretStr
from app.conf.llm_config import llm_config
from app.core.logger import logger

llm_client_cache = {}

def get_llm_client(model: Optional[str] = None, json_mode: bool = False) -> ChatOpenAI:
    '''
     获取带全局缓存的LangChain ChatOpenAI客户端实例

     model:模型名称

     json_model:是否开启JSON输出模式
    '''
    target_model = model or llm_config.llm_model 
    cache_key = (target_model,json_mode)
    if cache_key in llm_client_cache:
        logger.debug(f"[LLM客户端] 缓存命中，直接返回实例：模型={target_model}，JSON模式={json_mode}")
        return llm_client_cache[cache_key]

    if not llm_config.api_key:
         raise ValueError("[LLM客户端] 配置缺失：请在.env中配置OPENAI_API_KEY（大模型API密钥）")
    if not llm_config.base_url:
        raise ValueError("[LLM客户端] 配置缺失：请在.env中配置OPENAI_API_BASE（API接口基础地址）")

    logger.info(f"[LLM客户端] 开始初始化新实例：模型={target_model}，JSON模式={json_mode}")

    #参数配置
    extra_params = {"enable_thinkiung":False} # 千问专属：关闭思考链输出，减少冗余内容
    model_kwarges = {}

    if json_mode:
        model_kwarges['response_format'] = {"type":"json_object"}
        logger.debug(f"[LLM客户端] 已开启JSON输出模式，模型将返回标准JSON结构")

    try:
        llm_client = ChatOpenAI(
            model = target_model,
            temperature=llm_config.llm_temperature or 0.1,
            api_key = SecretStr(llm_config.api_key),
            base_url= llm_config.base_url,
            extra_body=extra_params,
            model_kwargs=model_kwarges,
        )
    except LangChainException as e:
        raise Exception(f"[LLM客户端] 模型【{target_model}】初始化失败（LangChain层）：{str(e)}") from e

    #新实例存入全局缓存，供后续复用
    llm_client_cache[cache_key] = llm_client
    logger.info(f"[LLM客户端] 实例初始化成功并缓存：模型={target_model}，JSON模式={json_mode}")

    return llm_client

