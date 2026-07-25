from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


def _get_env_str(key: str, default: str = "") -> str:
    value = os.getenv(key)
    return value if value is not None else default


@dataclass
class EmbeddingConfig:
    bge_m3_path: str  # 本地模型路径
    bge_m3: str       # 模型仓库标识
    bge_device: str   # 运行设备(cuda:0/cpu)
    bge_fp16: bool    # 是否开启半精度（1=True/0=False）


embedding_config = EmbeddingConfig(
    bge_m3_path=_get_env_str("BGE_M3_PATH"),
    bge_m3=_get_env_str("BGE_M3"),
    bge_device=_get_env_str("BGE_DEVICE"),
    # 特殊处理：将.env中的1/0转为布尔值，兼容常见的数字/字符串格式
    bge_fp16=os.getenv("BGE_FP16") in ("1", "True", "true", 1)
)