from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class RerankerConfig:
    bge_reranker_path: str
    bge_reranker_model_id: str
    bge_reranker_device: str
    bge_reranker_fp16: bool

# 实例化配置对象，和原代码lm_config风格保持一致
reranker_config = RerankerConfig(
    bge_reranker_path=os.getenv("BGE_RERANKER_PATH") or "",
    bge_reranker_model_id=os.getenv("BGE_RERANKER_MODEL_ID") or "BAAI/bge-reranker-v2-m3",
    bge_reranker_device=os.getenv("BGE_RERANKER_DEVICE") or "cpu",
    bge_reranker_fp16=os.getenv("BGE_RERANKER_FP16") in ("1", "True", "true", 1),
)
