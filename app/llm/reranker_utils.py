from pathlib import Path
from threading import Lock

from FlagEmbedding import FlagReranker
from app.conf.reranker_config import reranker_config
from app.core.logger import logger

_reranker_model = None
_reranker_lock = Lock()


def _has_model_weights(model_dir: Path) -> bool:
    return model_dir.is_dir() and any(
        (model_dir / filename).is_file()
        for filename in ("model.safetensors", "pytorch_model.bin")
    )

def get_reranker_model():
    global _reranker_model
    if _reranker_model is None:
        with _reranker_lock:
            if _reranker_model is None:
                local_path = reranker_config.bge_reranker_path.strip()
                model_name = local_path if local_path and _has_model_weights(Path(local_path)) else reranker_config.bge_reranker_model_id
                if local_path and model_name != local_path:
                    logger.warning(f"Reranker本地目录为空或不存在：{local_path}，改用模型仓库：{model_name}")
                _reranker_model= FlagReranker(
                    model_name_or_path=model_name,
                    device=reranker_config.bge_reranker_device,
                    use_fp16=reranker_config.bge_reranker_fp16,
                )
    return _reranker_model
