import os
from dataclasses import dataclass


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    return max(minimum, int(os.getenv(name, str(default))))


def _float_env(name: str, default: float, minimum: float = 0.0) -> float:
    return max(minimum, float(os.getenv(name, str(default))))


@dataclass(frozen=True)
class RetrievalConfig:
    retrieval_top_k: int
    topic_expansion_top_k: int
    rrf_top_k: int
    rerank_max_top_k: int
    rerank_min_top_k: int
    rerank_fallback_top_k: int
    rerank_min_score: float
    rerank_gap_abs: float
    rerank_gap_ratio: float


retrieval_config = RetrievalConfig(
    retrieval_top_k=_int_env("RETRIEVAL_TOP_K", 20),
    topic_expansion_top_k=_int_env("TOPIC_EXPANSION_TOP_K", 10),
    rrf_top_k=_int_env("RRF_TOP_K", 30),
    rerank_max_top_k=_int_env("RERANK_MAX_TOP_K", 8),
    rerank_min_top_k=_int_env("RERANK_MIN_TOP_K", 2),
    rerank_fallback_top_k=_int_env("RERANK_FALLBACK_TOP_K", 3),
    rerank_min_score=_float_env("RERANK_MIN_SCORE", 0.09),
    rerank_gap_abs=_float_env("RERANK_GAP_ABS", 0.18),
    rerank_gap_ratio=_float_env("RERANK_GAP_RATIO", 0.35),
)
