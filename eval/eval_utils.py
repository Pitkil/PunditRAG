"""Shared helpers for reproducible end-to-end evaluations."""

from __future__ import annotations

import hashlib
import random
import statistics
from typing import Any, Iterable, Sequence, TypeVar


T = TypeVar("T")


def stable_case_id(value: Any) -> str:
    """Return a filesystem-safe, collision-resistant identifier."""
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def deterministic_sample(items: Sequence[T], size: int, seed: int) -> list[T]:
    """Sample without replacement while keeping a stable output order."""
    if size <= 0:
        return []
    if size >= len(items):
        return list(items)
    indexes = sorted(random.Random(seed).sample(range(len(items)), size))
    return [items[index] for index in indexes]


def percentile(values: Iterable[float], percentile_value: float) -> float | None:
    """Calculate a linearly interpolated percentile for small or large samples."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if not 0 <= percentile_value <= 100:
        raise ValueError("percentile_value must be between 0 and 100")
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percentile_value / 100
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction


def rate(matches: Iterable[bool]) -> float | None:
    values = list(matches)
    if not values:
        return None
    return round(sum(bool(value) for value in values) / len(values), 4)


def latency_metrics(values: Iterable[float]) -> dict[str, float | None]:
    latencies = [float(value) for value in values if value is not None]
    if not latencies:
        return {
            "latency_avg_s": None,
            "latency_p50_s": None,
            "latency_p95_s": None,
            "latency_p99_s": None,
        }
    return {
        "latency_avg_s": round(statistics.mean(latencies), 2),
        "latency_p50_s": round(percentile(latencies, 50), 2),
        "latency_p95_s": round(percentile(latencies, 95), 2),
        "latency_p99_s": round(percentile(latencies, 99), 2),
    }
