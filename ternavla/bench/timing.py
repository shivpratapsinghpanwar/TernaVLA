"""Latency statistics with the warm-up excluded (same method as the Humanoid census)."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable


@dataclass
class Timing:
    n: int
    warmup: int
    p50_ms: float
    p95_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


def _pct(sorted_ms: list[float], q: float) -> float:
    if len(sorted_ms) == 1:
        return sorted_ms[0]
    pos = q * (len(sorted_ms) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_ms) - 1)
    return sorted_ms[lo] + (sorted_ms[hi] - sorted_ms[lo]) * (pos - lo)


def measure(fn: Callable[[], object], n: int = 10, warmup: int = 2) -> Timing:
    for _ in range(warmup):
        fn()
    ms = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ms.append((time.perf_counter() - t0) * 1e3)
    s = sorted(ms)
    return Timing(n=n, warmup=warmup, p50_ms=round(_pct(s, 0.5), 3), p95_ms=round(_pct(s, 0.95), 3),
                  mean_ms=round(sum(ms) / len(ms), 3), min_ms=round(s[0], 3), max_ms=round(s[-1], 3))
