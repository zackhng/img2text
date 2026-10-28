"""Latency measurement for a callable model: warmup, then p50 / p95 / mean in milliseconds."""

from __future__ import annotations

import statistics
import time
from typing import Callable, Iterable


def measure_latency(fn: Callable[[object], object], inputs: Iterable[object],
                    warmup: int = 3) -> dict:
    items = list(inputs)
    if not items:
        raise ValueError("no inputs to time")
    for x in items[:warmup]:
        fn(x)
    times_ms = []
    for x in items:
        t0 = time.perf_counter()
        fn(x)
        times_ms.append((time.perf_counter() - t0) * 1000)
    times_ms.sort()
    p95_index = min(len(times_ms) - 1, int(round(0.95 * (len(times_ms) - 1))))
    return {
        "n": len(times_ms),
        "p50_ms": statistics.median(times_ms),
        "p95_ms": times_ms[p95_index],
        "mean_ms": statistics.fmean(times_ms),
    }
