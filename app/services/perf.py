from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def timer() -> float:
    """
    Minimal timing helper.

    TODO(candidate):
    - integrate with diagnostics struct
    - aggregate p50/p95 in benchmark script
    """
    start = time.perf_counter()
    try:
        yield start
    finally:
        _ = (time.perf_counter() - start) * 1000

