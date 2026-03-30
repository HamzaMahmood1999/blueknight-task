"""Structured logging utilities with trace_id propagation."""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from typing import Any, Generator


logger = logging.getLogger("blueknight")

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def structured_log(
    trace_id: str,
    stage: str,
    duration_ms: float,
    item_count: int | None = None,
    **extra: Any,
) -> None:
    """Emit a structured JSON log entry at a stage boundary."""
    entry: dict[str, Any] = {
        "trace_id": trace_id,
        "stage": stage,
        "duration_ms": round(duration_ms, 2),
    }
    if item_count is not None:
        entry["item_count"] = item_count
    entry.update(extra)
    logger.info(json.dumps(entry))


@contextmanager
def timed_stage(trace_id: str, stage: str) -> Generator[dict[str, Any], None, None]:
    """Context manager that times a stage and emits a structured log on exit.

    Usage:
        with timed_stage(trace_id, "vector_recall") as ctx:
            results = do_vector_search()
            ctx["item_count"] = len(results)
    """
    ctx: dict[str, Any] = {}
    start = time.perf_counter()
    try:
        yield ctx
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        ctx["duration_ms"] = elapsed_ms
        structured_log(
            trace_id=trace_id,
            stage=stage,
            duration_ms=elapsed_ms,
            **{k: v for k, v in ctx.items() if k != "duration_ms"},
        )
