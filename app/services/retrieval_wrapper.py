"""Retrieval wrapper with retry, timeout, and concurrency limiting.

This is the ONLY module that calls mock_retrieve. All business logic
must go through retrieve_with_resilience().
"""
from __future__ import annotations

import asyncio
import logging
import time

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app import config
from app.retrieval import CompanyResult, RetrievalError
from app.utils.logging import structured_log

logger = logging.getLogger("blueknight")

# Module-level semaphore to prevent unbounded concurrency
_semaphore = asyncio.Semaphore(config.RETRIEVAL_CONCURRENCY_LIMIT)


@retry(
    retry=retry_if_exception_type(RetrievalError),
    stop=stop_after_attempt(config.RETRIEVAL_MAX_RETRIES),
    wait=wait_exponential(
        multiplier=config.RETRIEVAL_BACKOFF_BASE,
        min=0.5,
        max=4.0,
    ),
    reraise=True,
)
async def _retrieve_with_retry(query: str, top_k: int) -> list[CompanyResult]:
    """Inner retry wrapper — retries on transient RetrievalError."""
    from app.retrieval import mock_retrieve

    return await asyncio.wait_for(
        mock_retrieve(query, top_k),
        timeout=config.RETRIEVAL_TIMEOUT_SECONDS,
    )


async def retrieve_with_resilience(
    query: str,
    top_k: int,
    trace_id: str = "",
) -> list[CompanyResult]:
    """Production-grade retrieval: semaphore + retry + timeout + logging.

    Args:
        query: The search query text.
        top_k: Max number of results to retrieve.
        trace_id: Trace ID for structured logging.

    Returns:
        List of CompanyResult from the vector store.

    Raises:
        RetrievalError: If all retry attempts fail.
        asyncio.TimeoutError: If request exceeds timeout.
    """
    start = time.perf_counter()
    attempt_count = 0

    async with _semaphore:
        try:
            results = await _retrieve_with_retry(query, top_k)
            elapsed_ms = (time.perf_counter() - start) * 1000

            structured_log(
                trace_id=trace_id,
                stage="retrieval_wrapper",
                duration_ms=elapsed_ms,
                item_count=len(results),
                status="success",
                query_preview=query[:80],
            )
            return results

        except (RetrievalError, asyncio.TimeoutError) as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            structured_log(
                trace_id=trace_id,
                stage="retrieval_wrapper",
                duration_ms=elapsed_ms,
                status="failed",
                error=str(exc),
                query_preview=query[:80],
            )
            raise
