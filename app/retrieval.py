"""Vector retrieval with simulated operational characteristics.

Replaces mock_retrieve internals with real FAISS retrieval while keeping
the same function signature and simulating the documented behavior:
- ~200-300ms mean latency with occasional spikes
- ~5% transient failure rate (raises RetrievalError)
- Not safe to call with unbounded concurrency
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass

from app import config


class RetrievalError(RuntimeError):
    """Transient retrieval failure."""


@dataclass
class CompanyResult:
    id: str
    company_name: str
    country: str
    long_offering: str
    score: float


async def mock_retrieve(query: str, top_k: int) -> list[CompanyResult]:
    """Real vector retrieval with simulated operational characteristics.

    Behaviour (per spec — kept even with real FAISS to demonstrate wrapper handling):
    - ~200-300ms mean latency with occasional spikes
    - ~5% transient failure rate (raises RetrievalError)
    - Not safe to call with unbounded concurrency
    """
    from app.services.vector_store import get_vector_store

    # Simulate latency: 180-320ms base + occasional 200ms spike
    delay_ms = random.randint(
        config.RETRIEVAL_SIMULATED_LATENCY_MIN_MS,
        config.RETRIEVAL_SIMULATED_LATENCY_MAX_MS,
    ) + random.choice([0, 0, 0, 200])
    await asyncio.sleep(delay_ms / 1000.0)

    # Simulate transient failures (~5%)
    if random.random() < config.RETRIEVAL_SIMULATED_FAILURE_RATE:
        raise RetrievalError("Transient vector index error")

    # Real FAISS retrieval
    store = get_vector_store()
    results = await store.query(query_text=query, top_k=top_k)
    return results
