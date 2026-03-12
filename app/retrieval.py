from __future__ import annotations

import random
import time
from dataclasses import dataclass


class RetrievalError(RuntimeError):
    """Transient retrieval failure."""


@dataclass
class CompanyResult:
    id: str
    company_name: str
    country: str
    long_offering: str
    score: float


def mock_retrieve(query: str, top_k: int) -> list[CompanyResult]:
    """
    Simulates vector search over pre-embedded long_offering values.

    Behaviour:
    - ~200-300ms mean latency with occasional spikes
    - ~5% transient failure rate (raises RetrievalError)
    - Not safe to call with unbounded concurrency

    TODO: Replace this with real retrieval backed by a vector DB
    populated from data/companies.csv. Keep the same function signature so
    the rest of the pipeline does not need to change.
    """
    delay_ms = random.randint(180, 320) + random.choice([0, 0, 0, 200])
    time.sleep(delay_ms / 1000.0)

    if random.random() < 0.05:
        raise RetrievalError("Transient vector index error")

    raise NotImplementedError(
        "Replace mock_retrieve with real vector retrieval over data/companies.csv"
    )
