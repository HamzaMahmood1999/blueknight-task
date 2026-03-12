from __future__ import annotations


class Reranker:
    """
    Placeholder re-ranker.

    TODO:
    - implement deterministic rerank over candidate set
    - expose score components for diagnostics
    """

    def rerank(self, candidates: list[dict], query: dict, top_k: int) -> list[dict]:
        raise NotImplementedError("Implement Reranker.rerank")

