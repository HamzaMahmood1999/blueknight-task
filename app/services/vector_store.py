from __future__ import annotations

from typing import Any


class VectorStoreClient:
    """
    Placeholder abstraction for vector DB.

    Candidate can back this with Qdrant, Pinecone, FAISS, pgvector, etc.
    """

    async def upsert(self, items: list[dict[str, Any]]) -> None:
        raise NotImplementedError("Implement VectorStoreClient.upsert")

    async def query(
        self,
        embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError("Implement VectorStoreClient.query")

