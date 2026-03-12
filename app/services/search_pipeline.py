from __future__ import annotations

from app.schemas import SearchRequest, SearchResponse


class SearchPipeline:
    """
    Placeholder for Subtask 2.

    TODO(candidate):
    - vector recall
    - metadata post-filtering
    - reranking
    - diagnostics
    """

    async def run(self, request: SearchRequest) -> SearchResponse:
        raise NotImplementedError("Implement SearchPipeline.run")

