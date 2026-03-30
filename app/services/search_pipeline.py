"""Three-stage search pipeline: vector recall → post-filter → re-rank."""
from __future__ import annotations

import time
from typing import Any

from app import config
from app.retrieval import CompanyResult
from app.schemas import (
    Diagnostics,
    QueryPayload,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.services.reranker import Reranker
from app.services.retrieval_wrapper import retrieve_with_resilience
from app.utils.logging import timed_stage


class SearchPipeline:
    """Orchestrates the three-stage search pipeline with full diagnostics.

    Stage 1 — Vector Recall: Retrieve top_k_raw candidates via FAISS.
    Stage 2 — Post-Filter: Remove results based on geography mismatch,
              exclusion terms, and low vector scores.
    Stage 3 — Re-Rank: Composite scoring and final top_k_final selection.
    """

    def __init__(self) -> None:
        self._reranker = Reranker()

    async def run(self, request: SearchRequest) -> SearchResponse:
        """Execute the full search pipeline."""
        trace_id = request.trace_id
        query = request.query
        stage_latency_ms: dict[str, float] = {}
        drop_reasons: dict[str, int] = {}

        # ── Stage 1: Vector Recall ─────────────────────────────────
        with timed_stage(trace_id, "vector_recall") as ctx:
            candidates = await retrieve_with_resilience(
                query=query.query_text,
                top_k=request.top_k_raw,
                trace_id=trace_id,
            )
            ctx["item_count"] = len(candidates)

        raw_count = len(candidates)
        stage_latency_ms["vector_recall"] = round(ctx["duration_ms"], 1)

        # ── Stage 2: Post-Filter ───────────────────────────────────
        with timed_stage(trace_id, "post_filter") as ctx:
            filtered, drop_reasons = self._post_filter(candidates, query)
            ctx["item_count"] = len(filtered)
            ctx["dropped"] = raw_count - len(filtered)

        filtered_count = raw_count - len(filtered)
        stage_latency_ms["post_filter"] = round(ctx["duration_ms"], 1)

        # ── Stage 3: Re-Rank ──────────────────────────────────────
        with timed_stage(trace_id, "rerank") as ctx:
            results = self._reranker.rerank(
                candidates=filtered,
                query=query,
                top_k=request.top_k_final,
                offset=request.offset,
            )
            ctx["item_count"] = len(results)

        stage_latency_ms["rerank"] = round(ctx["duration_ms"], 1)

        # ── Build response ─────────────────────────────────────────
        diagnostics = Diagnostics(
            raw_count=raw_count,
            filtered_count=filtered_count,
            reranked_count=len(results),
            drop_reasons=drop_reasons,
            stage_latency_ms={k: int(v) for k, v in stage_latency_ms.items()},
            trace_id=trace_id,
        )

        return SearchResponse(
            results=results,
            total=len(results),
            diagnostics=diagnostics,
        )

    def _post_filter(
        self,
        candidates: list[CompanyResult],
        query: QueryPayload,
    ) -> tuple[list[CompanyResult], dict[str, int]]:
        """Filter candidates based on geography, exclusions, and score threshold.

        Returns:
            Tuple of (surviving candidates, drop_reasons counter).
        """
        drop_reasons: dict[str, int] = {}
        surviving: list[CompanyResult] = []

        # Prepare filter criteria
        geo_terms = [g.lower().strip() for g in query.geography] if query.geography else []
        exclusion_terms = [e.lower().strip() for e in query.exclusions] if query.exclusions else []

        for candidate in candidates:
            dropped = False
            offering_lower = candidate.long_offering.lower()
            country_lower = candidate.country.lower().strip()

            # Filter 1: Geography mismatch
            if geo_terms:
                geo_match = any(
                    geo in country_lower or geo in offering_lower
                    for geo in geo_terms
                )
                if not geo_match:
                    drop_reasons["geography_mismatch"] = (
                        drop_reasons.get("geography_mismatch", 0) + 1
                    )
                    dropped = True

            # Filter 2: Exclusion term present
            if not dropped and exclusion_terms:
                for term in exclusion_terms:
                    if term in offering_lower:
                        drop_reasons["exclude_term"] = (
                            drop_reasons.get("exclude_term", 0) + 1
                        )
                        dropped = True
                        break

            # Filter 3: Low vector score
            if not dropped and candidate.score < config.MIN_VECTOR_SCORE_THRESHOLD:
                drop_reasons["low_vector_score"] = (
                    drop_reasons.get("low_vector_score", 0) + 1
                )
                dropped = True

            if not dropped:
                surviving.append(candidate)

        return surviving, drop_reasons
