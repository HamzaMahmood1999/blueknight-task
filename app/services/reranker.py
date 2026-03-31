"""Composite re-ranker with transparent score components."""
from __future__ import annotations

import re
from typing import Any

from app import config
from app.retrieval import CompanyResult
from app.schemas import QueryPayload, SearchResult


class Reranker:
    """Re-ranks candidate results using a composite scoring approach.

    Scoring logic:
        final_score = W_VECTOR * vector_score
                    + W_GEO    * geo_boost
                    + W_KEYWORD * keyword_boost

    - vector_score: Cosine similarity from FAISS (already in [0, 1] after normalization)
    - geo_boost:    1.0 if company country matches any query geography, else 0.0
    - keyword_boost: Fraction of query keywords found in long_offering text (0.0 to 1.0)

    All weights are configurable via app.config.
    """

    def rerank(
        self,
        candidates: list[CompanyResult],
        query: QueryPayload,
        top_k: int,
        offset: int = 0,
    ) -> list[SearchResult]:
        """Score, sort, and slice candidates into final SearchResult list."""
        # Extract meaningful keywords from query (3+ char words, lowercased)
        query_keywords = self._extract_keywords(query.query_text)

        # Normalize geography terms for matching
        geo_terms = [g.lower().strip() for g in query.geography]

        scored: list[tuple[float, dict[str, float], CompanyResult]] = []

        for candidate in candidates:
            offering_lower = candidate.long_offering.lower()
            country_lower = candidate.country.lower().strip()

            # Component 1: vector similarity score
            vector_score = max(0.0, min(1.0, candidate.score))

            # Component 2: geography boost
            geo_boost = 0.0
            if geo_terms:
                for geo in geo_terms:
                    if re.search(rf"\b{re.escape(geo)}\b", country_lower):
                        geo_boost = 1.0
                        break

            # Component 3: keyword relevance boost
            keyword_boost = 0.0
            if query_keywords:
                matches = sum(1 for kw in query_keywords if kw in offering_lower)
                keyword_boost = matches / len(query_keywords)

            # Composite score
            final_score = (
                config.RERANK_WEIGHT_VECTOR * vector_score
                + config.RERANK_WEIGHT_GEO_BOOST * geo_boost
                + config.RERANK_WEIGHT_KEYWORD_BOOST * keyword_boost
            )

            score_components = {
                "vector": round(vector_score, 4),
                "geo_boost": round(geo_boost * config.RERANK_WEIGHT_GEO_BOOST, 4),
                "keyword_boost": round(
                    keyword_boost * config.RERANK_WEIGHT_KEYWORD_BOOST, 4
                ),
            }

            scored.append((final_score, score_components, candidate))

        # Sort by final score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Apply offset and top_k slicing
        sliced = scored[offset : offset + top_k]

        # Convert to SearchResult
        results: list[SearchResult] = []
        for final_score, components, candidate in sliced:
            results.append(
                SearchResult(
                    id=candidate.id,
                    company_name=candidate.company_name,
                    country=candidate.country,
                    score=round(final_score, 4),
                    score_components=components,
                    long_offering=candidate.long_offering,
                )
            )

        return results

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """Extract meaningful keywords from query text (3+ chars, lowered)."""
        # Remove common stop words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "is", "are", "was", "were",
            "not", "no", "that", "this", "it", "its", "has", "have", "had",
            "do", "does", "did", "be", "been", "being", "who", "which",
            "what", "where", "when", "how", "all", "each", "every",
        }
        words = re.findall(r"[a-zA-Z]{3,}", text.lower())
        return [w for w in words if w not in stop_words]
