"""Refinement agent with multi-signal termination loop.

The agent runs an iterative refinement loop:
1. LLM refines user intent → structured QueryPayload
2. Calls search pipeline internally
3. Evaluates result quality via multiple signals
4. Iterates or terminates with documented rationale

Termination conditions (any triggers exit):
- Score stability: top-10 result IDs ≥80% overlap with previous iteration
- Score quality: mean top-10 score ≥ threshold AND spread ≤ threshold
- Low filter-drop ratio: filtered_count/raw_count < threshold → good alignment
- Query identity: LLM produced same query_text as previous iteration
- Max iterations reached
"""
from __future__ import annotations

import logging
import time
from typing import Any
from uuid import uuid4

from app import config
from app.schemas import (
    Action,
    QueryPayload,
    RefineRequest,
    RefineResponse,
    SearchRequest,
)
from app.services.llm_client import LLMClient
from app.services.search_pipeline import SearchPipeline
from app.utils.logging import structured_log, timed_stage

logger = logging.getLogger("blueknight")


class QueryRefinerAgent:
    """Agentic query refiner with multi-iteration loop and smart termination."""

    def __init__(self) -> None:
        self._llm = LLMClient()
        self._pipeline = SearchPipeline()

    async def refine(self, request: RefineRequest) -> RefineResponse:
        """Run the refinement loop."""
        trace_id = request.trace_id
        max_iter = request.max_iterations
        overall_start = time.perf_counter()

        # State across iterations
        prev_top_ids: set[str] = set()
        prev_query_text: str = ""
        all_actions: list[Action] = []
        iteration_details: list[dict[str, Any]] = []
        termination_reason: str = ""
        current_query = request.base_query or default_query_payload()
        token_usage_total: dict[str, int] = {}

        for iteration in range(1, max_iter + 1):
            with timed_stage(trace_id, f"refine_iteration_{iteration}") as ctx:

                # ── Step 1: LLM refinement ─────────────────────────
                previous_results_summary = None
                if iteration > 1 and iteration_details:
                    prev = iteration_details[-1]
                    previous_results_summary = (
                        f"Top 5 results: {prev.get('top_5_summary', 'N/A')}\n"
                        f"Total results: {prev.get('total', 0)}\n"
                        f"Filtered out: {prev.get('filtered_count', 0)}\n"
                        f"Drop reasons: {prev.get('drop_reasons', {})}\n"
                        f"Mean top-10 score: {prev.get('mean_score', 0):.3f}"
                    )

                llm_result = await self._llm.refine_query(
                    user_message=request.message,
                    history=request.history,
                    previous_results_summary=previous_results_summary,
                    previous_query=(
                        current_query.model_dump() if iteration > 1 else None
                    ),
                )

                # ── Step 2: Parse & normalize into QueryPayload ────
                current_query = self._normalize_query(llm_result, current_query)

                # Track token usage
                usage = llm_result.get("_usage", {})
                for k, v in usage.items():
                    if isinstance(v, (int, float)):
                        token_usage_total[k] = token_usage_total.get(k, 0) + int(v)

                # ── Step 3: Run search pipeline ────────────────────
                search_request = SearchRequest(
                    query=current_query,
                    top_k_raw=config.DEFAULT_TOP_K_RAW,
                    top_k_final=config.DEFAULT_TOP_K_FINAL,
                    trace_id=trace_id,
                )
                search_response = await self._pipeline.run(search_request)

                # ── Step 4: Evaluate termination ───────────────────
                results = search_response.results
                diagnostics = search_response.diagnostics

                # Current top-10 analysis
                top_10 = results[:10]
                current_top_ids = {r.id for r in top_10}
                top_10_scores = [r.score for r in top_10] if top_10 else [0.0]
                mean_score = sum(top_10_scores) / len(top_10_scores)
                score_spread = (
                    max(top_10_scores) - min(top_10_scores) if top_10 else 0.0
                )
                filter_ratio = (
                    diagnostics.filtered_count / diagnostics.raw_count
                    if diagnostics.raw_count > 0
                    else 0.0
                )

                # Build iteration summary for logs
                top_5_summary = [
                    {
                        "id": r.id,
                        "name": r.company_name[:40],
                        "score": round(r.score, 3),
                    }
                    for r in results[:5]
                ]

                iter_detail = {
                    "iteration": iteration,
                    "query_text": current_query.query_text,
                    "geography": current_query.geography,
                    "exclusions": current_query.exclusions,
                    "total": search_response.total,
                    "filtered_count": diagnostics.filtered_count,
                    "drop_reasons": diagnostics.drop_reasons,
                    "mean_score": mean_score,
                    "score_spread": score_spread,
                    "filter_ratio": filter_ratio,
                    "top_5_summary": top_5_summary,
                    "reasoning": llm_result.get("reasoning", ""),
                }
                iteration_details.append(iter_detail)
                ctx["item_count"] = search_response.total
                ctx["mean_score"] = round(mean_score, 3)

                # ── Termination checks ─────────────────────────────

                # Check 1: Score stability — top-10 IDs overlap with previous
                if prev_top_ids and current_top_ids:
                    overlap = len(current_top_ids & prev_top_ids) / max(
                        len(current_top_ids), 1
                    )
                    if overlap >= config.SCORE_STABILITY_OVERLAP_THRESHOLD:
                        termination_reason = (
                            f"Stopped after {iteration} iterations — "
                            f"top-10 results stabilized ({overlap:.0%} overlap with previous iteration). "
                            f"Mean score: {mean_score:.3f}, filter drop ratio: {filter_ratio:.1%}."
                        )
                        break

                # Check 2: Score quality — good enough results
                if (
                    mean_score >= config.SCORE_QUALITY_MEAN_THRESHOLD
                    and score_spread <= config.SCORE_QUALITY_SPREAD_THRESHOLD
                    and len(top_10) >= 5
                ):
                    termination_reason = (
                        f"Stopped after {iteration} iterations — "
                        f"results meet quality threshold (mean score {mean_score:.3f}, "
                        f"spread {score_spread:.3f}). Filter drop ratio: {filter_ratio:.1%}."
                    )
                    break

                # Check 3: Low filter-drop ratio — well-targeted query
                if (
                    iteration >= 2
                    and filter_ratio < config.FILTER_DROP_RATIO_THRESHOLD
                    and mean_score >= 0.40
                ):
                    termination_reason = (
                        f"Stopped after {iteration} iterations — "
                        f"low filter drop ratio ({filter_ratio:.1%}) indicates "
                        f"well-targeted query. Mean score: {mean_score:.3f}."
                    )
                    break

                # Check 4: Query unchanged — LLM has converged
                if current_query.query_text == prev_query_text and iteration > 1:
                    termination_reason = (
                        f"Stopped after {iteration} iterations — "
                        f"query text unchanged from previous iteration, "
                        f"no further refinement possible. Mean score: {mean_score:.3f}."
                    )
                    break

                # Update state for next iteration
                prev_top_ids = current_top_ids
                prev_query_text = current_query.query_text

        else:
            # Max iterations reached without early termination
            termination_reason = (
                f"Reached maximum iterations ({max_iter}). "
                f"Final mean top-10 score: {mean_score:.3f}, "
                f"filter drop ratio: {filter_ratio:.1%}."
            )

        # ── Build response ─────────────────────────────────────────
        overall_ms = (time.perf_counter() - overall_start) * 1000
        structured_log(
            trace_id=trace_id,
            stage="refine_complete",
            duration_ms=overall_ms,
            iterations_used=iteration,
            termination_reason=termination_reason[:100],
        )

        actions = default_actions()
        if search_response.total > 0:
            actions.append(
                Action(id="show_results", label="Show results", payload={})
            )

        return RefineResponse(
            refined_query=current_query,
            rationale=termination_reason,
            actions=actions,
            iterations_used=iteration,
            meta={
                "iteration_details": iteration_details,
                "total_duration_ms": round(overall_ms, 1),
                "token_usage": token_usage_total,
            },
        )

    def _normalize_query(
        self,
        llm_output: dict[str, Any],
        fallback: QueryPayload,
    ) -> QueryPayload:
        """Merge LLM output with default payload — no field is ever missing."""
        try:
            return QueryPayload(
                query_text=llm_output.get("query_text", fallback.query_text) or fallback.query_text,
                geography=llm_output.get("geography", fallback.geography) or fallback.geography,
                exclusions=llm_output.get("exclusions", fallback.exclusions) or fallback.exclusions,
            )
        except Exception:
            return fallback


def default_actions() -> list[Action]:
    """Starter UI action contract. Candidate may extend."""
    return [Action(id="ideas", label="Suggest more search ideas")]


def default_query_payload() -> QueryPayload:
    """Default payload used for deterministic query shape."""
    return QueryPayload()
