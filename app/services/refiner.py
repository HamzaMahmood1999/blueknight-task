from __future__ import annotations

from app.schemas import Action, QueryPayload, RefineRequest, RefineResponse


class QueryRefinerAgent:
    """
    Placeholder for Subtask 1.

    TODO(candidate):
    - consume user intent + history
    - produce strict JSON response
    - normalize refined query against default payload
    - return actions for UI
    """

    async def refine(self, request: RefineRequest) -> RefineResponse:
        raise NotImplementedError("Implement QueryRefinerAgent.refine")


def default_actions() -> list[Action]:
    """Starter UI action contract. Candidate may extend."""
    return [Action(id="ideas", label="Suggest more search ideas")]


def default_query_payload() -> QueryPayload:
    """Default payload used for deterministic query shape."""
    return QueryPayload()

