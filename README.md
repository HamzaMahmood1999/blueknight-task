# Plan C Interview Task: Agentic Refiner + Re-ranker

This repository is an interview task scaffold.

It intentionally contains:
- API contracts
- placeholder modules
- TODO markers
- evaluation criteria

It intentionally does **not** contain:
- a reference implementation
- hidden solution code

## Goal

Build an iterative "query refiner agent" that:
1. proposes refinements from user input,
2. runs vector retrieval,
3. re-ranks candidates,
4. returns next-step UI actions and diagnostics.

## Task Structure (3 Subtasks)

### Subtask 1: Refinement Agent

Implement `POST /agent/refine`:
- input: base query + optional prior results
- output: strict JSON with:
  - refined query object
  - short rationale
  - suggested actions

Requirements:
- deterministic JSON contract
- robust fallback if model output is malformed
- required fields normalized via a default payload

### Subtask 2: Retrieval + Re-ranking Pipeline

Implement `POST /search/run`:
- stage 1: vector recall (`top_k_raw`)
- stage 2: metadata post-filtering
- stage 3: lightweight re-rank (`top_k_final`)

Return:
- final ranked results
- diagnostics:
  - filtered count
  - drop reasons
  - score components

Use any vector DB:
- Qdrant, Pinecone, Weaviate, pgvector, or FAISS (local)

### Subtask 3: Performance + Reliability

Add:
- bounded concurrency around remote calls
- timeout budgets
- retry policy for transient failures
- benchmark script for p50/p95 latency under load

Deliver:
- before/after benchmark summary
- short tradeoff note

## API Contracts

### 1) `POST /agent/refine`

Request:
- `thread_id: str`
- `message: str`
- `base_query: QueryPayload | null`
- `history: list[dict]`

Response:
- `refined_query: QueryPayload`
- `rationale: str`
- `actions: list[Action]`
- `meta: dict`

### 2) `POST /search/run`

Request:
- `query: QueryPayload`
- `top_k_raw: int` (default 5000)
- `top_k_final: int` (default 100)
- `offset: int` (default 0)

Response:
- `results: list[SearchResult]`
- `total: int`
- `diagnostics: Diagnostics`

### 3) `GET /health`

Simple liveness response.

## Suggested Data Model

Use `app/schemas.py` placeholders as contract.

## Non-goals

- perfect ranking model
- production-grade auth
- UI

## Setup

1. Create a Python env.
2. Install dependencies from your own choices.
3. Run:
   - `uvicorn app.main:app --reload`
4. Run tests:
   - `pytest -q`

## Candidate Deliverables

- working API for all 3 subtasks
- tests for contracts and edge cases
- benchmark output and short analysis
- README update describing assumptions

## Evaluation Rubric

- Correctness (35%)
- Robustness (20%)
- Performance awareness (25%)
- Code quality and maintainability (20%)

