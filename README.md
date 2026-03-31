# Refiner and Reranker - Take-Home Spec

## Context

You are building an agentic search workflow for an M&A-style company matching system.

Users describe the kind of company they are looking for in natural language. Your agent must
refine that intent into a structured query, retrieve matching companies, re-rank them, and decide
whether the results are good enough to return - or worth another iteration.

Matching is based on `**long_offering**` - a rich text field describing what each company does,
who it serves, and how it delivers value.

---

## What You Are Given

### 1. A dataset of ~1,000 companies

`data/companies.csv` contains the company corpus. Each row has:

| Column | Type | Notes |
|---|---|---|
| `id` | `str` | unique identifier |
| `company_name` | `text` | |
| `country` | `text` | |
| `long_offering` | `text` | rich text, 100-400 words - **this is the bio** |

You must ingest this CSV into a vector database of your choice (Qdrant, Pinecone, FAISS,
pgvector, etc.), embed the `long_offering` field, and make it queryable. All retrieval and
re-ranking logic should operate on `long_offering`.



### 2. A mock retrieval function

`app/retrieval.py` contains:

```python
def mock_retrieve(query: str, top_k: int) -> list[CompanyResult]:
    ...
```

This is a **placeholder** that simulates latency and transient failures. You must replace its
internals with real vector retrieval over the ingested dataset, keeping the same function
signature. Your retrieval wrapper must handle the operational characteristics described below
as if they still apply to the underlying vector DB:

- ~200-300ms mean latency with occasional spikes
- Transient failures (~5%) raising `RetrievalError`
- Not safe to call with unbounded concurrency

**You must wrap this function** - do not call it directly from business logic or endpoint
handlers.

### 3. Schema stubs

`app/schemas.py` contains placeholder models as the data contract. Do not rename existing
fields - you may add fields where needed.

---

## Example Queries

Use these to ground your design decisions.


| Query                                                             | What "good" looks like                                                               |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `"Vertical SaaS for logistics operators in the UK"`               | `long_offering` describes software product + logistics domain + UK market            |
| `"Industrial software providers with field-deployed delivery"`    | `long_offering` signals on-premise or field deployment, not cloud-only               |
| `"Companies solving onboarding inefficiency for frontline teams"` | Problem + use-case signal present in `long_offering`, not just category tags         |
| `"Fintech companies not focused on payments"`                     | Exclusion intent respected - payments-heavy `long_offering` should rank down or drop |
| `"a"`                                                             | Graceful fallback - must not crash, must return a sensible default response          |


---

## Task Structure

### Subtask 1 - Refinement Agent Loop

Implement `POST /agent/refine`.

The agent must run a **loop**, not a single pass. On each iteration it:

1. Refines the user's message into a structured `QueryPayload`
2. Calls the retrieval pipeline
3. Evaluates whether results are sufficient
4. Either iterates (up to `max_iterations`) or returns

**You define the termination condition.** Document your reasoning - this is a primary evaluation
criterion. A candidate who always exits after one iteration has not met this requirement.

**Input:**


| Field            | Type                  | Notes                             |
| ---------------- | --------------------- | --------------------------------- |
| `message`        | `str`                 | the user's natural language query |
| `base_query`     | `QueryPayload | null` | optional prior structured query   |
| `history`        | `list[dict]`          | prior turns in the session        |
| `max_iterations` | `int`                 | default `3`                       |


**Output:**


| Field             | Type           | Notes                            |
| ----------------- | -------------- | -------------------------------- |
| `refined_query`   | `QueryPayload` |                                  |
| `rationale`       | `str`          | why the loop stopped when it did |
| `actions`         | `list[Action]` |                                  |
| `iterations_used` | `int`          |                                  |
| `meta`            | `dict`         |                                  |


**Requirements:**

- Deterministic JSON contract with robust fallback if model output is malformed
- Required fields normalized via a default payload - no field should ever be missing in the response
- Loop termination logic must be explicit and deterministic

> **Note:** The agent loop in Subtask 1 must call `POST /search/run` internally rather than re-implementing retrieval logic.


---

### Subtask 2 - Retrieval and Re-ranking Pipeline

Implement `POST /search/run`.

**Three stages:**

1. **Vector recall** - call `mock_retrieve(query, top_k=top_k_raw)`, wrapped with retry and
  timeout logic
2. **Post-filtering** - filter results using signals extractable from `long_offering` text
  (e.g. geography mentions, domain keywords, explicit exclusions from the query). Document
   what signals you extract and why.
3. **Re-rank** - produce a final ordered list of `top_k_final` results. The ranking approach
  is your choice; a well-reasoned heuristic is fine. Document your scoring logic.

**Input:**


| Field         | Type           | Notes          |
| ------------- | -------------- | -------------- |
| `query`       | `QueryPayload` |                |
| `top_k_raw`   | `int`          | default `1000` |
| `top_k_final` | `int`          | default `50`   |
| `offset`      | `int`          | default `0`    |


**Output:**


| Field         | Type                 | Notes     |
| ------------- | -------------------- | --------- |
| `results`     | `list[SearchResult]` |           |
| `total`       | `int`                |           |
| `diagnostics` | `Diagnostics`        | see below |


`**Diagnostics` must include:**

- `filtered_count` - results removed in stage 2
- `drop_reasons` - why they were removed (e.g. `{"geography_mismatch": 18, "exclude_term": 7}`)
- `stage_latency_ms` - separate timings for `vector_recall`, `post_filter`, and `rerank`

Per-stage latency is **required** and must be readable in structured logs without a debugger
attached.

---

### Subtask 3 - Production Readiness *(written, ~200 words)*

No code required. Answer this in your README:

> *"The system serves 10,000 queries/day. Result relevance silently degrades - no errors are
> thrown, but users are getting poor matches. How would you detect this before users complain,
> and what is your first operational change?"*

There is no right answer. We are looking for how you reason about silent failure and
observability - not a specific solution.

---

## Observability Requirements *(cross-cutting)*

- Every request must carry a `trace_id` propagated through all stages
- Structured logs emitted at each stage boundary with at minimum:
`trace_id`, `stage`, `duration_ms`, `item_count`
- A latency spike in reranking must be distinguishable from a spike in vector recall from
logs alone

---

## API Contracts

### `POST /agent/refine`

```json
// Request
{
  "message": "industrial software providers for warehouse operations in UK",
  "base_query": null,
  "history": [],
  "max_iterations": 3
}

// Response
{
  "refined_query": {
    "query_text": "industrial software for warehouse operations in UK",
    "geography": ["United Kingdom"],
    "exclusions": []
  },
  "rationale": "Stopped after 2 iterations - second pass produced stable top-10 with sufficient score spread and low filter-drop ratio.",
  "actions": [
    { "id": "show_results", "label": "Show results", "payload": {} }
  ],
  "iterations_used": 2,
  "meta": {}
}
```

### `POST /search/run`

```json
// Request
{
  "query": {
    "query_text": "industrial software for warehouse operations in UK",
    "geography": ["United Kingdom"],
    "exclusions": []
  },
  "top_k_raw": 1000,
  "top_k_final": 50,
  "offset": 0
}

// Response
{
  "results": [
    {
      "id": "company-123",
      "company_name": "Example Co",
      "country": "United Kingdom",
      "score": 0.89,
      "score_components": {
        "vector": 0.81,
        "rerank_boost": 0.08
      },
      "long_offering": "Example Co provides warehouse management software..."
    }
  ],
  "total": 50,
  "diagnostics": {
    "raw_count": 1000,
    "filtered_count": 312,
    "reranked_count": 50,
    "drop_reasons": {
      "geography_mismatch": 200,
      "exclude_term": 57,
      "low_vector_score": 55
    },
    "stage_latency_ms": {
      "vector_recall": 238,
      "post_filter": 41,
      "rerank": 63
    }
  }
}
```

---

## Non-goals

- Perfect ranking quality
- Production-grade authentication
- Frontend UI

---

## Setup & Running Locally

This project requires Python 3.10+.

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment:** Create a `.env` file in the root directory and add your Groq API key:
   ```bash
   GROQ_API_KEY=gsk_your_key_here
   ```

3. **Data Ingestion (One-Time Run):** Pre-compute the local dense embeddings for the dataset. This runs entirely on your CPU and takes ~30 seconds.
   ```bash
   python scripts/ingest.py
   ```

4. **Start the Server:**
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

5. **Launch the UI:** Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser to interact with the frontend Agentic Search Explorer!

---

## Submission (Fork Required)

1. Fork this repository to your own GitHub account.
2. Complete the task in your fork.
3. Commit your changes with clear messages.
4. Share your fork URL (and branch name if not `main`) for review.

---

## Assignment Deliverables

### 1. Assumptions Made
- **API Limits & Local Embeddings**: I assumed that we shouldn't be blocked by API rate limits (e.g., free tier limits failing to embed 1,000 bios). Therefore, I embedded the dataset locally using `SentenceTransformers` (`all-MiniLM-L6-v2`), which embeds entirely on CPU in ~30 seconds, and cached it to disk (FAISS + numpy).
- **Fast Generation**: I assumed JSON output consistency is paramount, so I utilized Groq's API with `llama-3.3-70b-versatile` operating in strict `json_object` mode.
- **Geography Overlap**: If the user doesn't specify geography, the requirement is assumed to be global (no geographic post-filters applied).

### 2. Termination Condition Rationale
The Refinement Agent loop terminates early (before `max_iterations`) if it detects that further prompting is unlikely to yield better results. This relies on 4 multi-signal heuristics evaluated on each turn:
1. **Query Identity**: If the LLM produces the exact same `query_text`, `geography`, and `exclusions` as the previous turn, it has logically converged. It halts immediately.
2. **Score Stability Threshold**: If the top-10 IDs returned by the search match 80%+ of the top-10 IDs from the previous iteration, the semantic neighborhood is stable. Further word-tweaks won't surface radically different companies.
3. **Filter Drop Ratio**: If the number of results dropped during post-filtering (geography/exclusions) sits below 50% of the raw recalled results, the query constraints are well-aligned with the dataset distribution.
4. **Score Quality Mean**: If the mean score of the top-10 candidates exceeds `0.55` (a mathematically strong cosine-similarity baseline post-reranker weights), the results are deemed factually "good enough" for the user.

### 3. Subtask 3 - Production Readiness
> *"The system serves 10,000 queries/day. Result relevance silently degrades - no errors are thrown, but users are getting poor matches. How would you detect this before users complain, and what is your first operational change?"*

**Detection Strategies:**
- **Zero-Click & Reformulation Rates**: I would instrument the platform to track when users run a query and either click *none* of the results or immediately re-type a slightly different query. A spike in the Query Reformulation Rate indicates silent relevance failure.
- **Top-K Score Drift Monitoring**: The observability layer currently logs all component scores for each search. I would build a dashboard tracking the `p50` and `p90` final matching scores for the top-5 results of every query. If the rolling average top score decays over a week, the embedding space is drifting from the user search vocabulary.

**First Operational Change:**
- **Examine Logs & Adjust Reranking Weights**: I would pull the structured logs filtering for high reformulations. If users are searching for explicit hard-requirements (e.g., "SOC2 compliance") and dense vectors are retrieving semantically related but factually incorrect companies ("HIPAA compliant"), I would quickly intervene by increasing the `RERANK_WEIGHT_KEYWORD_BOOST` in the config. This instantly increases the penalty on lexical mismatches, restoring factual precision while the team investigates fine-tuning the core embedding model itself.

### 4. What I Would Do With More Time
- **Dedicated Vector DB**: Move from in-memory FAISS to a persistent vector database like Qdrant or Milvus to support real-time CRUD insertions of new M&A targets without needing to re-pickle an index array.
- **Dense/Sparse Hybrid Search**: Upgrade the vector recall stage from pure dense embeddings to a Hybrid search architecture (Dense vectors + Sparse BM25/Splade). This inherently solves the lexical mismatch problem without needing a manual keyword-boost in the Python reranker.
- **Async LLM Streaming**: Pipe the refinement agent's inner-thoughts and iteration progress back to the user via Server-Sent Events (SSE) so a Frontend UI could display *"Refining your search... Checking geographic constraints..."* rather than making the user wait 15 seconds for a bulk JSON response.

