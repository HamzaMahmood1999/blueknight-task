from fastapi import FastAPI

from app.schemas import RefineRequest, RefineResponse, SearchRequest, SearchResponse
from app.services.refiner import QueryRefinerAgent
from app.services.search_pipeline import SearchPipeline

app = FastAPI(title="Interview Task Plan C")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/agent/refine", response_model=RefineResponse)
async def refine(request: RefineRequest) -> RefineResponse:
    """
    TODO(candidate): implement refinement flow.
    Must:
    - run agent refinement
    - validate/normalize output
    - return deterministic contract
    """
    agent = QueryRefinerAgent()
    raise NotImplementedError("Implement /agent/refine")


@app.post("/search/run", response_model=SearchResponse)
async def search_run(request: SearchRequest) -> SearchResponse:
    """
    TODO(candidate): implement vector retrieval + re-rank flow.
    Must:
    - recall top_k_raw
    - post-filter
    - re-rank top_k_final
    - return diagnostics
    """
    pipeline = SearchPipeline()
    raise NotImplementedError("Implement /search/run")

