"""FastAPI application — wires up endpoints and startup initialization."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.schemas import RefineRequest, RefineResponse, SearchRequest, SearchResponse
from app.services.refiner import QueryRefinerAgent
from app.services.search_pipeline import SearchPipeline
from app.services.vector_store import get_vector_store

logger = logging.getLogger("blueknight")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialize vector store (load CSV, build/load FAISS index)."""
    logger.info("Initializing vector store...")
    store = get_vector_store()
    await store.initialize()
    logger.info("Vector store ready. Server accepting requests.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Refiner and Reranker",
    description="Agentic search workflow for M&A company matching",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount the frontend directory
import os
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def serve_frontend():
    """Serve the frontend UI."""
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/health")
async def health():
    """Health check endpoint."""
    store = get_vector_store()
    return {
        "status": "ok",
        "vector_store_initialized": store._initialized,
        "companies_loaded": len(store._metadata),
    }


@app.post("/agent/refine", response_model=RefineResponse)
async def refine(request: RefineRequest) -> RefineResponse:
    """Subtask 1: Refinement agent loop.

    Runs an iterative refinement loop that:
    1. Refines the user's message into a structured QueryPayload via LLM
    2. Calls the search pipeline (/search/run internally)
    3. Evaluates whether results are sufficient via multiple signals
    4. Either iterates (up to max_iterations) or returns
    """
    agent = QueryRefinerAgent()
    return await agent.refine(request)


@app.post("/search/run", response_model=SearchResponse)
async def search_run(request: SearchRequest) -> SearchResponse:
    """Subtask 2: Vector retrieval + post-filter + re-rank pipeline.

    Three stages:
    1. Vector recall — retrieve top_k_raw candidates from FAISS
    2. Post-filter — geography, exclusion terms, low score threshold
    3. Re-rank — composite scoring with score component breakdown
    """
    pipeline = SearchPipeline()
    return await pipeline.run(request)
