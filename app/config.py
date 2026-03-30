"""Centralized configuration for the agentic search system."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "companies.csv"
EMBEDDINGS_CACHE_PATH = DATA_DIR / "embeddings.npy"
FAISS_INDEX_PATH = DATA_DIR / "index.faiss"
METADATA_CACHE_PATH = DATA_DIR / "metadata.pkl"

# ── Embedding model (local, no API needed) ─────────────────────────
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"  # 384 dimensions, fast on CPU
EMBEDDING_DIMENSIONS: int = 384

# ── Groq API (used only for LLM refinement in Subtask 1) ─────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
LLM_MODEL: str = "llama-3.3-70b-versatile"

# ── Retrieval ──────────────────────────────────────────────────────
RETRIEVAL_MAX_RETRIES: int = 3
RETRIEVAL_BACKOFF_BASE: float = 0.5        # exponential backoff base (seconds)
RETRIEVAL_TIMEOUT_SECONDS: float = 2.0     # per-attempt timeout
RETRIEVAL_CONCURRENCY_LIMIT: int = 5       # semaphore slots
RETRIEVAL_SIMULATED_FAILURE_RATE: float = 0.05
RETRIEVAL_SIMULATED_LATENCY_MIN_MS: int = 180
RETRIEVAL_SIMULATED_LATENCY_MAX_MS: int = 320

# ── Search pipeline defaults ───────────────────────────────────────
DEFAULT_TOP_K_RAW: int = 1000
DEFAULT_TOP_K_FINAL: int = 50
MIN_VECTOR_SCORE_THRESHOLD: float = 0.30

# ── Reranker weights ──────────────────────────────────────────────
RERANK_WEIGHT_VECTOR: float = 0.85
RERANK_WEIGHT_GEO_BOOST: float = 0.05
RERANK_WEIGHT_KEYWORD_BOOST: float = 0.10

# ── Agent loop ─────────────────────────────────────────────────────
DEFAULT_MAX_ITERATIONS: int = 3
SCORE_STABILITY_OVERLAP_THRESHOLD: float = 0.80   # top-10 ID overlap to consider stable
SCORE_QUALITY_MEAN_THRESHOLD: float = 0.55         # mean top-10 score for "good enough"
SCORE_QUALITY_SPREAD_THRESHOLD: float = 0.35       # max spread in top-10 scores
FILTER_DROP_RATIO_THRESHOLD: float = 0.50          # filtered/raw below this = good alignment
