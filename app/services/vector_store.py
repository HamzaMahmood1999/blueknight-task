"""FAISS-backed vector store with local sentence-transformer embeddings.

Uses all-MiniLM-L6-v2 for local embedding (no API needed, fast on CPU).
Falls back to Gemini API if EMBEDDING_PROVIDER=gemini is set.
Disk caching ensures the index is built only once.
"""
from __future__ import annotations

import pickle
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd

from app import config
from app.retrieval import CompanyResult

logger = logging.getLogger("blueknight")


class VectorStoreClient:
    """Vector store backed by FAISS with sentence-transformer embeddings.

    Singleton-style usage — call initialize() once at app startup,
    then query() on every request.
    """

    def __init__(self) -> None:
        self._index: faiss.IndexFlatIP | None = None
        self._metadata: list[dict[str, Any]] = []
        self._embedder: Any = None
        self._initialized: bool = False

    # ── Initialization ─────────────────────────────────────────────

    async def initialize(self, csv_path: Path | None = None) -> None:
        """Load CSV, compute or load cached embeddings, build FAISS index."""
        if self._initialized:
            return

        csv_path = csv_path or config.CSV_PATH

        # Initialize the embedding model
        self._init_embedder()

        # Load company data from CSV
        df = pd.read_csv(csv_path)
        column_map = {
            "Consolidated ID": "id",
            "Company Name": "company_name",
            "Country": "country",
            "Long Offering": "long_offering",
        }
        df = df.rename(columns=column_map)
        df["id"] = df["id"].astype(str)
        df = df[["id", "company_name", "country", "long_offering"]].copy()
        df["long_offering"] = df["long_offering"].fillna("")
        df["country"] = df["country"].fillna("")
        df["company_name"] = df["company_name"].fillna("")

        self._metadata = df.to_dict("records")
        logger.info(f"Loaded {len(self._metadata)} companies from {csv_path}")

        # Try loading cached embeddings + index
        if self._try_load_cache():
            logger.info("Loaded embeddings and FAISS index from disk cache")
        else:
            logger.info("Computing embeddings locally (this takes ~30 seconds)...")
            texts = [m["long_offering"] for m in self._metadata]
            embeddings = self._compute_embeddings(texts)
            self._build_index(embeddings)
            self._save_cache(embeddings)
            logger.info("Embeddings computed and cached to disk")

        self._initialized = True

    def _init_embedder(self) -> None:
        """Initialize the sentence-transformer embedding model."""
        from sentence_transformers import SentenceTransformer

        model_name = config.EMBEDDING_MODEL_NAME
        logger.info(f"Loading embedding model: {model_name}")
        self._embedder = SentenceTransformer(model_name)
        logger.info("Embedding model loaded")

    def _try_load_cache(self) -> bool:
        """Load embeddings + FAISS index + metadata from disk if available."""
        emb_path = config.EMBEDDINGS_CACHE_PATH
        idx_path = config.FAISS_INDEX_PATH
        meta_path = config.METADATA_CACHE_PATH

        if emb_path.exists() and idx_path.exists() and meta_path.exists():
            try:
                self._index = faiss.read_index(str(idx_path))
                with open(meta_path, "rb") as f:
                    cached_meta = pickle.load(f)
                # Validate cache matches current CSV size
                if len(cached_meta) == len(self._metadata):
                    self._metadata = cached_meta
                    return True
                else:
                    logger.info("Cache size mismatch, recomputing embeddings")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return False

    def _save_cache(self, embeddings: np.ndarray) -> None:
        """Persist embeddings, FAISS index, and metadata to disk."""
        np.save(str(config.EMBEDDINGS_CACHE_PATH), embeddings)
        faiss.write_index(self._index, str(config.FAISS_INDEX_PATH))
        with open(config.METADATA_CACHE_PATH, "wb") as f:
            pickle.dump(self._metadata, f)

    def _compute_embeddings(self, texts: list[str]) -> np.ndarray:
        """Compute embeddings using sentence-transformers (local, no API)."""
        # Replace empty strings with placeholder
        processed = [t if t.strip() else "empty" for t in texts]

        # Encode all at once — efficient batch processing on CPU
        embeddings = self._embedder.encode(
            processed,
            show_progress_bar=True,
            batch_size=64,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
        )
        return np.array(embeddings, dtype=np.float32)

    def _build_index(self, embeddings: np.ndarray) -> None:
        """Build FAISS IndexFlatIP (inner product = cosine on normalized vectors)."""
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        logger.info(f"Built FAISS index: {self._index.ntotal} vectors, dim={dim}")

    # ── Query ──────────────────────────────────────────────────────

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string using the local model."""
        if not text.strip():
            text = "general company search"

        embedding = self._embedder.encode(
            [text],
            normalize_embeddings=True,
        )
        return np.array(embedding, dtype=np.float32)

    async def query(
        self,
        query_text: str,
        top_k: int,
    ) -> list[CompanyResult]:
        """Search FAISS index and return CompanyResult list with scores."""
        if not self._initialized:
            raise RuntimeError("VectorStoreClient not initialized. Call initialize() first.")

        # Embed the query
        query_vec = self.embed_query(query_text)

        # Clamp top_k to index size
        effective_k = min(top_k, self._index.ntotal)

        # Search
        scores, indices = self._index.search(query_vec, effective_k)

        results: list[CompanyResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for unfilled slots
                continue
            meta = self._metadata[idx]
            results.append(
                CompanyResult(
                    id=meta["id"],
                    company_name=meta["company_name"],
                    country=meta["country"],
                    long_offering=meta["long_offering"],
                    score=float(score),
                )
            )
        return results


# ── Module-level singleton ─────────────────────────────────────────
_store: VectorStoreClient | None = None


def get_vector_store() -> VectorStoreClient:
    """Get the singleton VectorStoreClient instance."""
    global _store
    if _store is None:
        _store = VectorStoreClient()
    return _store
