"""Standalone script to pre-compute embeddings and build FAISS index.

Run this ONCE before starting the server:
    python scripts/ingest.py

Embeddings are cached to disk, so subsequent server starts are instant.
"""
import os
import sys
import time
import pickle
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import faiss
import pandas as pd
from dotenv import load_dotenv
from google import genai

load_dotenv()

# Config
CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "companies.csv"
DATA_DIR = CSV_PATH.parent
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
INDEX_PATH = DATA_DIR / "index.faiss"
METADATA_PATH = DATA_DIR / "metadata.pkl"
EMBEDDING_MODEL = "gemini-embedding-001"
API_KEY = os.getenv("GEMINI_API_KEY", "")

# Checkpoint file to resume if interrupted
CHECKPOINT_PATH = DATA_DIR / "_embed_checkpoint.pkl"


def main():
    print(f"Loading companies from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    df = df.rename(columns={
        "Consolidated ID": "id",
        "Company Name": "company_name",
        "Country": "country",
        "Long Offering": "long_offering",
    })
    df["id"] = df["id"].astype(str)
    df = df[["id", "company_name", "country", "long_offering"]].copy()
    df["long_offering"] = df["long_offering"].fillna("")
    
    metadata = df.to_dict("records")
    texts = [m["long_offering"] for m in metadata]
    print(f"Loaded {len(texts)} companies")

    # Check if we already have cached embeddings
    if EMBEDDINGS_PATH.exists() and INDEX_PATH.exists() and METADATA_PATH.exists():
        print("Cache files already exist! Skipping embedding computation.")
        print("Delete data/embeddings.npy, data/index.faiss, data/metadata.pkl to recompute.")
        return

    client = genai.Client(api_key=API_KEY)

    # Try to resume from checkpoint
    all_embeddings = []
    start_idx = 0
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "rb") as f:
            checkpoint = pickle.load(f)
        all_embeddings = checkpoint["embeddings"]
        start_idx = checkpoint["next_idx"]
        print(f"Resuming from checkpoint: {start_idx}/{len(texts)} texts already embedded")

    # Embed one at a time with generous delays
    for i in range(start_idx, len(texts)):
        text = texts[i] if texts[i].strip() else "empty"
        
        # Retry with exponential backoff
        for attempt in range(10):
            try:
                response = client.models.embed_content(
                    model=f"models/{EMBEDDING_MODEL}",
                    contents=[text],
                )
                all_embeddings.append(response.embeddings[0].values)
                break
            except Exception as e:
                if attempt < 9 and ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) 
                                    or "RetryInfo" in str(e) or "503" in str(e)):
                    wait = min(2 ** (attempt + 1), 120)
                    print(f"  Rate limited on {i+1}, waiting {wait}s (attempt {attempt+1}/10)")
                    time.sleep(wait)
                else:
                    print(f"FATAL error on text {i+1}: {e}")
                    # Save checkpoint before dying
                    _save_checkpoint(all_embeddings, i)
                    raise

        # Progress
        if (i + 1) % 25 == 0 or i + 1 == len(texts):
            print(f"  [{i+1}/{len(texts)}] embedded")
        
        # Save checkpoint every 100 texts
        if (i + 1) % 100 == 0:
            _save_checkpoint(all_embeddings, i + 1)
            print(f"  Checkpoint saved at {i+1}")

        # Delay between requests — 4 requests per minute on free tier = 15s per request
        # But we can usually do better, start with 2s and backoff handles the rest
        if i + 1 < len(texts):
            time.sleep(2)

    # Build FAISS index
    print("Building FAISS index...")
    embeddings = np.array(all_embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    # Save everything
    print("Saving cache files...")
    np.save(str(EMBEDDINGS_PATH), embeddings)
    faiss.write_index(index, str(INDEX_PATH))
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    
    # Clean up checkpoint
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
    
    print(f"Done! Index has {index.ntotal} vectors of dimension {dim}")
    print("You can now start the server with: uvicorn app.main:app --reload")


def _save_checkpoint(embeddings: list, next_idx: int):
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump({"embeddings": embeddings, "next_idx": next_idx}, f)


if __name__ == "__main__":
    main()
