"""
index.py
--------
Takes chunks from data/processed/chunks.json and builds two indexes:
  1. FAISS index (dense, embedding-based)
  2. BM25 index (sparse, keyword-based)

Both are saved to disk so we don't re-embed every time.
"""

import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ── Config ────────────────────────────────────────────────────────────────────
CHUNKS_FILE = Path("data/processed/chunks.json")
INDEX_DIR = Path("faiss_index")

EMBED_MODEL = "all-MiniLM-L6-v2"  # fast, good quality, runs well on CPU
BATCH_SIZE = 64

def load_chunks():
    with open(CHUNKS_FILE) as f:
        return json.load(f)

def build_faiss_index(chunks: list[dict], model: SentenceTransformer):
    """Embed all chunks and store in a FAISS flat index."""
    print(f"\nEmbedding {len(chunks)} chunks with {EMBED_MODEL}...")
    
    texts = [c["text"] for c in chunks]
    
    # Embed in batches so you can see progress
    all_embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding batches"):
        batch = texts[i : i + BATCH_SIZE]
        embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.append(embeddings)
    
    embeddings_matrix = np.vstack(all_embeddings).astype("float32")
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings_matrix)
    
    # Build flat index (exact search — fine for 1829 chunks)
    dim = embeddings_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product = cosine after normalization
    index.add(embeddings_matrix)
    
    print(f"  FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index, embeddings_matrix

def build_bm25_index(chunks: list[dict]):
    """Build a BM25 keyword index over chunk texts."""
    print("\nBuilding BM25 index...")
    
    # Tokenize: lowercase, split on whitespace
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    
    print(f"  BM25 index built over {len(tokenized)} documents")
    return bm25

def main():
    INDEX_DIR.mkdir(exist_ok=True)
    
    # Load chunks
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")
    
    # Load embedding model
    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    
    # Build FAISS
    faiss_index, embeddings = build_faiss_index(chunks, model)
    
    # Save FAISS index
    faiss.write_index(faiss_index, str(INDEX_DIR / "index.faiss"))
    np.save(INDEX_DIR / "embeddings.npy", embeddings)
    print(f"  Saved FAISS index to {INDEX_DIR}/")
    
    # Build BM25
    bm25 = build_bm25_index(chunks)
    
    # Save BM25 (pickle)
    with open(INDEX_DIR / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    print(f"  Saved BM25 index to {INDEX_DIR}/bm25.pkl")
    
    # Save chunk metadata separately (needed at query time)
    with open(INDEX_DIR / "chunks_meta.json", "w") as f:
        json.dump(chunks, f)
    print(f"  Saved chunk metadata to {INDEX_DIR}/chunks_meta.json")
    
    print("\nAll indexes built successfully.")
    print(f"  Total chunks indexed: {len(chunks)}")

if __name__ == "__main__":
    main()