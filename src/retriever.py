"""
retriever.py
------------
Hybrid retrieval: combines dense (FAISS) + sparse (BM25) results
using Reciprocal Rank Fusion (RRF).

RRF formula: score(d) = sum(1 / (k + rank(d))) for each ranked list
This is a proven, parameter-light fusion method used in production systems.
"""

import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
INDEX_DIR = Path("faiss_index")
EMBED_MODEL = "all-MiniLM-L6-v2"

RRF_K = 60          # RRF constant (60 is standard, from the original paper)
TOP_K_EACH = 20     # how many results to pull from each index before fusion
TOP_K_FINAL = 10    # how many to return after fusion


class HybridRetriever:
    def __init__(self):
        print("Loading retriever indexes...")

        # Load FAISS
        self.faiss_index = faiss.read_index(str(INDEX_DIR / "index.faiss"))

        # Load BM25
        with open(INDEX_DIR / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)

        # Load chunk metadata
        with open(INDEX_DIR / "chunks_meta.json") as f:
            self.chunks = json.load(f)

        # Load embedding model
        self.model = SentenceTransformer(EMBED_MODEL)

        print(f"  Retriever ready. {len(self.chunks)} chunks available.")

    def _dense_retrieve(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Return (chunk_index, score) pairs from FAISS."""
        query_vec = self.model.encode([query], show_progress_bar=False).astype("float32")
        faiss.normalize_L2(query_vec)

        scores, indices = self.faiss_index.search(query_vec, top_k)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]

    def _sparse_retrieve(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Return (chunk_index, score) pairs from BM25."""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)

        # Get top_k indices sorted by score descending
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[tuple[int, float]],
        sparse_results: list[tuple[int, float]],
        k: int = RRF_K,
    ) -> list[tuple[int, float]]:
        """
        Fuse two ranked lists using RRF.
        Returns list of (chunk_index, rrf_score) sorted by score descending.
        """
        rrf_scores: dict[int, float] = {}

        for rank, (idx, _) in enumerate(dense_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)

        for rank, (idx, _) in enumerate(sparse_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)

        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

    def retrieve(self, query: str, top_k: int = TOP_K_FINAL) -> list[dict]:
        """
        Main retrieval method.
        Returns list of chunk dicts with an added 'retrieval_score' field.
        """
        dense_results = self._dense_retrieve(query, TOP_K_EACH)
        sparse_results = self._sparse_retrieve(query, TOP_K_EACH)

        fused = self._reciprocal_rank_fusion(dense_results, sparse_results)

        # Build output: chunk metadata + score
        retrieved = []
        for chunk_idx, rrf_score in fused[:top_k]:
            chunk = self.chunks[chunk_idx].copy()
            chunk["retrieval_score"] = round(rrf_score, 6)
            retrieved.append(chunk)

        return retrieved


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    retriever = HybridRetriever()

    test_queries = [
        "What are NYC's greenhouse gas emission targets?",
        "How does the Climate Leadership and Community Protection Act work?",
        "What does the Paris Agreement require from member countries?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retriever.retrieve(query, top_k=3)
        for i, r in enumerate(results):
            print(f"  [{i+1}] [{r['level'].upper()}] {r['doc_name']} | score={r['retrieval_score']}")
            print(f"       {r['text'][:120]}...")