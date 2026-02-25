"""
pipeline.py
-----------
Assembles the full RAG pipeline with model-sharing to avoid
memory issues on Mac CPU (segfault from multiple model loads).
"""

import time
import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from generator import generate
from faithfulness import FaithfulnessScorer, gate, FAITHFULNESS_THRESHOLD

INDEX_DIR = Path("faiss_index")
EMBED_MODEL = "all-MiniLM-L6-v2"


class SharedResources:
    """
    Load all models and indexes ONCE and share across pipelines.
    This prevents the segfault from multiple model instantiations.
    """
    def __init__(self):
        print("Loading all shared resources...")

        # Embedding model
        print("  [1/4] Embedding model...")
        self.embed_model = SentenceTransformer(EMBED_MODEL)

        # FAISS index
        print("  [2/4] FAISS index...")
        self.faiss_index = faiss.read_index(str(INDEX_DIR / "index.faiss"))

        # BM25 index
        print("  [3/4] BM25 index...")
        with open(INDEX_DIR / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)

        # Chunk metadata
        print("  [4/4] Chunk metadata...")
        with open(INDEX_DIR / "chunks_meta.json") as f:
            self.chunks = json.load(f)

        # NLI scorer (loaded once, shared)
        self.scorer = FaithfulnessScorer()

        print(f"All resources loaded. {len(self.chunks)} chunks available.\n")


class RAGPipeline:
    def __init__(self, variant: str, resources: SharedResources):
        assert variant in ("baseline", "hybrid", "full")
        self.variant = variant
        self.res = resources

    def _dense_retrieve(self, query: str, top_k: int) -> list[dict]:
        query_vec = self.res.embed_model.encode(
            [query], show_progress_bar=False
        ).astype("float32")
        faiss.normalize_L2(query_vec)
        scores, indices = self.res.faiss_index.search(query_vec, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:
                chunk = self.res.chunks[int(idx)].copy()
                chunk["retrieval_score"] = float(score)
                results.append(chunk)
        return results

    def _sparse_retrieve(self, query: str, top_k: int) -> list[tuple[int, float]]:
        tokens = query.lower().split()
        scores = self.res.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]

    def _hybrid_retrieve(self, query: str, top_k: int) -> list[dict]:
        dense = self._dense_retrieve(query, top_k * 2)
        sparse = self._sparse_retrieve(query, top_k * 2)

        # RRF fusion
        rrf: dict[int, float] = {}
        for rank, chunk in enumerate(dense):
            idx = self.res.chunks.index(
                next(c for c in self.res.chunks if c["chunk_id"] == chunk["chunk_id"])
            )
            rrf[idx] = rrf.get(idx, 0) + 1 / (60 + rank + 1)

        for rank, (idx, _) in enumerate(sparse):
            rrf[idx] = rrf.get(idx, 0) + 1 / (60 + rank + 1)

        sorted_idx = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for idx, score in sorted_idx:
            chunk = self.res.chunks[idx].copy()
            chunk["retrieval_score"] = round(score, 6)
            results.append(chunk)
        return results

    def run(self, query: str, top_k: int = 5) -> dict:
        t0 = time.time()

        # Retrieve
        if self.variant == "baseline":
            chunks = self._dense_retrieve(query, top_k)
        else:
            chunks = self._hybrid_retrieve(query, top_k)

        # Generate
        gen_result = generate(query, chunks)

        # Faithfulness gate
        if self.variant == "full":
            faith_result = gate(gen_result["answer"], chunks, self.res.scorer)
            final_answer = faith_result["final_answer"]
            faithfulness = faith_result["faithfulness"]
            abstained = faith_result["abstained"]
            sentence_scores = faith_result["sentence_scores"]
        else:
            final_answer = gen_result["answer"]
            faithfulness = None
            abstained = False
            sentence_scores = []

        return {
            "query": query,
            "variant": self.variant,
            "final_answer": final_answer,
            "faithfulness": faithfulness,
            "abstained": abstained,
            "cited_chunk_ids": gen_result["cited_chunk_ids"],
            "chunks_used": [c["chunk_id"] for c in chunks],
            "chunk_levels": [c["level"] for c in chunks],
            "sentence_scores": sentence_scores,
            "latency_sec": round(time.time() - t0, 2),
        }


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load everything ONCE
    resources = SharedResources()

    # Share across all three pipelines
    baseline = RAGPipeline("baseline", resources)
    hybrid = RAGPipeline("hybrid", resources)
    full = RAGPipeline("full", resources)

    query = "What are NYC's greenhouse gas emission reduction targets?"

    print("\n--- BASELINE ---")
    r1 = baseline.run(query)
    print(f"Answer: {r1['final_answer'][:300]}")
    print(f"Chunks from levels: {r1['chunk_levels']}")

    print("\n--- HYBRID ---")
    r2 = hybrid.run(query)
    print(f"Answer: {r2['final_answer'][:300]}")
    print(f"Chunks from levels: {r2['chunk_levels']}")

    print("\n--- FULL (hybrid + gate) ---")
    r3 = full.run(query)
    print(f"Answer: {r3['final_answer'][:300]}")
    print(f"Faithfulness: {r3['faithfulness']}")
    print(f"Abstained: {r3['abstained']}")