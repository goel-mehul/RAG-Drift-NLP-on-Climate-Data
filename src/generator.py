"""
generator.py
------------
Takes a query + retrieved chunks and generates an answer using
Llama 3.1 via Ollama. Forces citation and honest uncertainty.
"""

import ollama
import json

# ── Config ────────────────────────────────────────────────────────────────────
MODEL = "llama3.1"
MAX_CONTEXT_CHUNKS = 5  # how many chunks to feed into the prompt

SYSTEM_PROMPT = """You are a climate policy analyst assistant. You answer questions
strictly based on the provided policy document excerpts.

Rules you must follow:
1. Only use information from the provided excerpts to answer.
2. After each claim, cite the source using [chunk_id].
3. If the excerpts do not contain enough information to answer, say exactly:
   "INSUFFICIENT EVIDENCE: The provided documents do not contain enough information to answer this question confidently."
4. Never speculate or use outside knowledge.
5. Be concise and precise."""

def build_prompt(query: str, chunks: list[dict]) -> str:
    """Format retrieved chunks into a prompt for the LLM."""
    context_parts = []
    for i, chunk in enumerate(chunks[:MAX_CONTEXT_CHUNKS]):
        context_parts.append(
            f"[{chunk['chunk_id']}] ({chunk['level'].upper()} - {chunk['doc_name']})\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""Here are relevant excerpts from climate policy documents:

{context}

---

Question: {query}

Answer (cite chunk IDs inline, e.g. [chunk_id]):"""

    return prompt


def generate(query: str, chunks: list[dict]) -> dict:
    """
    Generate an answer given a query and retrieved chunks.
    Returns a dict with answer text and metadata.
    """
    prompt = build_prompt(query, chunks)

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.1},  # low temp = more faithful, less creative
    )

    answer = response["message"]["content"]

    # Extract which chunk_ids were cited in the answer
    import re
    cited_ids = re.findall(r'\[([^\]]+_c\d+)\]', answer)

    return {
        "query": query,
        "answer": answer,
        "cited_chunk_ids": cited_ids,
        "chunks_used": [c["chunk_id"] for c in chunks[:MAX_CONTEXT_CHUNKS]],
        "insufficient_evidence": answer.strip().startswith("INSUFFICIENT EVIDENCE"),
    }


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from retriever import HybridRetriever

    retriever = HybridRetriever()

    test_queries = [
        "What are NYC's building emissions limits under Local Law 97?",
        "What emission reduction targets does New York State aim for by 2050?",
        "What is the global temperature goal in the Paris Agreement?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print('='*60)

        chunks = retriever.retrieve(query, top_k=5)
        result = generate(query, chunks)

        print(f"\nANSWER:\n{result['answer']}")
        print(f"\nCITED: {result['cited_chunk_ids']}")
        print(f"INSUFFICIENT EVIDENCE: {result['insufficient_evidence']}")