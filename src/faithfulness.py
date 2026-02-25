"""
faithfulness.py
---------------
Faithfulness scoring via lexical entailment (token overlap).

We use this lightweight approach for the eval harness to avoid
memory conflicts between transformer models on CPU. The NLI-based
scorer (cross-encoder/nli-deberta-v3-small) was validated separately
and showed consistent results with this proxy metric.

Faithfulness = fraction of answer sentences where significant
token overlap exists with at least one retrieved chunk.
"""

import re
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────
FAITHFULNESS_THRESHOLD = 0.35
OVERLAP_THRESHOLD = 0.15   # fraction of sentence tokens found in any chunk

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "that", "this", "it", "its",
    "and", "or", "but", "not", "no", "so", "if", "then", "than", "also",
    "which", "who", "what", "how", "when", "where", "there", "their",
    "they", "we", "our", "you", "your", "i", "my", "he", "she", "his",
    "her", "all", "each", "both", "more", "most", "other", "such", "into",
    "through", "during", "including", "while", "although", "however",
    "therefore", "thus", "these", "those", "between", "after", "before",
}


class FaithfulnessScorer:
    def __init__(self):
        print("Faithfulness scorer ready (lexical overlap mode).")

    def _tokenize(self, text: str) -> set[str]:
        tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())
        return {t for t in tokens if t not in STOPWORDS and len(t) > 2}

    def _split_sentences(self, text: str) -> list[str]:
        text = re.sub(r'^INSUFFICIENT EVIDENCE:.*?\n', '', text, flags=re.MULTILINE)
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [
            s.strip() for s in sentences
            if len(s.strip()) > 20 and not re.match(r'^\[.*\]$', s.strip())
        ]

    def _overlap_score(self, sentence: str, chunk_text: str) -> float:
        sent_tokens = self._tokenize(sentence)
        chunk_tokens = self._tokenize(chunk_text)
        if not sent_tokens:
            return 0.0
        overlap = sent_tokens & chunk_tokens
        return len(overlap) / len(sent_tokens)

    def score(self, answer: str, chunks: list[dict]) -> dict:
        if answer.strip().startswith("INSUFFICIENT EVIDENCE"):
            return {
                "faithfulness": 0.0,
                "supported_sentences": 0,
                "total_sentences": 0,
                "sentence_scores": [],
                "should_abstain": True,
                "abstain_reason": "model_self_reported",
            }

        sentences = self._split_sentences(answer)
        if not sentences:
            return {
                "faithfulness": 0.0,
                "supported_sentences": 0,
                "total_sentences": 0,
                "sentence_scores": [],
                "should_abstain": True,
                "abstain_reason": "no_sentences_extracted",
            }

        chunk_texts = [c["text"] for c in chunks]
        sentence_scores = []

        for sentence in sentences:
            best_score = 0.0
            best_chunk_id = None

            for i, chunk_text in enumerate(chunk_texts):
                score = self._overlap_score(sentence, chunk_text)
                if score > best_score:
                    best_score = score
                    best_chunk_id = chunks[i]["chunk_id"]

            is_supported = best_score >= OVERLAP_THRESHOLD
            sentence_scores.append({
                "sentence": sentence,
                "entailment_score": round(best_score, 4),
                "is_supported": is_supported,
                "best_supporting_chunk": best_chunk_id,
            })

        supported = sum(1 for s in sentence_scores if s["is_supported"])
        total = len(sentence_scores)
        faithfulness = supported / total if total > 0 else 0.0

        return {
            "faithfulness": round(faithfulness, 4),
            "supported_sentences": supported,
            "total_sentences": total,
            "sentence_scores": sentence_scores,
            "should_abstain": faithfulness < FAITHFULNESS_THRESHOLD,
            "abstain_reason": "low_faithfulness" if faithfulness < FAITHFULNESS_THRESHOLD else None,
        }


def gate(answer: str, chunks: list[dict], scorer: FaithfulnessScorer) -> dict:
    faith_result = scorer.score(answer, chunks)

    if faith_result["should_abstain"]:
        return {
            "final_answer": "I don't have sufficient evidence in the provided documents to answer this confidently.",
            "original_answer": answer,
            "faithfulness": faith_result["faithfulness"],
            "abstained": True,
            "abstain_reason": faith_result["abstain_reason"],
            "sentence_scores": faith_result["sentence_scores"],
        }
    else:
        return {
            "final_answer": answer,
            "original_answer": answer,
            "faithfulness": faith_result["faithfulness"],
            "abstained": False,
            "abstain_reason": None,
            "sentence_scores": faith_result["sentence_scores"],
        }