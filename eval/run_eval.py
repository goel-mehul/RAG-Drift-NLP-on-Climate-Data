"""
run_eval.py
-----------
Runs all 3 pipeline variants across all questions and drift levels.
Saves results to eval/results.csv.

This is the core evaluation harness — the research artifact.
"""

import sys
import json
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import RAGPipeline, SharedResources

QUESTIONS_FILE = Path("eval/questions.json")
RESULTS_FILE = Path("eval/results.csv")


def flatten_question(q: dict) -> list[dict]:
    """Turn one question dict into a list of (id, drift_level, query) rows."""
    rows = []

    # Base question = drift level 0
    rows.append({
        "question_id": q["id"],
        "drift_level": 0,
        "drift_type": "base",
        "query": q["base"],
        "question_level": q["level"],
    })

    # Drifted variants
    drift_map = {
        "1_paraphrase": 1,
        "2_scope": 2,
        "3_entity": 3,
        "4_negation": 4,
    }
    for drift_key, drift_num in drift_map.items():
        if drift_key in q["drifts"]:
            rows.append({
                "question_id": q["id"],
                "drift_level": drift_num,
                "drift_type": drift_key,
                "query": q["drifts"][drift_key],
                "question_level": q["level"],
            })

    return rows


def run_eval():
    # Load questions
    with open(QUESTIONS_FILE) as f:
        questions = json.load(f)

    all_rows = []
    for q in questions:
        all_rows.extend(flatten_question(q))

    total = len(all_rows) * 3  # 3 pipeline variants
    print(f"Running eval: {len(all_rows)} queries × 3 pipelines = {total} total runs")
    print("This will take a while on CPU. Go get coffee.\n")

    # Load shared resources ONCE
    resources = SharedResources()

    pipelines = {
        "baseline": RAGPipeline("baseline", resources),
        "hybrid": RAGPipeline("hybrid", resources),
        "full": RAGPipeline("full", resources),
    }

    results = []
    run_num = 0

    for row in all_rows:
        for variant_name, pipeline in pipelines.items():
            run_num += 1
            print(f"[{run_num}/{total}] {variant_name} | q={row['question_id']} drift={row['drift_level']} | {row['query'][:60]}...")

            try:
                result = pipeline.run(row["query"], top_k=5)

                results.append({
                    "question_id": row["question_id"],
                    "question_level": row["question_level"],
                    "drift_level": row["drift_level"],
                    "drift_type": row["drift_type"],
                    "variant": variant_name,
                    "query": row["query"],
                    "faithfulness": result["faithfulness"],
                    "abstained": result["abstained"],
                    "answer_length": len(result["final_answer"]),
                    "num_chunks_used": len(result["chunks_used"]),
                    "chunk_levels": ",".join(result["chunk_levels"]),
                    "cited_chunks": ",".join(result["cited_chunk_ids"]),
                    "latency_sec": result["latency_sec"],
                    "answer_preview": result["final_answer"][:200],
                })

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    "question_id": row["question_id"],
                    "question_level": row["question_level"],
                    "drift_level": row["drift_level"],
                    "drift_type": row["drift_type"],
                    "variant": variant_name,
                    "query": row["query"],
                    "faithfulness": None,
                    "abstained": None,
                    "answer_length": None,
                    "num_chunks_used": None,
                    "chunk_levels": None,
                    "cited_chunks": None,
                    "latency_sec": None,
                    "answer_preview": f"ERROR: {e}",
                })

    # Save results
    df = pd.DataFrame(results)
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    df.to_csv(RESULTS_FILE, index=False)

    print(f"\nDone! Results saved to {RESULTS_FILE}")
    print(f"Total runs: {len(df)}")

    # Quick summary
    full_df = df[df["variant"] == "full"].copy()
    print("\nQuick faithfulness summary (full pipeline):")
    print(full_df.groupby("drift_level")["faithfulness"].mean().round(3))
    print(f"\nAbstain rate by drift level:")
    print(full_df.groupby("drift_level")["abstained"].mean().round(3))


if __name__ == "__main__":
    run_eval()