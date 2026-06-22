# scripts/run_ragas_eval.py
"""
Run the RAGAS golden set through the live pipeline (hybrid retrieval ->
rerank -> generate), score each result with RAGAS (faithfulness,
answer_relevancy, context_precision, context_recall), and save per-question
+ aggregate scores.

Usage:
  python -m scripts.run_ragas_eval
  python -m scripts.run_ragas_eval --golden-dataset eval/ragas_dataset.yaml --output eval/results/ragas_baseline.json
"""

import argparse
import json
from pathlib import Path

from app.eval.golden_dataset import load_corpus_texts
from app.eval.ragas_dataset import load_ragas_dataset
from app.eval.ragas_runner import (
    build_metrics,
    run_all,
    score_all,
    to_per_question_records,
)

METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


def aggregate(records: list[dict]) -> dict:
    return {m: sum(r[m] for r in records) / len(records) for m in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RAGAS eval against the live pipeline."
    )
    parser.add_argument("--golden-dataset", default="eval/ragas_dataset.yaml")
    parser.add_argument("--output", default="eval/results/ragas_baseline.json")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max questions scored concurrently (each fires ~4 LLM calls; "
        "lower this if you hit OpenAI rate limits).",
    )
    args = parser.parse_args()

    items = load_ragas_dataset(args.golden_dataset, load_corpus_texts())
    print(f"Loaded {len(items)} RAGAS golden-set questions\n")

    print("Running pipeline (retrieve -> rerank -> generate)...")
    results = run_all(items)

    print("\nScoring with RAGAS...")
    metrics = build_metrics()
    scores = score_all(results, metrics, max_concurrency=args.concurrency)

    records = to_per_question_records(results, scores)
    output = {"aggregate": aggregate(records), "per_question": records}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\nAggregate scores: {output['aggregate']}")
    print(f"Wrote {len(records)} per-question results to {output_path}")


if __name__ == "__main__":
    main()
