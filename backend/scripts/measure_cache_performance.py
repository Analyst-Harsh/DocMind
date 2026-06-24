"""
Drive the live /query endpoint across the RAGAS golden set and record
cost/latency/cache-hit per question, for the before/after caching
comparison.

Usage:
  # Day 1 baseline -- run against today's /query endpoint, before any
  # caching code exists (or later, with ENABLE_SEMANTIC_CACHE=false):
  python -m scripts.measure_cache_performance --mode baseline

  # Post-caching warm run -- run against a server with caching enabled.
  # Populates the cache with all 35 questions, then re-runs the
  # paraphrased wording from eval/cache_threshold_pairs.yaml to generate
  # hits:
  python -m scripts.measure_cache_performance --mode warm
"""

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
import yaml

from app.eval.golden_dataset import load_corpus_texts
from app.eval.ragas_dataset import load_ragas_dataset

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_GOLDEN_DATASET = "eval/ragas_dataset.yaml"
DEFAULT_PAIRS_FILE = "eval/cache_threshold_pairs.yaml"


@dataclass
class QueryRecord:
    question: str
    cost_usd: float
    latency_ms: int
    cache_hit: bool


@dataclass
class PerformanceSummary:
    n: int
    avg_cost_usd: float
    cache_hit_rate: float
    p50_latency_ms_hit: float | None
    p50_latency_ms_miss: float | None
    p50_latency_ms_overall: float


def query_once(
    client: httpx.Client, base_url: str, question: str
) -> QueryRecord:
    response = client.post(
        f"{base_url}/query",
        json={"question": question, "hybrid": True},
        timeout=60.0,
    )
    response.raise_for_status()
    body = response.json()
    return QueryRecord(
        question=question,
        cost_usd=body["cost_usd"],
        latency_ms=body["latency_ms"],
        cache_hit=body.get("cache_hit", False),
    )


def run_questions(base_url: str, questions: list[str]) -> list[QueryRecord]:
    records: list[QueryRecord] = []
    with httpx.Client() as client:
        for i, question in enumerate(questions, start=1):
            print(f"[{i}/{len(questions)}] {question[:70]}")
            records.append(query_once(client, base_url, question))
    return records


def summarize(records: list[QueryRecord]) -> PerformanceSummary:
    costs = [r.cost_usd for r in records]
    hits = [r for r in records if r.cache_hit]
    misses = [r for r in records if not r.cache_hit]
    return PerformanceSummary(
        n=len(records),
        avg_cost_usd=sum(costs) / len(costs),
        cache_hit_rate=len(hits) / len(records),
        p50_latency_ms_hit=(
            statistics.median([r.latency_ms for r in hits]) if hits else None
        ),
        p50_latency_ms_miss=(
            statistics.median([r.latency_ms for r in misses])
            if misses
            else None
        ),
        p50_latency_ms_overall=statistics.median(
            [r.latency_ms for r in records]
        ),
    )


def load_paraphrases(pairs_path: str) -> list[str]:
    raw = yaml.safe_load(Path(pairs_path).read_text(encoding="utf-8"))
    return [pair["paraphrase"] for pair in raw["paraphrase_pairs"]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure /query cost/latency/cache-hit rate across the golden set."
    )
    parser.add_argument("--mode", choices=["baseline", "warm"], required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--golden-dataset", default=DEFAULT_GOLDEN_DATASET)
    parser.add_argument("--pairs-file", default=DEFAULT_PAIRS_FILE)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    items = load_ragas_dataset(args.golden_dataset, load_corpus_texts())
    questions = [item.question for item in items]
    print(f"Loaded {len(questions)} golden-set questions\n")

    print(f"=== Pass 1: full golden set ({args.mode}) ===")
    records = run_questions(args.base_url, questions)

    if args.mode == "warm":
        paraphrases = load_paraphrases(args.pairs_file)
        print(
            f"\n=== Pass 2: {len(paraphrases)} paraphrased questions (expect hits) ==="
        )
        records += run_questions(args.base_url, paraphrases)

    summary = summarize(records)
    output_path = Path(
        args.output or f"eval/results/cache_performance_{args.mode}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "summary": asdict(summary),
                "records": [asdict(r) for r in records],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nSummary: {summary}")
    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
