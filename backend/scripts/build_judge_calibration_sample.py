# scripts/build_judge_calibration_sample.py
"""
Build a stratified sample of already-RAGAS-scored questions for LLM-judge
calibration (Experiment 10). Pulls per_question records from prior eval runs,
tags each with a pipeline context, and picks the most diagnostic records per
context: the extremes (most likely to reveal judge disagreement) plus a few
mid-range controls (to confirm unremarkable scores really are unremarkable).

Writes two files:
  - eval/results/judge_calibration_sample.json: full records incl. RAGAS scores
  - eval/judge_calibration_blind.json: same records with RAGAS scores stripped,
    for scoring without anchoring on what RAGAS said

Usage:
  python -m scripts.build_judge_calibration_sample
  python -m scripts.build_judge_calibration_sample --per-group-extreme 5 --per-group-middle 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

PIPELINE_CONTEXTS: dict[str, list[str]] = {
    "baseline": ["eval/results/ragas_baseline_1.json"],
    "agentic": [
        "eval/results/agentic_rag_comparison.json",
        "eval/results/agentic_rag_multihop.json",
    ],
    "multimodal": ["eval/results/multimodal_comparison.json"],
    "graph": ["eval/results/graph_rag_multihop.json"],
}


def extremity(record: dict) -> tuple[float, str | None]:
    scored = {m: record[m] for m in METRICS if record.get(m) is not None}
    if not scored:
        return 0.0, None
    metric, score = max(scored.items(), key=lambda kv: abs(kv[1] - 0.5))
    return abs(score - 0.5), metric


def load_records(pipeline_context: str, paths: list[str]) -> list[dict]:
    records = []
    for path in paths:
        source_run = Path(path).stem
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for i, r in enumerate(data["per_question"]):
            ext, dominant_metric = extremity(r)
            records.append(
                {
                    "sample_id": f"{source_run}#{i}",
                    "pipeline_context": pipeline_context,
                    "source_run": source_run,
                    "question": r["question"],
                    "answer": r["answer"],
                    "contexts": r["contexts"],
                    "reference": r["reference"],
                    "category": r["category"],
                    "source_docs": r["source_docs"],
                    "ragas_faithfulness": r.get("faithfulness"),
                    "ragas_answer_relevancy": r.get("answer_relevancy"),
                    "ragas_context_precision": r.get("context_precision"),
                    "ragas_context_recall": r.get("context_recall"),
                    "dominant_metric": dominant_metric,
                    "extremity": ext,
                }
            )
    return records


def select_group(records: list[dict], n_extreme: int, n_middle: int) -> list[dict]:
    ordered = sorted(records, key=lambda r: r["extremity"], reverse=True)
    extremes = ordered[:n_extreme]
    remaining = ordered[n_extreme:]
    if not remaining:
        return extremes
    median = sorted(r["extremity"] for r in remaining)[len(remaining) // 2]
    middle = sorted(remaining, key=lambda r: abs(r["extremity"] - median))[:n_middle]
    return extremes + middle


def render_coverage(sample: list[dict]) -> Table:
    table = Table(title="pipeline_context x dominant_metric coverage")
    table.add_column("pipeline_context")
    for m in METRICS:
        table.add_column(m)
    table.add_column("n")

    for context in PIPELINE_CONTEXTS:
        rows = [r for r in sample if r["pipeline_context"] == context]
        counts = [sum(1 for r in rows if r["dominant_metric"] == m) for m in METRICS]
        table.add_row(context, *(str(c) for c in counts), str(len(rows)))
    return table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a stratified sample for LLM-judge calibration."
    )
    parser.add_argument("--per-group-extreme", type=int, default=5)
    parser.add_argument("--per-group-middle", type=int, default=2)
    parser.add_argument(
        "--sample-out", default="eval/results/judge_calibration_sample.json"
    )
    parser.add_argument("--blind-out", default="eval/judge_calibration_blind.json")
    args = parser.parse_args()

    sample: list[dict] = []
    for pipeline_context, paths in PIPELINE_CONTEXTS.items():
        records = load_records(pipeline_context, paths)
        sample.extend(
            select_group(records, args.per_group_extreme, args.per_group_middle)
        )

    sample_path = Path(args.sample_out)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text(
        json.dumps(
            {
                "generated_from": sorted(
                    {r["source_run"] for r in sample}
                ),
                "records": sample,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    blind_records = [
        {
            "sample_id": r["sample_id"],
            "pipeline_context": r["pipeline_context"],
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["contexts"],
            "reference": r["reference"],
            "category": r["category"],
        }
        for r in sample
    ]
    blind_path = Path(args.blind_out)
    blind_path.parent.mkdir(parents=True, exist_ok=True)
    blind_path.write_text(
        json.dumps({"records": blind_records}, indent=2), encoding="utf-8"
    )

    console = Console()
    console.print(f"Sample built: {len(sample)} records -> {sample_path}")
    console.print(f"Blind copy (no RAGAS scores) -> {blind_path}")
    console.print(render_coverage(sample))


if __name__ == "__main__":
    main()
