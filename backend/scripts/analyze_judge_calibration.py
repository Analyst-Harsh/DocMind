# scripts/analyze_judge_calibration.py
"""
Compare RAGAS's scores against a second, independent LLM-judge scoring pass
(Experiment 10's LLM-judge calibration), and flag per-metric disagreements.

Usage:
  python -m scripts.analyze_judge_calibration
  python -m scripts.analyze_judge_calibration --threshold 0.15 --verbose
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def load_joined(sample_path: str, manual_path: str) -> list[dict]:
    sample = json.loads(Path(sample_path).read_text(encoding="utf-8"))["records"]
    manual = json.loads(Path(manual_path).read_text(encoding="utf-8"))["scores"]

    sample_by_id = {r["sample_id"]: r for r in sample}
    manual_by_id = {s["sample_id"]: s for s in manual}

    if sample_by_id.keys() != manual_by_id.keys():
        missing_manual = sample_by_id.keys() - manual_by_id.keys()
        missing_sample = manual_by_id.keys() - sample_by_id.keys()
        raise ValueError(
            f"sample_id mismatch between {sample_path} and {manual_path}: "
            f"missing from manual={sorted(missing_manual)}, "
            f"missing from sample={sorted(missing_sample)}"
        )

    joined = []
    for sample_id, rec in sample_by_id.items():
        joined.append({**rec, **manual_by_id[sample_id]})
    return joined


def compute_disagreements(joined: list[dict], threshold: float) -> list[dict]:
    disagreements = []
    for rec in joined:
        for metric in METRICS:
            ragas_score = rec.get(f"ragas_{metric}")
            manual_score = rec.get(f"manual_{metric}")
            if ragas_score is None or manual_score is None:
                continue
            gap = abs(manual_score - ragas_score)
            if gap > threshold:
                disagreements.append(
                    {
                        "sample_id": rec["sample_id"],
                        "pipeline_context": rec["pipeline_context"],
                        "source_run": rec["source_run"],
                        "category": rec["category"],
                        "metric": metric,
                        "ragas_score": ragas_score,
                        "manual_score": manual_score,
                        "gap": gap,
                        "manual_reasoning": rec.get(f"manual_{metric}_reasoning", ""),
                    }
                )
    disagreements.sort(key=lambda d: d["gap"], reverse=True)
    return disagreements


def render_by_metric(joined: list[dict], disagreements: list[dict]) -> Table:
    table = Table(title="Disagreement rate by metric")
    table.add_column("metric")
    table.add_column("n scored")
    table.add_column("n disagree")
    table.add_column("rate")
    for metric in METRICS:
        n_scored = sum(1 for r in joined if r.get(f"ragas_{metric}") is not None)
        n_disagree = sum(1 for d in disagreements if d["metric"] == metric)
        rate = n_disagree / n_scored if n_scored else 0.0
        table.add_row(metric, str(n_scored), str(n_disagree), f"{rate:.1%}")
    return table


def render_by_pipeline(joined: list[dict], disagreements: list[dict]) -> Table:
    table = Table(title="Disagreement rate by pipeline_context")
    table.add_column("pipeline_context")
    table.add_column("n scores")
    table.add_column("n disagree")
    table.add_column("rate")
    contexts = sorted({r["pipeline_context"] for r in joined})
    for context in contexts:
        n_scores = sum(
            1
            for r in joined
            if r["pipeline_context"] == context
            for m in METRICS
            if r.get(f"ragas_{m}") is not None
        )
        n_disagree = sum(
            1 for d in disagreements if d["pipeline_context"] == context
        )
        rate = n_disagree / n_scores if n_scores else 0.0
        table.add_row(context, str(n_scores), str(n_disagree), f"{rate:.1%}")
    return table


def render_disagreements(disagreements: list[dict]) -> Table:
    table = Table(title="Disagreements (sorted by gap desc)")
    table.add_column("sample_id")
    table.add_column("pipeline")
    table.add_column("category")
    table.add_column("metric")
    table.add_column("ragas")
    table.add_column("manual")
    table.add_column("gap")
    for d in disagreements:
        table.add_row(
            d["sample_id"],
            d["pipeline_context"],
            d["category"],
            d["metric"],
            f"{d['ragas_score']:.2f}",
            f"{d['manual_score']:.2f}",
            f"{d['gap']:.2f}",
        )
    return table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare RAGAS scores against a second LLM-judge pass."
    )
    parser.add_argument(
        "--sample", default="eval/results/judge_calibration_sample.json"
    )
    parser.add_argument("--manual", default="eval/judge_calibration_manual_scores.json")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument(
        "--out", default="eval/results/judge_calibration_disagreements.json"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full question/answer/first context for each disagreement.",
    )
    args = parser.parse_args()

    joined = load_joined(args.sample, args.manual)
    disagreements = compute_disagreements(joined, args.threshold)

    n_scores = sum(1 for r in joined for m in METRICS if r.get(f"ragas_{m}") is not None)

    console = Console()
    console.print(f"Sample size: {len(joined)} records, {n_scores} metric scores\n")
    console.print(
        f"Disagreements (gap > {args.threshold}): {len(disagreements)} "
        f"({len(disagreements) / n_scores:.1%})\n"
    )
    console.print(render_by_metric(joined, disagreements))
    console.print()
    console.print(render_by_pipeline(joined, disagreements))
    console.print()
    console.print(render_disagreements(disagreements))

    if args.verbose:
        for d in disagreements:
            rec = next(r for r in joined if r["sample_id"] == d["sample_id"])
            console.print(
                f"\n[bold]{d['sample_id']}[/bold] {d['metric']}: "
                f"ragas={d['ragas_score']:.2f} manual={d['manual_score']:.2f}"
            )
            console.print(f"  question: {rec['question']}")
            console.print(f"  answer:   {rec['answer']}")
            console.print(f"  context[0]: {rec['contexts'][0][:300]}")
            console.print(f"  manual reasoning: {d['manual_reasoning']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(disagreements, indent=2), encoding="utf-8")
    console.print(f"\nWrote {len(disagreements)} disagreements -> {out_path}")


if __name__ == "__main__":
    main()
