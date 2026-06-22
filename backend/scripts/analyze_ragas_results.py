# scripts/analyze_ragas_results.py
"""
Analyze a saved RAGAS run: worst performers per metric (to spot real
pipeline problems vs RAGAS judgment quirks) and metric averages broken down
by golden-set category.

Usage:
  python -m scripts.analyze_ragas_results
  python -m scripts.analyze_ragas_results --results eval/results/ragas_baseline.json --bottom 3 --verbose
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


def render_worst(records: list[dict], metric: str, n: int) -> Table:
    worst = sorted(records, key=lambda r: r[metric])[:n]
    table = Table(title=f"Lowest {metric}")
    table.add_column("score")
    table.add_column("category")
    table.add_column("question")
    table.add_column("answer")
    for r in worst:
        table.add_row(
            f"{r[metric]:.2f}", r["category"], r["question"], r["answer"][:80]
        )
    return table


def print_worst_detail(
    records: list[dict], metric: str, n: int, console: Console
) -> None:
    worst = sorted(records, key=lambda r: r[metric])[:n]
    for r in worst:
        console.print(
            f"\n[bold]{metric}={r[metric]:.2f}[/bold]  ({r['category']})  {r['question']}"
        )
        console.print(f"  reference: {r['reference']}")
        console.print(f"  answer:    {r['answer']}")
        for i, ctx in enumerate(r["contexts"]):
            console.print(f"  context[{i}]: {ctx[:300]}")


def render_by_category(records: list[dict]) -> Table:
    by_category: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_category[r["category"]].append(r)

    table = Table(title="Metric averages by category")
    table.add_column("category")
    for m in METRICS:
        table.add_column(m)
    table.add_column("n")

    for category, rows in sorted(by_category.items()):
        averages = [sum(r[m] for r in rows) / len(rows) for m in METRICS]
        table.add_row(category, *(f"{a:.2f}" for a in averages), str(len(rows)))

    overall = [sum(r[m] for r in records) / len(records) for m in METRICS]
    table.add_row("overall", *(f"{a:.2f}" for a in overall), str(len(records)))
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a saved RAGAS run.")
    parser.add_argument("--results", default="eval/results/ragas_baseline.json")
    parser.add_argument("--bottom", type=int, default=3)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full question/reference/answer/contexts for worst performers.",
    )
    args = parser.parse_args()

    data = json.loads(Path(args.results).read_text(encoding="utf-8"))
    records = data["per_question"]

    console = Console()
    console.print(f"Aggregate: {data['aggregate']}\n")

    for metric in METRICS:
        console.print(render_worst(records, metric, args.bottom))
        if args.verbose:
            print_worst_detail(records, metric, args.bottom, console)
        console.print()

    console.print(render_by_category(records))


if __name__ == "__main__":
    main()
