# scripts/eval_graph.py
"""
Evaluate Neo4j knowledge-graph retrieval against the golden query set,
using the same precision/recall/MRR harness as scripts/eval_chunking.py so
results are directly comparable to the Qdrant-based strategies.

Usage:
  python -m scripts.eval_graph
  python -m scripts.eval_graph --k 5 --verbose
  python -m scripts.eval_graph --output report.json
"""

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from app.config import get_settings
from app.eval.golden_dataset import load_corpus_texts, load_golden_dataset
from app.eval.matcher import MATCH_THRESHOLD
from app.eval.runner import StrategyReport, run_strategy
from app.graph.client import get_neo4j_driver
from app.graph.graph_searcher import retrieve_graph


def graph_stats(driver, database: str) -> tuple[int, float]:
    """Return (chunk_count, avg_tokens) from Chunk nodes in Neo4j."""
    records, _, _ = driver.execute_query(
        "MATCH (c:Chunk) RETURN count(c) AS count, "
        "avg(c.token_count) AS avg_tokens",
        database_=database,
    )
    count = records[0]["count"] or 0
    avg_tokens = records[0]["avg_tokens"] or 0.0
    return count, avg_tokens


def report_to_dict(report: StrategyReport) -> dict:
    return {
        "strategy": report.strategy,
        "precision": report.aggregate.precision,
        "recall": report.aggregate.recall,
        "mrr": report.aggregate.mrr,
        "num_queries": report.aggregate.num_queries,
        "chunk_count": report.chunk_count,
        "avg_tokens": report.avg_tokens,
        "per_query": [
            {
                "query": q.query,
                "precision": q.precision,
                "recall": q.recall,
                "reciprocal_rank": q.reciprocal_rank,
                "matched_items": q.matched_items,
                "total_items": q.total_items,
            }
            for q in report.per_query
        ],
    }


def render_table(report: StrategyReport, k: int) -> Table:
    table = Table(title=f"Graph retrieval eval (k={k})")
    table.add_column("Strategy", style="bold")
    table.add_column(f"Precision@{k}", justify="right")
    table.add_column(f"Recall@{k}", justify="right")
    table.add_column(f"MRR@{k}", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Avg tokens", justify="right")
    table.add_row(
        report.strategy,
        f"{report.aggregate.precision:.3f}",
        f"{report.aggregate.recall:.3f}",
        f"{report.aggregate.mrr:.3f}",
        str(report.chunk_count),
        f"{report.avg_tokens:.0f}",
    )
    return table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Neo4j graph retrieval."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k chunks to retrieve per query when computing "
        "precision/recall/MRR.",
    )
    parser.add_argument(
        "--golden-dataset",
        default="eval/golden_dataset.yaml",
        help="Path to the golden query/relevant-chunk YAML dataset.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="If set, write the full per-query JSON report to this path.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    console = Console()

    doc_texts = load_corpus_texts()
    golden = load_golden_dataset(args.golden_dataset, doc_texts)
    console.print(f"Loaded {len(golden)} golden queries\n")

    driver = get_neo4j_driver()
    chunk_count, _ = graph_stats(driver, settings.neo4j_database)
    if chunk_count == 0:
        console.print(
            "[red]No Chunk nodes found in Neo4j. Run:[/red] "
            "python -m scripts.ingest_graph"
        )
        return

    def retrieve_fn(query, top_k):
        return retrieve_graph(query, top_k=top_k, driver=driver, rerank=True)

    report = run_strategy("graph", retrieve_fn, golden, args.k)
    report.chunk_count, report.avg_tokens = graph_stats(
        driver, settings.neo4j_database
    )

    console.print(render_table(report, args.k))

    if args.verbose:
        console.print(f"\n[bold]{report.strategy}[/bold] per-query:")
        for q in report.per_query:
            console.print(
                f"  P={q.precision:.2f} R={q.recall:.2f} "
                f"RR={q.reciprocal_rank:.2f}  {q.query}"
            )

    if args.output:
        Path(args.output).write_text(
            json.dumps(
                {
                    "k": args.k,
                    "match_threshold": MATCH_THRESHOLD,
                    "strategies": [report_to_dict(report)],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        console.print(f"\nWrote report to {args.output}")


if __name__ == "__main__":
    main()
