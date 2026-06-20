"""
Evaluate chunking strategies against the golden query set.

Usage:
  python -m scripts.eval_chunking
  python -m scripts.eval_chunking --strategies fixed_size,recursive --k 5 --verbose
  python -m scripts.eval_chunking --output report.json
  python -m scripts.eval_chunking --embedding-model text-embedding-3-large
"""

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from app.config import get_settings
from app.eval.golden_dataset import load_corpus_texts, load_golden_dataset
from app.eval.matcher import MATCH_THRESHOLD
from app.eval.runner import StrategyReport, collection_stats, run_strategy
from app.ingestion.chunker.base_chunker import ChunkStrategy
from app.ingestion.indexer import collection_name_for, get_qdrant_client
from app.retrieval.searcher import retrieve

ALL_STRATEGIES = [s.value for s in ChunkStrategy]


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


def build_report_json(
    reports: list[StrategyReport], k: int, embedding_model: str
) -> dict:
    return {
        "k": k,
        "embedding_model": embedding_model,
        "match_threshold": MATCH_THRESHOLD,
        "strategies": [report_to_dict(r) for r in reports],
    }


def render_table(reports: list[StrategyReport], k: int) -> Table:
    table = Table(title=f"Chunking strategy eval (k={k})")
    table.add_column("Strategy", style="bold")
    table.add_column(f"Precision@{k}", justify="right")
    table.add_column(f"Recall@{k}", justify="right")
    table.add_column(f"MRR@{k}", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Avg tokens", justify="right")

    def _best(attr):
        return max((getattr(r.aggregate, attr) for r in reports), default=0.0)

    best_p, best_r, best_m = _best("precision"), _best("recall"), _best("mrr")

    def _cell(value, best):
        text = f"{value:.3f}"
        return (
            f"[green bold]{text}[/green bold]"
            if value == best and value > 0
            else text
        )

    for r in reports:
        table.add_row(
            r.strategy,
            _cell(r.aggregate.precision, best_p),
            _cell(r.aggregate.recall, best_r),
            _cell(r.aggregate.mrr, best_m),
            str(r.chunk_count),
            f"{r.avg_tokens:.0f}",
        )
    return table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate chunking strategies."
    )
    parser.add_argument(
        "--strategies",
        default=",".join(ALL_STRATEGIES),
        help=(
            "Comma-separated chunk strategies to evaluate "
            f"(default: all of {ALL_STRATEGIES})."
        ),
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
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model to query against (default: settings.embedding_model). "
        "Must match a model already ingested into a Qdrant collection.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    console = Console()
    model = args.embedding_model or settings.embedding_model
    strategies = [s.strip() for s in args.strategies.split(",")]

    doc_texts = load_corpus_texts()
    golden = load_golden_dataset(args.golden_dataset, doc_texts)
    console.print(f"Loaded {len(golden)} golden queries (model: {model})\n")

    client = get_qdrant_client()
    existing = {c.name for c in client.get_collections().collections}

    reports: list[StrategyReport] = []
    for strategy in strategies:
        collection = collection_name_for(strategy, model)
        if collection not in existing:
            console.print(
                f"[red]Collection {collection} not found. Run:[/red] "
                f"python -m scripts.ingest --strategy {strategy}"
            )
            continue

        def retrieve_fn(query, top_k, _col=collection, _model=model):
            return retrieve(
                query,
                top_k=top_k,
                client=client,
                collection_name=_col,
                embedding_model=_model,
            )

        report = run_strategy(strategy, retrieve_fn, golden, args.k)
        report.chunk_count, report.avg_tokens = collection_stats(
            client, collection
        )
        reports.append(report)

    if not reports:
        console.print("[red]No collections evaluated. Ingest first.[/red]")
        return

    console.print(render_table(reports, args.k))

    if args.verbose:
        for r in reports:
            console.print(f"\n[bold]{r.strategy}[/bold] per-query:")
            for q in r.per_query:
                console.print(
                    f"  P={q.precision:.2f} R={q.recall:.2f} "
                    f"RR={q.reciprocal_rank:.2f}  {q.query}"
                )

    if args.output:
        Path(args.output).write_text(
            json.dumps(build_report_json(reports, args.k, model), indent=2),
            encoding="utf-8",
        )
        console.print(f"\nWrote report to {args.output}")


if __name__ == "__main__":
    main()
