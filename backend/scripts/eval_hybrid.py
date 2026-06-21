"""
Compare dense-only vs hybrid (dense+BM25) retrieval for recursive chunking
+ text-embedding-3-small against the golden query set.

Usage:
  python -m scripts.eval_hybrid
  python -m scripts.eval_hybrid --k 5 --verbose
"""

import argparse

from rich.console import Console

from app.eval.golden_dataset import load_corpus_texts, load_golden_dataset
from app.eval.runner import StrategyReport, collection_stats, run_strategy
from app.ingestion.indexer import collection_name_for, get_qdrant_client
from app.retrieval.searcher import retrieve_hybrid
from scripts.eval_chunking import render_table

STRATEGY = "recursive"
MODEL = "text-embedding-3-small"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare dense vs hybrid retrieval for recursive chunking."
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--golden-dataset", default="eval/golden_dataset.yaml")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    console = Console()
    doc_texts = load_corpus_texts()
    golden = load_golden_dataset(args.golden_dataset, doc_texts)
    console.print(f"Loaded {len(golden)} golden queries\n")

    client = get_qdrant_client()
    existing = {c.name for c in client.get_collections().collections}

    hybrid_collection = collection_name_for(STRATEGY, MODEL, hybrid=True)

    reports: list[StrategyReport] = []

    if hybrid_collection not in existing:
        console.print(
            f"[red]Collection {hybrid_collection} not found. Run:[/red] "
            f"python -m scripts.ingest --strategy {STRATEGY} --hybrid"
        )
    else:

        def hybrid_fn(query, top_k):
            return retrieve_hybrid(
                query,
                top_k=top_k,
                client=client,
                collection_name=hybrid_collection,
                embedding_model=MODEL,
            )

        report = run_strategy(f"{STRATEGY} (hybrid)", hybrid_fn, golden, args.k)
        report.chunk_count, report.avg_tokens = collection_stats(
            client, hybrid_collection
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


if __name__ == "__main__":
    main()
