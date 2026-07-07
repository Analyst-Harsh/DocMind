# scripts/run_multimodal_comparison.py
"""
Compare the text-only baseline pipeline against the multimodal (table +
figure) pipeline on eval/multimodal_dataset.yaml — a question set whose
answers live in table cells, figure captions, or a mix of prose and
table/figure content.

Both pipelines use the same embedding model, hybrid+rerank retrieval, and
generation model, querying the text-only recursive-chunking collection vs.
the multimodal collection (tables as KV text, figures as GPT-4o Vision
captions) respectively. The multimodal arm additionally reserves
multimodal_slots (default 2) of its top-k for table/figure chunks via two
independently Qdrant-filtered searches (app.retrieval.searcher's
retrieve_with_multimodal_quota) -- this guarantees multimodal content
reaches generation whenever it exists in the collection for a query,
rather than leaving that entirely to unfiltered RRF fusion against prose.

Beyond the standard RAGAS metrics, this also reports a hit rate: for each
question, did at least one table or figure chunk appear in the multimodal
pipeline's top-5 retrieved chunks? A question with a multimodal hit but low
faithfulness is a generation problem; a question with no hit at all is a
retrieval problem.

Usage:
  python -m scripts.run_multimodal_comparison
  python -m scripts.run_multimodal_comparison --concurrency 4
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from app.eval.golden_dataset import load_corpus_texts
from app.eval.ragas_dataset import RagasItem, load_ragas_dataset
from app.eval.ragas_runner import (
    HYBRID_MODEL,
    HYBRID_STRATEGY,
    PipelineResult,
    build_metrics,
    score_all,
)
from app.generation.generator import generate_answer
from app.ingestion.indexer import (
    collection_name_for,
    multimodal_collection_name,
)
from app.retrieval.searcher import (
    RetrievedChunk,
    retrieve_reranked,
    retrieve_with_multimodal_quota,
)

METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]
CATEGORIES = ["table_only", "diagram_only", "hybrid"]

# Same truncation as scripts/run_comparison_eval.py: RAGAS's faithfulness
# NLI-statement extraction can exceed instructor's inferred max_tokens on
# long answers/contexts (table KV text especially), raising
# IncompleteOutputException. Applied identically to both arms.
_SCORE_MAX_ANSWER_CHARS = 1_500
_SCORE_MAX_CONTEXT_CHARS = 1_000


def _for_scoring(prs: list[PipelineResult]) -> list[PipelineResult]:
    return [
        PipelineResult(
            question=pr.question,
            answer=pr.answer[:_SCORE_MAX_ANSWER_CHARS],
            contexts=[c[:_SCORE_MAX_CONTEXT_CHARS] for c in pr.contexts],
            reference=pr.reference,
            category=pr.category,
            source_docs=pr.source_docs,
        )
        for pr in prs
    ]


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    n = len(s)
    idx = p / 100 * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def _text_collection() -> str:
    return collection_name_for(HYBRID_STRATEGY, HYBRID_MODEL, hybrid=True)


def _multimodal_collection() -> str:
    return multimodal_collection_name(HYBRID_MODEL, hybrid=True)


def multimodal_hit(chunks: list[RetrievedChunk]) -> bool:
    """True if a table or figure chunk appears among the retrieved chunks."""
    return any(c.is_table or c.is_figure for c in chunks)


def run_text_baseline(item: RagasItem, top_k: int = 5) -> tuple[PipelineResult, float, int]:
    """retrieve_reranked against the text-only collection -> generate_answer."""
    start = time.time()
    chunks = retrieve_reranked(
        query=item.question,
        top_k=top_k,
        collection_name=_text_collection(),
        embedding_model=HYBRID_MODEL,
    )
    gen = generate_answer(question=item.question, chunks=chunks)
    latency_ms = int((time.time() - start) * 1000)
    return (
        PipelineResult(
            question=item.question,
            answer=gen.answer,
            contexts=[c.text for c in chunks],
            reference=item.reference_answer,
            category=item.category,
            source_docs=item.source_docs,
        ),
        gen.cost_usd,
        latency_ms,
    )


def run_multimodal(
    item: RagasItem, top_k: int = 5, multimodal_slots: int = 2
) -> tuple[PipelineResult, float, int, bool]:
    """retrieve_with_multimodal_quota against the multimodal collection ->
    generate_answer.

    Returns (result, cost_usd, latency_ms, multimodal_hit). Unlike
    run_text_baseline, retrieval here reserves multimodal_slots of top_k
    for table/figure chunks via filtered search rather than a single
    unfiltered reranked search.
    """
    start = time.time()
    chunks = retrieve_with_multimodal_quota(
        query=item.question,
        top_k=top_k,
        multimodal_slots=multimodal_slots,
        collection_name=_multimodal_collection(),
        embedding_model=HYBRID_MODEL,
    )
    gen = generate_answer(question=item.question, chunks=chunks)
    latency_ms = int((time.time() - start) * 1000)
    return (
        PipelineResult(
            question=item.question,
            answer=gen.answer,
            contexts=[c.text for c in chunks],
            reference=item.reference_answer,
            category=item.category,
            source_docs=item.source_docs,
        ),
        gen.cost_usd,
        latency_ms,
        multimodal_hit(chunks),
    )


def _aggregate(records: list[dict], costs: list[float], latencies: list[int]) -> dict:
    agg = {m: sum(r[m] for r in records) / len(records) for m in METRICS}
    agg["avg_cost_usd"] = sum(costs) / len(costs)
    agg["p50_latency_ms"] = round(
        _percentile([float(lat) for lat in latencies], 50)
    )
    agg["p95_latency_ms"] = round(
        _percentile([float(lat) for lat in latencies], 95)
    )
    return agg


def _hit_rate_breakdown(records: list[dict]) -> dict:
    by_category: dict[str, list[bool]] = defaultdict(list)
    for r in records:
        by_category[r["category"]].append(r["multimodal_hit"])
    overall = [r["multimodal_hit"] for r in records]
    breakdown = {
        cat: sum(hits) / len(hits) if hits else 0.0
        for cat, hits in by_category.items()
    }
    breakdown["overall"] = sum(overall) / len(overall) if overall else 0.0
    return breakdown


def build_output(
    pipeline: str,
    records: list[dict],
    costs: list[float],
    latencies: list[int],
    extra: dict | None = None,
) -> dict:
    out: dict = {
        "pipeline": pipeline,
        "comparison_set_size": len(records),
        "aggregate": _aggregate(records, costs, latencies),
        "per_question": records,
    }
    if extra:
        out.update(extra)
    return out


def print_comparison(text_agg: dict, multimodal_agg: dict, hit_rates: dict) -> None:
    rows = [
        ("faithfulness", "faithfulness", "score"),
        ("answer_relevancy", "answer_relevancy", "score"),
        ("context_precision", "context_precision", "score"),
        ("context_recall", "context_recall", "score"),
        ("avg_cost_usd", "avg_cost_usd", "cost"),
        ("p50_latency_ms", "p50_latency_ms", "ms"),
        ("p95_latency_ms", "p95_latency_ms", "ms"),
    ]

    def fmt_val(v: float, kind: str) -> str:
        if kind == "score":
            return f"{v:.4f}"
        if kind == "cost":
            return f"${v:.5f}"
        return f"{int(v)}ms"

    def fmt_delta(d: float, kind: str) -> str:
        sign = "+" if d >= 0 else ""
        if kind == "score":
            return f"{sign}{d:.4f}"
        if kind == "cost":
            return f"{sign}${d:.5f}"
        return f"{sign}{int(d)}ms"

    header = f"{'Metric':<24}{'Text Baseline':>15}{'Multimodal':>14}{'Delta':>11}"
    print("\n" + header)
    print("-" * len(header))
    for label, key, kind in rows:
        t, m = text_agg[key], multimodal_agg[key]
        print(
            f"{label:<24}{fmt_val(t, kind):>15}{fmt_val(m, kind):>14}"
            f"{fmt_delta(m - t, kind):>11}"
        )

    print("\nMultimodal chunk hit rate (table/figure chunk in top-5):")
    for cat in CATEGORIES:
        if cat in hit_rates:
            print(f"  {cat:<16}: {hit_rates[cat]:.1%}")
    print(f"  {'overall':<16}: {hit_rates['overall']:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Text-only vs multimodal (table/figure) RAG comparison."
    )
    parser.add_argument("--dataset", default="eval/multimodal_dataset.yaml")
    parser.add_argument(
        "--text-output", default="eval/results/text_baseline_comparison.json"
    )
    parser.add_argument(
        "--multimodal-output", default="eval/results/multimodal_comparison.json"
    )
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument(
        "--multimodal-slots",
        type=int,
        default=2,
        help="Top-k slots reserved for table/figure chunks in the "
        "multimodal arm (default: 2 of 5).",
    )
    args = parser.parse_args()

    items = load_ragas_dataset(args.dataset, load_corpus_texts())
    print(f"Comparison set: {len(items)} questions")
    for cat in CATEGORIES:
        n = sum(1 for i in items if i.category == cat)
        print(f"  {n} {cat}")

    # --- Text-only baseline pipeline ---
    print("\nRunning text-only baseline pipeline...")
    text_prs: list[PipelineResult] = []
    text_costs: list[float] = []
    text_latencies: list[int] = []
    for idx, item in enumerate(items, 1):
        print(f"  [{idx}/{len(items)}] {item.question[:72]}")
        pr, cost, lat = run_text_baseline(item)
        text_prs.append(pr)
        text_costs.append(cost)
        text_latencies.append(lat)

    # --- Multimodal pipeline ---
    print("\nRunning multimodal pipeline...")
    mm_prs: list[PipelineResult] = []
    mm_costs: list[float] = []
    mm_latencies: list[int] = []
    mm_hits: list[bool] = []
    for idx, item in enumerate(items, 1):
        print(f"  [{idx}/{len(items)}] {item.question[:72]}")
        pr, cost, lat, hit = run_multimodal(
            item, multimodal_slots=args.multimodal_slots
        )
        mm_prs.append(pr)
        mm_costs.append(cost)
        mm_latencies.append(lat)
        mm_hits.append(hit)

    # --- RAGAS scoring ---
    print("\nScoring with RAGAS (this fires LLM calls for each metric)...")
    metrics = build_metrics()
    text_scores = score_all(
        _for_scoring(text_prs), metrics, max_concurrency=args.concurrency
    )
    mm_scores = score_all(
        _for_scoring(mm_prs), metrics, max_concurrency=args.concurrency
    )

    # --- Assemble records ---
    text_records: list[dict] = []
    for pr, score, cost, lat in zip(
        text_prs, text_scores, text_costs, text_latencies, strict=True
    ):
        rec = asdict(pr)
        rec.update(score)
        rec["cost_usd"] = cost
        rec["latency_ms"] = lat
        text_records.append(rec)

    mm_records: list[dict] = []
    for pr, score, cost, lat, hit in zip(
        mm_prs, mm_scores, mm_costs, mm_latencies, mm_hits, strict=True
    ):
        rec = asdict(pr)
        rec.update(score)
        rec["cost_usd"] = cost
        rec["latency_ms"] = lat
        rec["multimodal_hit"] = hit
        mm_records.append(rec)

    hit_rates = _hit_rate_breakdown(mm_records)

    # --- Save ---
    text_out = build_output("text_baseline", text_records, text_costs, text_latencies)
    mm_out = build_output(
        "multimodal",
        mm_records,
        mm_costs,
        mm_latencies,
        extra={"hit_rate": hit_rates},
    )

    for path_str, data in [
        (args.text_output, text_out),
        (args.multimodal_output, mm_out),
    ]:
        p = Path(path_str)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Wrote {p}")

    print_comparison(text_out["aggregate"], mm_out["aggregate"], hit_rates)


if __name__ == "__main__":
    main()
