# scripts/run_graph_comparison_eval.py
"""
Compare naive RAG, agentic RAG, and graph RAG on multi-hop questions
(the multi_doc_synthesis category) - the case Neo4j's entity-relationship
traversal is specifically meant to help with, since these questions need
facts synthesized across >=2 documents that one-shot vector search over a
single collection may not surface together.

All three pipelines use the same generation model and the same RAGAS
metrics; the only variable is retrieval. Mirrors scripts/run_comparison_eval.py's
naive-vs-agentic comparison, extended with a third graph-based pipeline.

Usage:
  python -m scripts.run_graph_comparison_eval
  python -m scripts.run_graph_comparison_eval --concurrency 4
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from app.agent.loop import run_agent_loop
from app.config import get_settings
from app.eval.golden_dataset import load_corpus_texts
from app.eval.ragas_dataset import RagasItem, load_ragas_dataset
from app.eval.ragas_runner import (
    HYBRID_MODEL,
    HYBRID_STRATEGY,
    PipelineResult,
    build_metrics,
    score_all,
)
from app.generation.generator import generate_answer, generate_partial_answer
from app.graph.client import get_neo4j_driver
from app.graph.graph_searcher import retrieve_graph
from app.ingestion.indexer import collection_name_for
from app.retrieval.reranker import rerank as _rerank
from app.retrieval.searcher import retrieve_reranked

METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]

# Mirrors run_comparison_eval.py's truncation: RAGAS faithfulness scoring
# can exceed its inferred max_tokens budget on long answers/contexts.
# Applied identically to all three pipelines so the comparison stays fair.
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


def _collection() -> str:
    return collection_name_for(HYBRID_STRATEGY, HYBRID_MODEL, hybrid=True)


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    n = len(s)
    idx = p / 100 * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def build_multihop_set(all_items: list[RagasItem]) -> list[RagasItem]:
    """Multi-hop = multi_doc_synthesis: questions needing facts synthesized
    across >=2 documents, enforced by app/eval/ragas_dataset.py at load
    time."""
    return [item for item in all_items if item.category == "multi_doc_synthesis"]


def run_naive(item: RagasItem) -> tuple[PipelineResult, float, int]:
    """One-shot retrieve → generate. Returns (result, cost_usd, latency_ms)."""
    start = time.time()
    chunks = retrieve_reranked(
        query=item.question,
        top_k=5,
        collection_name=_collection(),
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


def run_agentic(
    item: RagasItem,
    top_k: int = 5,
) -> tuple[PipelineResult, float, int, int, str]:
    """Iterative loop → finalize → generate. Returns (result, cost_usd, latency_ms, iterations, terminated_by)."""
    start = time.time()
    state = run_agent_loop(
        question=item.question,
        top_k=top_k,
        embedding_model=HYBRID_MODEL,
        collection_name=_collection(),
    )
    final_chunks = _rerank(item.question, state.accumulated_chunks, top_k)
    if state.loop_terminated_by == "sufficiency_reached":
        gen = generate_answer(question=item.question, chunks=final_chunks)
    else:
        missing = (
            state.sufficiency_history[-1].missing_aspects
            if state.sufficiency_history
            else []
        )
        gen = generate_partial_answer(
            question=item.question,
            chunks=final_chunks,
            missing_aspects=missing,
        )
    latency_ms = int((time.time() - start) * 1000)
    cost_usd = state.loop_cost + gen.cost_usd
    return (
        PipelineResult(
            question=item.question,
            answer=gen.answer,
            contexts=[c.text for c in final_chunks],
            reference=item.reference_answer,
            category=item.category,
            source_docs=item.source_docs,
        ),
        cost_usd,
        latency_ms,
        state.iteration,
        state.loop_terminated_by,
    )


def run_graph(
    item: RagasItem, driver, top_k: int = 5
) -> tuple[PipelineResult, float, int]:
    """Graph vector search + 1-hop shared-entity expansion (reranked) →
    generate. Returns (result, cost_usd, latency_ms)."""
    start = time.time()
    chunks = retrieve_graph(
        item.question, top_k=top_k, driver=driver, rerank=True
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


def _aggregate(
    records: list[dict], costs: list[float], latencies: list[int]
) -> dict:
    # app/eval/ragas_runner.py records a metric as None (instead of raising)
    # when that question's scoring call fails - average only over the
    # questions that scored successfully, and surface how many didn't.
    agg: dict = {}
    for m in METRICS:
        scored = [r[m] for r in records if r[m] is not None]
        agg[m] = sum(scored) / len(scored) if scored else 0.0
        failed = len(records) - len(scored)
        if failed:
            agg[f"{m}_failed"] = failed
    agg["avg_cost_usd"] = sum(costs) / len(costs)
    latencies_f = [float(lat) for lat in latencies]
    agg["p50_latency_ms"] = round(_percentile(latencies_f, 50))
    agg["p95_latency_ms"] = round(_percentile(latencies_f, 95))
    return agg


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


def print_comparison(
    naive_agg: dict, agentic_agg: dict, graph_agg: dict, iter_dist: dict
) -> None:
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

    header = (
        f"{'Metric':<20}{'Naive':>10}{'Agentic':>10}{'Graph':>10}"
        f"{'Δ Agentic':>12}{'Δ Graph':>10}"
    )
    print("\n" + header)
    print("-" * len(header))
    for label, key, kind in rows:
        n, a, g = naive_agg[key], agentic_agg[key], graph_agg[key]
        print(
            f"{label:<20}{fmt_val(n, kind):>10}{fmt_val(a, kind):>10}"
            f"{fmt_val(g, kind):>10}{fmt_delta(a - n, kind):>12}"
            f"{fmt_delta(g - n, kind):>10}"
        )

    failures = [
        f"  {pipeline}/{m}: {agg[f'{m}_failed']} question(s) excluded"
        for pipeline, agg in [
            ("naive", naive_agg),
            ("agentic", agentic_agg),
            ("graph", graph_agg),
        ]
        for m in METRICS
        if f"{m}_failed" in agg
    ]
    if failures:
        print("\nScoring failures (excluded from the averages above):")
        for line in failures:
            print(line)

    print("\nAgentic iteration distribution:")
    dist_copy = dict(iter_dist)
    cap_count = dist_copy.pop("cap_reached", 0)
    for k in sorted(dist_copy, key=int):
        label = (
            f"iteration {k} (sufficient immediately)"
            if k == "1"
            else f"iteration {k}"
        )
        print(f"  {label}: {dist_copy[k]} queries")
    if cap_count:
        print(f"  cap reached (5 iterations): {cap_count} queries")
    print(f"  total: {sum(iter_dist.values())} queries")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Naive vs agentic vs graph RAG comparison on multi-hop questions."
    )
    parser.add_argument("--golden-dataset", default="eval/ragas_dataset.yaml")
    parser.add_argument(
        "--naive-output", default="eval/results/naive_rag_multihop.json"
    )
    parser.add_argument(
        "--agentic-output", default="eval/results/agentic_rag_multihop.json"
    )
    parser.add_argument(
        "--graph-output", default="eval/results/graph_rag_multihop.json"
    )
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    settings = get_settings()
    all_items = load_ragas_dataset(args.golden_dataset, load_corpus_texts())
    comparison = build_multihop_set(all_items)
    print(f"Multi-hop (multi_doc_synthesis) comparison set: {len(comparison)} questions")
    if not comparison:
        print("No multi_doc_synthesis questions found - nothing to compare.")
        return

    driver = get_neo4j_driver()
    graph_records, _, _ = driver.execute_query(
        "MATCH (c:Chunk) RETURN count(c) AS count",
        database_=settings.neo4j_database,
    )
    if graph_records[0]["count"] == 0:
        print(
            "No Chunk nodes found in Neo4j. Run: python -m scripts.ingest_graph"
        )
        return

    # --- Naive pipeline ---
    print("\nRunning naive RAG pipeline...")
    naive_prs: list[PipelineResult] = []
    naive_costs: list[float] = []
    naive_latencies: list[int] = []
    for idx, item in enumerate(comparison, 1):
        print(f"  [{idx}/{len(comparison)}] {item.question[:72]}")
        pr, cost, lat = run_naive(item)
        naive_prs.append(pr)
        naive_costs.append(cost)
        naive_latencies.append(lat)

    # --- Agentic pipeline ---
    print("\nRunning agentic RAG pipeline...")
    agentic_prs: list[PipelineResult] = []
    agentic_costs: list[float] = []
    agentic_latencies: list[int] = []
    agentic_meta: list[tuple[int, str]] = []
    for idx, item in enumerate(comparison, 1):
        print(f"  [{idx}/{len(comparison)}] {item.question[:72]}")
        pr, cost, lat, iters, term = run_agentic(item)
        agentic_prs.append(pr)
        agentic_costs.append(cost)
        agentic_latencies.append(lat)
        agentic_meta.append((iters, term))

    # --- Graph pipeline ---
    print("\nRunning graph RAG pipeline...")
    graph_prs: list[PipelineResult] = []
    graph_costs: list[float] = []
    graph_latencies: list[int] = []
    for idx, item in enumerate(comparison, 1):
        print(f"  [{idx}/{len(comparison)}] {item.question[:72]}")
        pr, cost, lat = run_graph(item, driver)
        graph_prs.append(pr)
        graph_costs.append(cost)
        graph_latencies.append(lat)

    # --- RAGAS scoring ---
    print("\nScoring with RAGAS (this fires LLM calls for each metric)...")
    metrics = build_metrics()
    naive_scores = score_all(
        _for_scoring(naive_prs), metrics, max_concurrency=args.concurrency
    )
    agentic_scores = score_all(
        _for_scoring(agentic_prs), metrics, max_concurrency=args.concurrency
    )
    graph_scores = score_all(
        _for_scoring(graph_prs), metrics, max_concurrency=args.concurrency
    )

    # --- Assemble records ---
    def _records(prs, scores, costs, latencies):
        out = []
        for pr, score, cost, lat in zip(
            prs, scores, costs, latencies, strict=True
        ):
            rec = asdict(pr)
            rec.update(score)
            rec["cost_usd"] = cost
            rec["latency_ms"] = lat
            out.append(rec)
        return out

    naive_records = _records(naive_prs, naive_scores, naive_costs, naive_latencies)
    graph_records_out = _records(
        graph_prs, graph_scores, graph_costs, graph_latencies
    )

    agentic_records: list[dict] = []
    iter_dist: Counter[str] = Counter()
    for pr, score, cost, lat, (iters, term) in zip(
        agentic_prs,
        agentic_scores,
        agentic_costs,
        agentic_latencies,
        agentic_meta,
        strict=True,
    ):
        rec = asdict(pr)
        rec.update(score)
        rec["cost_usd"] = cost
        rec["latency_ms"] = lat
        rec["iterations_used"] = iters
        rec["loop_terminated_by"] = term
        agentic_records.append(rec)
        iter_dist["cap_reached" if term == "cap_reached" else str(iters)] += 1

    # --- Save ---
    naive_out = build_output(
        "naive_rag", naive_records, naive_costs, naive_latencies
    )
    agentic_out = build_output(
        "agentic_rag",
        agentic_records,
        agentic_costs,
        agentic_latencies,
        extra={"iteration_distribution": dict(iter_dist)},
    )
    graph_out = build_output(
        "graph_rag", graph_records_out, graph_costs, graph_latencies
    )

    for path_str, data in [
        (args.naive_output, naive_out),
        (args.agentic_output, agentic_out),
        (args.graph_output, graph_out),
    ]:
        p = Path(path_str)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Wrote {p}")

    print_comparison(
        naive_out["aggregate"],
        agentic_out["aggregate"],
        graph_out["aggregate"],
        dict(iter_dist),
    )


if __name__ == "__main__":
    main()
