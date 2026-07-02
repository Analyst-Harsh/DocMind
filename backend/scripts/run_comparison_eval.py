# scripts/run_comparison_eval.py
"""
Compare naive RAG vs agentic RAG on the questions where one-shot retrieval is
known to struggle: all multi_doc_synthesis questions plus any factual_single_doc
question that scored context_recall <= 0.5 in the Week 3 baseline.

Both pipelines use the same Qdrant collection, embedding model, and generation
model. The only variable is whether retrieval is one-shot or iterative.

Usage:
  python -m scripts.run_comparison_eval
  python -m scripts.run_comparison_eval --baseline eval/results/ragas_baseline_1.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from app.agent.loop import run_agent_loop
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
from app.ingestion.indexer import collection_name_for
from app.retrieval.reranker import rerank as _rerank
from app.retrieval.searcher import retrieve_reranked

METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]
CONTEXT_RECALL_THRESHOLD = 0.5

# RAGAS faithfulness extracts statements from the answer and judges each against
# the joined contexts in a single LLM call. Instructor infers max_tokens from
# the NLIStatementOutput schema; if the answer is long (especially partial_answer
# with a Gaps section) or contexts are large, the response can exceed that budget
# and raise IncompleteOutputException. Truncating before scoring is applied
# identically to both pipelines so the comparison remains valid.
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


def build_comparison_set(
    all_items: list[RagasItem],
    baseline: dict,
    threshold: float = CONTEXT_RECALL_THRESHOLD,
) -> list[RagasItem]:
    """Multi_doc_synthesis + factual_single_doc where baseline context_recall <= threshold."""
    by_question = {r["question"]: r for r in baseline.get("per_question", [])}
    result = []
    for item in all_items:
        if item.category == "multi_doc_synthesis":
            result.append(item)
        elif item.category == "factual_single_doc":
            record = by_question.get(item.question)
            if record and record.get("context_recall", 1.0) <= threshold:
                result.append(item)
    return result


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
    # Mirror agent/router.py's _finalize_chunks: rerank all accumulated chunks
    # down to top_k before generation and scoring. Without this, agentic can
    # accumulate up to MAX_ITERATIONS * top_k chunks, making the faithfulness
    # scoring prompt too long for the LLM's max_tokens limit.
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


def _aggregate(
    records: list[dict], costs: list[float], latencies: list[int]
) -> dict:
    agg = {m: sum(r[m] for r in records) / len(records) for m in METRICS}
    agg["avg_cost_usd"] = sum(costs) / len(costs)
    agg["p50_latency_ms"] = round(
        _percentile([float(l) for l in latencies], 50)
    )
    agg["p95_latency_ms"] = round(
        _percentile([float(l) for l in latencies], 95)
    )
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
    naive_agg: dict, agentic_agg: dict, iter_dist: dict
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

    header = f"{'Metric':<24}{'Naive RAG':>13}{'Agentic RAG':>14}{'Delta':>11}"
    print("\n" + header)
    print("-" * len(header))
    for label, key, kind in rows:
        n, a = naive_agg[key], agentic_agg[key]
        print(
            f"{label:<24}{fmt_val(n, kind):>13}{fmt_val(a, kind):>14}"
            f"{fmt_delta(a - n, kind):>11}"
        )

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
        description="Naive RAG vs agentic RAG comparison on multi-hop/low-recall questions."
    )
    parser.add_argument("--golden-dataset", default="eval/ragas_dataset.yaml")
    parser.add_argument(
        "--baseline", default="eval/results/ragas_baseline_1.json"
    )
    parser.add_argument(
        "--naive-output", default="eval/results/naive_rag_comparison.json"
    )
    parser.add_argument(
        "--agentic-output", default="eval/results/agentic_rag_comparison.json"
    )
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline not found: {baseline_path}\n"
            "Run `python -m scripts.run_ragas_eval` first to generate it."
        )

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    all_items = load_ragas_dataset(args.golden_dataset, load_corpus_texts())
    comparison = build_comparison_set(all_items, baseline)

    n_multi = sum(1 for i in comparison if i.category == "multi_doc_synthesis")
    n_factual = sum(1 for i in comparison if i.category == "factual_single_doc")
    print(f"Comparison set: {len(comparison)} questions")
    print(f"  {n_multi} multi_doc_synthesis")
    print(
        f"  {n_factual} factual_single_doc (baseline context_recall <= {CONTEXT_RECALL_THRESHOLD})"
    )

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

    # --- RAGAS scoring ---
    print("\nScoring with RAGAS (this fires LLM calls for each metric)...")
    metrics = build_metrics()
    naive_scores = score_all(
        _for_scoring(naive_prs), metrics, max_concurrency=args.concurrency
    )
    agentic_scores = score_all(
        _for_scoring(agentic_prs), metrics, max_concurrency=args.concurrency
    )

    # --- Assemble records ---
    naive_records: list[dict] = []
    for pr, score, cost, lat in zip(
        naive_prs, naive_scores, naive_costs, naive_latencies, strict=True
    ):
        rec = asdict(pr)
        rec.update(score)
        rec["cost_usd"] = cost
        rec["latency_ms"] = lat
        naive_records.append(rec)

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

    for path_str, data in [
        (args.naive_output, naive_out),
        (args.agentic_output, agentic_out),
    ]:
        p = Path(path_str)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Wrote {p}")

    print_comparison(
        naive_out["aggregate"], agentic_out["aggregate"], dict(iter_dist)
    )


if __name__ == "__main__":
    main()
