"""Drives the live retrieval -> rerank -> generate pipeline per RAGAS
golden-set item and scores each result with ragas's modern, client-based
metric classes (ragas.metrics.collections) -- not the legacy
evaluate()/EvaluationDataset path, which only supports langchain-backed
LLMs/embeddings.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass

from openai import AsyncOpenAI
from ragas.embeddings.base import BaseRagasEmbedding, embedding_factory
from ragas.llms import llm_factory
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from structlog import get_logger

from app.config import get_settings
from app.eval.ragas_dataset import RagasItem
from app.generation.generator import generate_answer
from app.ingestion.indexer import collection_name_for
from app.retrieval.searcher import retrieve_reranked

log = get_logger(__name__)
settings = get_settings()

# Mirrors main.py's hybrid retrieval config -- hybrid+rerank is currently
# only ingested for this strategy/model combo (see scripts/ingest.py
# --hybrid).
HYBRID_STRATEGY = "recursive"
HYBRID_MODEL = "text-embedding-3-small"

# ragas.llms.llm_factory auto-infers max_tokens for a metric's structured-
# output call from the response schema, not from the input answer length -
# a claim-dense answer (many distinct statements for Faithfulness to judge)
# can overflow that inferred budget and raise IncompleteOutputException even
# after truncating the answer text. Setting an explicit ceiling here removes
# that failure mode for all but pathological cases.
SCORING_MAX_TOKENS = 4096


@dataclass
class PipelineResult:
    question: str
    answer: str
    contexts: list[str]
    reference: str
    category: str
    source_docs: list[str]


def run_pipeline(item: RagasItem, top_k: int = 5) -> PipelineResult:
    """Runs one question through the same hybrid+rerank path main.py's
    /query endpoint uses, capturing the actual chunk text passed into the
    generation prompt (not reconstructed after the fact)."""
    chunks = retrieve_reranked(
        query=item.question,
        top_k=top_k,
        collection_name=collection_name_for(
            HYBRID_STRATEGY, HYBRID_MODEL, hybrid=True
        ),
        embedding_model=HYBRID_MODEL,
    )
    result = generate_answer(question=item.question, chunks=chunks)
    return PipelineResult(
        question=item.question,
        answer=result.answer,
        contexts=[c.text for c in chunks],
        reference=item.reference_answer,
        category=item.category,
        source_docs=item.source_docs,
    )


def run_all(items: list[RagasItem]) -> list[PipelineResult]:
    results = []
    for i, item in enumerate(items, start=1):
        print(f"[{i}/{len(items)}] {item.question[:70]}")
        results.append(run_pipeline(item))
    return results


@dataclass
class RagasMetrics:
    faithfulness: Faithfulness
    answer_relevancy: AnswerRelevancy
    context_precision: ContextPrecision
    context_recall: ContextRecall


def build_metrics() -> RagasMetrics:
    client = AsyncOpenAI()
    llm: InstructorBaseRagasLLM = llm_factory(
        settings.llm_model, client=client, max_tokens=SCORING_MAX_TOKENS
    )
    embeddings: BaseRagasEmbedding = embedding_factory(
        "openai", model=settings.embedding_model, client=client
    )
    return RagasMetrics(
        faithfulness=Faithfulness(llm=llm),
        answer_relevancy=AnswerRelevancy(llm=llm, embeddings=embeddings),
        context_precision=ContextPrecision(llm=llm),
        context_recall=ContextRecall(llm=llm),
    )


METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


async def _score_one(result: PipelineResult, metrics: RagasMetrics) -> dict:
    """
    Scores one question against all four metrics concurrently. A metric
    that raises - e.g. instructor's IncompleteOutputException, which fires
    when a claim-dense answer decomposes into more statements than the
    library's auto-inferred max_tokens budget allows - is recorded as None
    for that metric rather than propagating. One bad question must not
    take down every other question's already-computed scores.
    """
    metric_calls = {
        "faithfulness": metrics.faithfulness.ascore(
            user_input=result.question,
            response=result.answer,
            retrieved_contexts=result.contexts,
        ),
        "answer_relevancy": metrics.answer_relevancy.ascore(
            user_input=result.question, response=result.answer
        ),
        "context_precision": metrics.context_precision.ascore(
            user_input=result.question,
            reference=result.reference,
            retrieved_contexts=result.contexts,
        ),
        "context_recall": metrics.context_recall.ascore(
            user_input=result.question,
            retrieved_contexts=result.contexts,
            reference=result.reference,
        ),
    }
    outcomes = await asyncio.gather(
        *metric_calls.values(), return_exceptions=True
    )

    scores: dict[str, float | None] = {}
    for name, outcome in zip(metric_calls.keys(), outcomes, strict=True):
        if isinstance(outcome, BaseException):
            log.warning(
                "ragas_metric_failed",
                metric=name,
                question=result.question[:80],
                error=str(outcome),
            )
            scores[name] = None
        else:
            scores[name] = outcome.value
    return scores


async def _score_all_async(
    results: list[PipelineResult],
    metrics: RagasMetrics,
    max_concurrency: int,
) -> list[dict]:
    # Each question fires 4 metric calls at once; scoring every question in
    # the golden set concurrently (35 x 4 = 140+ simultaneous chat
    # completions, since faithfulness/answer_relevancy can themselves issue
    # more than one call) blows through OpenAI's per-minute token limit in
    # one burst. A semaphore caps how many *questions* are in flight at
    # once, which bounds peak concurrent requests to roughly
    # max_concurrency * 4.
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _bounded(result: PipelineResult) -> dict:
        async with semaphore:
            try:
                return await _score_one(result, metrics)
            except Exception as exc:
                # Last-resort guard for failures outside the per-metric
                # gather in _score_one (e.g. a bug in scoring setup) - keeps
                # this question's slot in the batch instead of losing every
                # other question's scores to one unhandled exception.
                log.warning(
                    "ragas_scoring_failed",
                    question=result.question[:80],
                    error=str(exc),
                )
                return dict.fromkeys(METRIC_NAMES)

    return await asyncio.gather(*(_bounded(r) for r in results))


def score_all(
    results: list[PipelineResult],
    metrics: RagasMetrics,
    max_concurrency: int = 4,
) -> list[dict]:
    return asyncio.run(_score_all_async(results, metrics, max_concurrency))


def to_per_question_records(
    results: list[PipelineResult], scores: list[dict]
) -> list[dict]:
    records = []
    for result, score in zip(results, scores, strict=True):
        record = asdict(result)
        record.update(score)
        records.append(record)
    return records
