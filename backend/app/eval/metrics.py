"""Precision@k / recall@k / reciprocal-rank scoring for chunking eval."""

from dataclasses import dataclass

from app.eval.golden_dataset import GoldenQuery
from app.eval.matcher import MATCH_THRESHOLD, is_relevant
from app.retrieval.searcher import RetrievedChunk


@dataclass
class QueryScore:
    query: str
    precision: float
    recall: float
    reciprocal_rank: float
    matched_items: int
    total_items: int


@dataclass
class AggregateScore:
    precision: float
    recall: float
    mrr: float
    num_queries: int


def score_query(
    golden: GoldenQuery,
    retrieved: list[RetrievedChunk],
    k: int,
    threshold: int = MATCH_THRESHOLD,
) -> QueryScore:
    """Score a single query's retrieved chunks against its golden items."""
    top = retrieved[:k]
    total_items = len(golden.items)

    matched_item_idx: set[int] = set()
    relevant_chunk_count = 0
    first_relevant_rank: int | None = None

    for rank, chunk in enumerate(top, start=1):
        chunk_is_relevant = False
        for idx, item in enumerate(golden.items):
            if is_relevant(item, chunk, threshold):
                chunk_is_relevant = True
                matched_item_idx.add(idx)
        if chunk_is_relevant:
            relevant_chunk_count += 1
            if first_relevant_rank is None:
                first_relevant_rank = rank

    precision = relevant_chunk_count / len(top) if top else 0.0
    recall = len(matched_item_idx) / total_items if total_items else 0.0
    reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank else 0.0

    return QueryScore(
        query=golden.query,
        precision=precision,
        recall=recall,
        reciprocal_rank=reciprocal_rank,
        matched_items=len(matched_item_idx),
        total_items=total_items,
    )


def aggregate(scores: list[QueryScore]) -> AggregateScore:
    """Aggregate a list of QueryScore into an overall score."""
    n = len(scores)
    if n == 0:
        return AggregateScore(0.0, 0.0, 0.0, 0)
    return AggregateScore(
        precision=sum(s.precision for s in scores) / n,
        recall=sum(s.recall for s in scores) / n,
        mrr=sum(s.reciprocal_rank for s in scores) / n,
        num_queries=n,
    )
