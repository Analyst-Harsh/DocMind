"""Run the golden query set against one strategy's collection."""

from collections.abc import Callable
from dataclasses import dataclass

from qdrant_client import QdrantClient

from app.eval.golden_dataset import GoldenQuery
from app.eval.matcher import MATCH_THRESHOLD
from app.eval.metrics import AggregateScore, QueryScore, aggregate, score_query
from app.retrieval.searcher import RetrievedChunk


@dataclass
class StrategyReport:
    strategy: str
    aggregate: AggregateScore
    per_query: list[QueryScore]
    chunk_count: int = 0
    avg_tokens: float = 0.0


RetrieveFn = Callable[[str, int], list[RetrievedChunk]]


def run_strategy(
    strategy: str,
    retrieve_fn: RetrieveFn,
    golden_queries: list[GoldenQuery],
    k: int,
    threshold: int = MATCH_THRESHOLD,
) -> StrategyReport:
    scores = [
        score_query(golden, retrieve_fn(golden.query, k), k, threshold)
        for golden in golden_queries
    ]
    return StrategyReport(
        strategy=strategy,
        aggregate=aggregate(scores),
        per_query=scores,
    )


def collection_stats(
    client: QdrantClient, collection_name: str
) -> tuple[int, float]:
    """Return (chunk_count, avg_tokens) from a collection's point payloads."""
    count = client.get_collection(collection_name).points_count or 0
    if count == 0:
        return 0, 0.0

    token_counts: list[int] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        token_counts.extend(
            p.payload.get("token_count", 0) for p in points if p.payload
        )
        if offset is None:
            break

    avg = sum(token_counts) / len(token_counts) if token_counts else 0.0
    return count, avg
