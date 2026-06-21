from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import TYPE_CHECKING

from sentence_transformers import CrossEncoder

from app.tracing.spans import traced_span

if TYPE_CHECKING:
    from app.retrieval.searcher import RetrievedChunk

# bge-reranker-base: cross-encoder, scores (query, passage) pairs jointly
# rather than comparing two independently-computed embeddings - too slow
# to run over a whole corpus, but cheap over a small candidate pool.
RERANKER_MODEL = "BAAI/bge-reranker-base"


@lru_cache(maxsize=1)
def _load_reranker_model() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL)


def rerank(
    query: str, chunks: list[RetrievedChunk], top_k: int
) -> list[RetrievedChunk]:
    """
    Score each (query, chunk) pair jointly with a cross-encoder and return
    the top_k chunks sorted by that score descending. Overwrites each
    chunk's .score with the cross-encoder score - the candidate pool's
    original (e.g. RRF-fused) score is no longer meaningful once the
    reranker has re-scored every pair from scratch.
    """
    if not chunks:
        return []

    with traced_span(
        "rerank",
        as_type="retriever",
        input={"num_candidates": len(chunks), "model": RERANKER_MODEL},
    ) as span:
        model = _load_reranker_model()
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = model.predict(pairs)

        rescored = [
            replace(chunk, score=float(score))
            for chunk, score in zip(chunks, scores, strict=True)
        ]
        rescored.sort(key=lambda c: c.score, reverse=True)
        result = rescored[:top_k]

        span.update(
            output={
                "top_score": result[0].score if result else None,
                "num_results": len(result),
            }
        )

    return result
