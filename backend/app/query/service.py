# app/query/service.py
import time
from dataclasses import dataclass
from typing import Any

from app.caching.cache import get_semantic_cache
from app.caching.schema import CachedResponse, CacheLookupResult
from app.config import get_settings
from app.generation.generator import generate_answer
from app.ingestion.embedder import embed_query
from app.ingestion.indexer import (
    HYBRID_MODEL,
    HYBRID_STRATEGY,
    collection_name_for,
    get_qdrant_client,
)
from app.repo_ingest.service import repo_collection_name
from app.retrieval.searcher import retrieve, retrieve_hybrid, retrieve_reranked
from app.tracing.spans import flush_traces, new_trace_id, root_span, traced_span


class RepoNotIngestedError(Exception):
    def __init__(self, repo: str):
        self.repo = repo
        super().__init__(
            f"Repo {repo!r} has not been ingested — POST /ingest/repo first."
        )


class NoRelevantChunksError(Exception):
    def __init__(self):
        super().__init__("No relevant documents found")


@dataclass
class QueryResult:
    answer: str
    sources: list[dict[str, Any]]
    cost_usd: float
    latency_ms: int
    trace_id: str
    cache_hit: bool


def _resolve_repo_collection(repo: str) -> str:
    collection = repo_collection_name(repo, HYBRID_MODEL)
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        raise RepoNotIngestedError(repo)
    return collection


def run_query(
    question: str,
    top_k: int = 5,
    hybrid: bool = True,
    repo: str | None = None,
) -> QueryResult:
    trace_id = new_trace_id()

    start_time = time.time()
    cache = get_semantic_cache()
    settings = get_settings()

    is_repo_query = repo is not None
    use_hybrid = hybrid or is_repo_query
    embedding_model = HYBRID_MODEL if use_hybrid else None
    retrieval_mode = (
        "dense"
        if not use_hybrid
        else ("hybrid_rerank" if settings.use_reranker else "hybrid")
    )
    resolved_model = embedding_model or settings.embedding_model
    cache_scope = repo or "docs"

    with root_span(
        "docmind-query",
        trace_id,
        input={"question": question, "top_k": top_k},
    ) as root:
        query_vector = embed_query(question, model=embedding_model)

        with traced_span("cache-check") as cache_span:
            lookup = (
                cache.check(
                    query_vector,
                    retrieval_mode,
                    resolved_model,
                    scope=cache_scope,
                )
                if settings.enable_semantic_cache
                else CacheLookupResult(hit=None, best_similarity=0.0)
            )
            cache_span.update(
                output={
                    "cache_hit": lookup.hit is not None,
                    "similarity": lookup.best_similarity,
                    "matched_query": lookup.hit.query if lookup.hit else None,
                }
            )

        if lookup.hit:
            answer = lookup.hit.response.answer
            sources = lookup.hit.response.sources
            # The cached cost_usd is what the answer originally cost to
            # generate, not what this request cost -- a hit makes no LLM
            # call, so the true marginal cost here is 0.
            cost_usd = 0.0
            cache_hit = True
        else:
            with traced_span(
                "retrieval",
                input={"query": question},
            ) as retrieval_span:
                if use_hybrid:
                    retrieve_fn = (
                        retrieve_reranked
                        if settings.use_reranker
                        else retrieve_hybrid
                    )
                    collection = (
                        _resolve_repo_collection(repo)
                        if repo is not None
                        else collection_name_for(
                            HYBRID_STRATEGY, HYBRID_MODEL, hybrid=True
                        )
                    )
                    chunks = retrieve_fn(
                        query=question,
                        top_k=top_k,
                        collection_name=collection,
                        embedding_model=HYBRID_MODEL,
                        query_vector=query_vector,
                    )
                else:
                    chunks = retrieve(
                        query=question,
                        top_k=top_k,
                        query_vector=query_vector,
                    )
                retrieval_span.update(
                    output={
                        "num_chunks": len(chunks),
                        "top_score": chunks[0].score if chunks else 0,
                        "chunk_ids": [c.chunk_id for c in chunks],
                        "reranked": bool(use_hybrid and settings.use_reranker),
                    }
                )

            if not chunks:
                root.update(output={"error": "no_chunks_found"})
                flush_traces()
                raise NoRelevantChunksError()

            # generate_answer owns its own "answer-generation" span (nests
            # under this retrieval span's parent automatically via OTEL
            # context).
            result = generate_answer(question=question, chunks=chunks)
            answer = result.answer
            cost_usd = result.cost_usd
            cache_hit = False
            sources = [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "doc_title": c.doc_title,
                    "chunk_index": c.chunk_index,
                    "score": c.score,
                    "source_path": c.source_path,
                }
                for c in chunks
            ]

            if settings.enable_semantic_cache:
                cache.write(
                    question,
                    query_vector,
                    CachedResponse(
                        answer=answer, sources=sources, cost_usd=cost_usd
                    ),
                    retrieval_mode,
                    resolved_model,
                    scope=cache_scope,
                )

        latency_ms = int((time.time() - start_time) * 1000)

        root.update(
            output={"answer": answer},
            metadata={
                "cost_usd": cost_usd,
                "latency_ms": latency_ms,
                "cache_hit": cache_hit,
            },
        )

    flush_traces()

    return QueryResult(
        answer=answer,
        sources=sources,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        trace_id=trace_id,
        cache_hit=cache_hit,
    )
