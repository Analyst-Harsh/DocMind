from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.agent.loop import run_agent_loop
from app.caching.cache import get_semantic_cache
from app.caching.schema import CachedResponse, CacheLookupResult
from app.config import get_settings
from app.generation.generator import generate_answer, generate_partial_answer
from app.ingestion.embedder import embed_query
from app.ingestion.indexer import collection_name_for
from app.retrieval.reranker import rerank
from app.retrieval.searcher import RetrievedChunk
from app.tracing.spans import flush_traces, new_trace_id, root_span, traced_span

router = APIRouter(prefix="/agent", tags=["agent"])
settings = get_settings()

_HYBRID_STRATEGY = "recursive"
_HYBRID_MODEL = "text-embedding-3-small"


def _finalize_chunks(
    question: str, chunks: list[RetrievedChunk], top_k: int
) -> list[RetrievedChunk]:
    if not settings.use_reranker:
        return chunks
    with traced_span("final-rerank") as span:
        reranked = rerank(question, chunks, top_k)
        span.update(output={"num_chunks": len(reranked)})
    return reranked


class AgentQueryRequest(BaseModel):
    question: str
    top_k: int = 5


class AgentQueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    cost_usd: float
    latency_ms: int
    trace_id: str
    cache_hit: bool
    iterations_used: int
    loop_terminated_by: str


@router.post("/query", response_model=AgentQueryResponse)
def agent_query(request: AgentQueryRequest):
    trace_id = new_trace_id()
    start_time = time.time()
    cache = get_semantic_cache()
    retrieval_mode = "hybrid_rerank" if settings.use_reranker else "hybrid"

    with root_span(
        "agentic-query",
        trace_id,
        input={"question": request.question, "top_k": request.top_k},
    ) as root:
        query_vector = embed_query(request.question, model=_HYBRID_MODEL)

        with traced_span("cache-check") as cache_span:
            lookup = (
                cache.check(query_vector, retrieval_mode, _HYBRID_MODEL)
                if settings.enable_semantic_cache
                else CacheLookupResult(hit=None, best_similarity=0.0)
            )
            cache_span.update(
                output={
                    "cache_hit": lookup.hit is not None,
                    "similarity": lookup.best_similarity,
                }
            )

        if lookup.hit:
            answer = lookup.hit.response.answer
            sources = lookup.hit.response.sources
            cost_usd = 0.0
            iterations_used = 0
            loop_terminated_by = "cache_hit"
            cache_hit = True
        else:
            collection = collection_name_for(
                _HYBRID_STRATEGY, _HYBRID_MODEL, hybrid=True
            )

            with traced_span("agent-loop") as loop_span:
                state = run_agent_loop(
                    question=request.question,
                    top_k=request.top_k,
                    embedding_model=_HYBRID_MODEL,
                    collection_name=collection,
                )

                if not state.accumulated_chunks:
                    root.update(output={"error": "no_chunks_found"})
                    flush_traces()
                    raise HTTPException(
                        status_code=404, detail="No relevant documents found."
                    )

                state.accumulated_chunks = _finalize_chunks(
                    request.question, state.accumulated_chunks, request.top_k
                )
                loop_span.update(output={
                    "total_iterations": state.iteration,
                    "termination_reason": state.loop_terminated_by,
                    "total_unique_chunks": len(state.accumulated_chunks),
                })

            last_check = state.sufficiency_history[-1] if state.sufficiency_history else None
            missing_aspects = last_check.missing_aspects if (last_check and last_check.missing_aspects) else []

            if state.loop_terminated_by == "sufficiency_reached":
                result = generate_answer(
                    question=request.question,
                    chunks=state.accumulated_chunks,
                )
            else:
                result = generate_partial_answer(
                    question=request.question,
                    chunks=state.accumulated_chunks,
                    missing_aspects=missing_aspects,
                )

            answer = result.answer
            cost_usd = result.cost_usd + state.loop_cost
            iterations_used = state.iteration
            loop_terminated_by = state.loop_terminated_by
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
                for c in state.accumulated_chunks
            ]

            if settings.enable_semantic_cache:
                cache.write(
                    request.question,
                    query_vector,
                    CachedResponse(
                        answer=answer, sources=sources, cost_usd=cost_usd
                    ),
                    retrieval_mode,
                    _HYBRID_MODEL,
                )

        latency_ms = int((time.time() - start_time) * 1000)
        root.update(
            output={"answer": answer},
            metadata={
                "cost_usd": cost_usd,
                "latency_ms": latency_ms,
                "cache_hit": cache_hit,
                "iterations_used": iterations_used,
                "loop_terminated_by": loop_terminated_by,
            },
        )

    flush_traces()

    return AgentQueryResponse(
        answer=answer,
        sources=sources,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        trace_id=trace_id,
        cache_hit=cache_hit,
        iterations_used=iterations_used,
        loop_terminated_by=loop_terminated_by,
    )
