# app/main.py
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agent.router import router as agent_router
from app.caching.cache import get_semantic_cache
from app.caching.schema import CachedResponse, CacheLookupResult
from app.config import get_settings
from app.generation.generator import generate_answer
from app.ingestion.embedder import embed_query
from app.ingestion.indexer import collection_name_for
from app.retrieval.searcher import retrieve, retrieve_hybrid, retrieve_reranked
from app.streaming.pipeline import stream_query_pipeline
from app.tracing.spans import flush_traces, new_trace_id, root_span, traced_span

app = FastAPI(title="DocMind", version="0.1.0")
app.include_router(agent_router)
settings = get_settings()

# Hybrid (dense+BM25) retrieval is currently only ingested for this
# strategy/model combo (see scripts/ingest.py --hybrid).
HYBRID_STRATEGY = "recursive"
HYBRID_MODEL = "text-embedding-3-small"


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    hybrid: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    cost_usd: float
    latency_ms: int
    trace_id: str
    cache_hit: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    trace_id = new_trace_id()

    start_time = time.time()
    cache = get_semantic_cache()

    embedding_model = HYBRID_MODEL if request.hybrid else None
    retrieval_mode = (
        "dense"
        if not request.hybrid
        else ("hybrid_rerank" if settings.use_reranker else "hybrid")
    )
    resolved_model = embedding_model or settings.embedding_model

    with root_span(
        "docmind-query",
        trace_id,
        input={"question": request.question, "top_k": request.top_k},
    ) as root:
        query_vector = embed_query(request.question, model=embedding_model)

        with traced_span("cache-check") as cache_span:
            lookup = (
                cache.check(query_vector, retrieval_mode, resolved_model)
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
                input={"query": request.question},
            ) as retrieval_span:
                if request.hybrid:
                    retrieve_fn = (
                        retrieve_reranked
                        if settings.use_reranker
                        else retrieve_hybrid
                    )
                    chunks = retrieve_fn(
                        query=request.question,
                        top_k=request.top_k,
                        collection_name=collection_name_for(
                            HYBRID_STRATEGY, HYBRID_MODEL, hybrid=True
                        ),
                        embedding_model=HYBRID_MODEL,
                        query_vector=query_vector,
                    )
                else:
                    chunks = retrieve(
                        query=request.question,
                        top_k=request.top_k,
                        query_vector=query_vector,
                    )
                retrieval_span.update(
                    output={
                        "num_chunks": len(chunks),
                        "top_score": chunks[0].score if chunks else 0,
                        "chunk_ids": [c.chunk_id for c in chunks],
                        "reranked": bool(
                            request.hybrid and settings.use_reranker
                        ),
                    }
                )

            if not chunks:
                root.update(output={"error": "no_chunks_found"})
                flush_traces()
                raise HTTPException(
                    status_code=404, detail="No relevant documents found"
                )

            # generate_answer owns its own "answer-generation" span (nests under
            # this retrieval span's parent automatically via OTEL context).
            result = generate_answer(question=request.question, chunks=chunks)
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
                    request.question,
                    query_vector,
                    CachedResponse(
                        answer=answer, sources=sources, cost_usd=cost_usd
                    ),
                    retrieval_mode,
                    resolved_model,
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

    return QueryResponse(
        answer=answer,
        sources=sources,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        trace_id=trace_id,
        cache_hit=cache_hit,
    )


@app.post("/query/stream")
def query_stream(request: QueryRequest):
    """Streaming endpoint — full pipeline: embed → cache → retrieve → rerank → generate."""
    trace_id = new_trace_id()
    start_time = time.time()

    embedding_model = HYBRID_MODEL if request.hybrid else None
    resolved_model = embedding_model or settings.embedding_model
    retrieval_mode = (
        "dense"
        if not request.hybrid
        else ("hybrid_rerank" if settings.use_reranker else "hybrid")
    )

    query_vector = embed_query(request.question, model=embedding_model)

    cache = get_semantic_cache()
    cache_lookup = (
        cache.check(query_vector, retrieval_mode, resolved_model)
        if settings.enable_semantic_cache
        else CacheLookupResult(hit=None, best_similarity=0.0)
    )

    if cache_lookup.hit:
        chunks: list = []
        sources: list[dict] = cache_lookup.hit.response.sources
    else:
        if request.hybrid:
            retrieve_fn = (
                retrieve_reranked if settings.use_reranker else retrieve_hybrid
            )
            chunks = retrieve_fn(
                query=request.question,
                top_k=request.top_k,
                collection_name=collection_name_for(
                    HYBRID_STRATEGY, HYBRID_MODEL, hybrid=True
                ),
                embedding_model=HYBRID_MODEL,
                query_vector=query_vector,
            )
        else:
            chunks = retrieve(
                query=request.question,
                top_k=request.top_k,
                query_vector=query_vector,
            )

        if not chunks:
            raise HTTPException(
                status_code=404, detail="No relevant documents found"
            )

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

    return StreamingResponse(
        stream_query_pipeline(
            question=request.question,
            chunks=chunks,
            cache_hit=cache_lookup.hit.response if cache_lookup.hit else None,
            trace_id=trace_id,
            start_time=start_time,
            query_vector=query_vector,
            sources=sources,
            resolved_model=resolved_model,
            retrieval_mode=retrieval_mode,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
