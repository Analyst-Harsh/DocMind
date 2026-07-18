# app/main.py
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agent.router import router as agent_router
from app.caching.cache import get_semantic_cache
from app.caching.schema import CacheLookupResult
from app.config import get_settings
from app.documents.router import router as documents_router
from app.ingestion.embedder import embed_query
from app.ingestion.indexer import (
    HYBRID_MODEL,
    HYBRID_STRATEGY,
    collection_name_for,
    get_qdrant_client,
)
from app.query.service import (
    NoRelevantChunksError,
    RepoNotIngestedError,
    run_query,
)
from app.repo_ingest.router import router as ingest_router
from app.repo_ingest.service import repo_collection_name
from app.retrieval.searcher import retrieve, retrieve_hybrid, retrieve_reranked
from app.streaming.pipeline import stream_query_pipeline
from app.tracing.spans import new_trace_id

app = FastAPI(title="DocMind", version="0.1.0")
app.include_router(agent_router)
app.include_router(documents_router)
app.include_router(ingest_router)
settings = get_settings()


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    hybrid: bool = True
    # When set, queries a repo ingested via POST /ingest/repo instead of
    # the fixed docs corpus. Repo collections are hybrid-only (see
    # app.repo_ingest.service.repo_collection_name), so setting repo
    # implies hybrid retrieval regardless of the hybrid flag above.
    repo: str | None = None


def _resolve_repo_collection_for_stream(repo: str) -> str:
    """POST /query/stream's own copy of the repo-resolution check --
    kept separate from app.query.service's version (which raises a
    domain exception for /query and the MCP tools) so streaming's
    behavior stays untouched; /query/stream is out of scope for the
    app.query.service extraction."""
    collection = repo_collection_name(repo, HYBRID_MODEL)
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        raise HTTPException(
            status_code=404,
            detail=f"Repo {repo!r} has not been ingested — POST /ingest/repo first.",
        )
    return collection


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
    try:
        result = run_query(
            question=request.question,
            top_k=request.top_k,
            hybrid=request.hybrid,
            repo=request.repo,
        )
    except RepoNotIngestedError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except NoRelevantChunksError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    return QueryResponse(
        answer=result.answer,
        sources=result.sources,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
        trace_id=result.trace_id,
        cache_hit=result.cache_hit,
    )


@app.post("/query/stream")
def query_stream(request: QueryRequest):
    """Streaming endpoint — full pipeline: embed → cache → retrieve → rerank → generate."""
    trace_id = new_trace_id()
    start_time = time.time()

    is_repo_query = request.repo is not None
    use_hybrid = request.hybrid or is_repo_query
    embedding_model = HYBRID_MODEL if use_hybrid else None
    resolved_model = embedding_model or settings.embedding_model
    retrieval_mode = (
        "dense"
        if not use_hybrid
        else ("hybrid_rerank" if settings.use_reranker else "hybrid")
    )
    cache_scope = request.repo or "docs"

    query_vector = embed_query(request.question, model=embedding_model)

    cache = get_semantic_cache()
    cache_lookup = (
        cache.check(query_vector, retrieval_mode, resolved_model, scope=cache_scope)
        if settings.enable_semantic_cache
        else CacheLookupResult(hit=None, best_similarity=0.0)
    )

    if cache_lookup.hit:
        chunks: list = []
        sources: list[dict] = cache_lookup.hit.response.sources
    else:
        if use_hybrid:
            retrieve_fn = (
                retrieve_reranked if settings.use_reranker else retrieve_hybrid
            )
            collection = (
                _resolve_repo_collection_for_stream(request.repo)
                if request.repo is not None
                else collection_name_for(
                    HYBRID_STRATEGY, HYBRID_MODEL, hybrid=True
                )
            )
            chunks = retrieve_fn(
                query=request.question,
                top_k=request.top_k,
                collection_name=collection,
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
            cache_scope=cache_scope,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
