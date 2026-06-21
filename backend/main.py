# app/main.py
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import get_settings
from app.generation.generator import generate_answer, stream_answer
from app.ingestion.indexer import collection_name_for
from app.retrieval.searcher import retrieve, retrieve_hybrid, retrieve_reranked
from app.tracing.spans import flush_traces, new_trace_id, root_span, traced_span

app = FastAPI(title="DocMind", version="0.1.0")
settings = get_settings()

# Hybrid (dense+BM25) retrieval is currently only ingested for this
# strategy/model combo (see scripts/ingest.py --hybrid).
HYBRID_STRATEGY = "recursive"
HYBRID_MODEL = "text-embedding-3-small"


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    hybrid: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    cost_usd: float
    latency_ms: int
    trace_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    trace_id = new_trace_id()

    start_time = time.time()

    with root_span(
        "docmind-query",
        trace_id,
        input={"question": request.question, "top_k": request.top_k},
    ) as root:
        with traced_span(
            "retrieval",
            input={"query": request.question},
        ) as retrieval_span:
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
                )
            else:
                chunks = retrieve(query=request.question, top_k=request.top_k)
            retrieval_span.update(
                output={
                    "num_chunks": len(chunks),
                    "top_score": chunks[0].score if chunks else 0,
                    "chunk_ids": [c.chunk_id for c in chunks],
                    "reranked": bool(request.hybrid and settings.use_reranker),
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

        latency_ms = int((time.time() - start_time) * 1000)

        root.update(
            output={"answer": result.answer},
            metadata={"cost_usd": result.cost_usd, "latency_ms": latency_ms},
        )

    flush_traces()

    sources = [
        {
            "chunk_id": c.chunk_id,
            "doc_title": c.doc_title,
            "score": c.score,
            "source_path": c.source_path,
        }
        for c in chunks
    ]

    return QueryResponse(
        answer=result.answer,
        sources=sources,
        cost_usd=result.cost_usd,
        latency_ms=latency_ms,
        trace_id=trace_id,
    )


@app.post("/query/stream")
def query_stream(request: QueryRequest):
    """Streaming endpoint for the frontend (Week 3)."""
    chunks = retrieve(query=request.question, top_k=request.top_k)
    if not chunks:
        raise HTTPException(
            status_code=404, detail="No relevant documents found"
        )

    return StreamingResponse(
        stream_answer(question=request.question, chunks=chunks),
        media_type="text/plain",
    )
