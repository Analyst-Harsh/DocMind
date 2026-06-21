# app/main.py
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import get_settings
from app.generation.generator import generate_answer, stream_answer
from app.generation.prompts import build_qa_prompt
from app.ingestion.indexer import collection_name_for
from app.retrieval.searcher import retrieve, retrieve_hybrid
from app.tracing.langfuse import get_langfuse

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
    langfuse = get_langfuse()
    trace_id = langfuse.create_trace_id()

    start_time = time.time()

    with langfuse.start_as_current_observation(
        trace_context={"trace_id": trace_id},
        name="docmind-query",
        as_type="span",
        input={"question": request.question, "top_k": request.top_k},
    ) as root_span:
        with root_span.start_as_current_observation(
            name="retrieval",
            as_type="span",
            input={"query": request.question},
        ) as retrieval_span:
            if request.hybrid:
                chunks = retrieve_hybrid(
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
                }
            )

        if not chunks:
            root_span.update(output={"error": "no_chunks_found"})
            langfuse.flush()
            raise HTTPException(
                status_code=404, detail="No relevant documents found"
            )

        # Generation: typed observation with model/usage/cost fields
        prompt = build_qa_prompt(request.question, chunks)
        with root_span.start_as_current_observation(
            name="answer-generation",
            as_type="generation",
            model=settings.llm_model,
            input=prompt,
        ) as generation:
            result = generate_answer(question=request.question, chunks=chunks)
            generation.update(
                output=result.answer,
                usage_details={
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                },
                # Langfuse can compute cost itself from its model price table,
                # or you can pass your own:
                cost_details={"total": result.cost_usd},
            )

        latency_ms = int((time.time() - start_time) * 1000)

        root_span.update(
            output={"answer": result.answer},
            metadata={"cost_usd": result.cost_usd, "latency_ms": latency_ms},
        )

    langfuse.flush()

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
