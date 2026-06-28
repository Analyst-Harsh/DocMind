import json
import time
from collections.abc import Iterator

from app.caching.cache import get_semantic_cache
from app.caching.schema import CachedResponse
from app.config import get_settings
from app.generation.generator import (
    COST_PER_INPUT_TOKEN,
    COST_PER_OUTPUT_TOKEN,
    client,
)
from app.generation.prompts import build_qa_prompt
from app.retrieval.searcher import RetrievedChunk
from app.tracing.spans import flush_traces, root_span, traced_span

settings = get_settings()


def stream_query_pipeline(
    question: str,
    chunks: list[RetrievedChunk],
    cache_hit: CachedResponse | None,
    trace_id: str,
    start_time: float,
    query_vector: list[float],
    sources: list[dict],
    resolved_model: str,
    retrieval_mode: str,
) -> Iterator[str]:
    """
    Yields SSE-formatted events for the full streaming pipeline.
    Retrieval and cache lookup must have already run in the endpoint caller.
    Event order: token(s) → done → metadata
    """
    collected: list[str] = []
    cost = 0.0
    latency_ms = 0

    with root_span(
        "docmind-stream", trace_id, input={"question": question}
    ) as root:
        if cache_hit is not None:
            yield f"event: token\ndata: {cache_hit.answer}\n\n"
            latency_ms = int((time.time() - start_time) * 1000)
            root.update(
                output={"answer": cache_hit.answer},
                metadata={
                    "cost_usd": 0.0,
                    "latency_ms": latency_ms,
                    "cache_hit": True,
                },
            )
        else:
            prompt = build_qa_prompt(question, chunks)

            with traced_span(
                "answer-generation",
                as_type="generation",
                model=settings.llm_model,
                input=prompt,
            ) as span:
                stream = client.chat.completions.create(
                    model=settings.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    stream=True,
                    stream_options={"include_usage": True},
                )

                usage = None
                for chunk in stream:
                    if not chunk.choices:
                        usage = chunk.usage
                        continue
                    delta = chunk.choices[0].delta.content
                    if delta:
                        collected.append(delta)
                        yield f"event: token\ndata: {delta}\n\n"

                if usage is not None:
                    cost = (
                        usage.prompt_tokens * COST_PER_INPUT_TOKEN
                        + usage.completion_tokens * COST_PER_OUTPUT_TOKEN
                    )
                    span.update(
                        output="".join(collected),
                        usage_details={
                            "prompt_tokens": usage.prompt_tokens,
                            "completion_tokens": usage.completion_tokens,
                        },
                        cost_details={"total": cost},
                    )

            if settings.enable_semantic_cache:
                cache = get_semantic_cache()
                cache.write(
                    question,
                    query_vector,
                    CachedResponse(
                        answer="".join(collected),
                        sources=sources,
                        cost_usd=cost,
                    ),
                    retrieval_mode,
                    resolved_model,
                )

            latency_ms = int((time.time() - start_time) * 1000)
            root.update(
                output={"answer": "".join(collected)},
                metadata={
                    "cost_usd": cost,
                    "latency_ms": latency_ms,
                    "cache_hit": False,
                },
            )

    flush_traces()

    metadata_payload = {
        "sources": sources,
        "cost_usd": cost,
        "latency_ms": latency_ms,
        "cache_hit": cache_hit is not None,
        "trace_id": trace_id,
    }

    yield "event: done\ndata: \n\n"
    yield f"event: metadata\ndata: {json.dumps(metadata_payload)}\n\n"
