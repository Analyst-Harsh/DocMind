from __future__ import annotations

from structlog import get_logger

from app.agent.reformulation import reformulate_query
from app.agent.state import AgentLoopState, SufficiencyResult
from app.agent.sufficiency import assess_sufficiency
from app.config import get_settings
from app.retrieval.searcher import (
    RetrievedChunk,
    retrieve_hybrid,
    retrieve_reranked,
)
from app.tracing.spans import traced_span

MAX_ITERATIONS = 5

settings = get_settings()
log = get_logger(__name__)


def _retrieve(
    query: str,
    top_k: int,
    embedding_model: str | None,
    collection_name: str | None,
) -> list[RetrievedChunk]:
    if settings.use_reranker:
        return retrieve_reranked(
            query=query,
            top_k=top_k,
            collection_name=collection_name,
            embedding_model=embedding_model,
        )
    return retrieve_hybrid(
        query=query,
        top_k=top_k,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )


def run_agent_loop(
    question: str,
    top_k: int = 5,
    embedding_model: str | None = None,
    collection_name: str | None = None,
) -> AgentLoopState:
    state = AgentLoopState(
        original_question=question,
        current_query=question,
    )

    while state.iteration < MAX_ITERATIONS:
        state.iteration += 1

        with traced_span(
            f"iteration-{state.iteration}"
        ) as iter_span:
            if state.iteration > 1:
                last = state.sufficiency_history[-1]
                state.current_query, reform_cost = reformulate_query(
                    state.original_question,
                    last.missing_aspects,
                )
                state.loop_cost += reform_cost

            with traced_span("retrieve") as retrieve_span:
                new_chunks = _retrieve(
                    state.current_query, top_k, embedding_model, collection_name
                )
                retrieve_span.update(output={"num_chunks": len(new_chunks)})
                state.query_history.append(state.current_query)

            seen_ids = {c.chunk_id for c in state.accumulated_chunks}
            new_unique = [c for c in new_chunks if c.chunk_id not in seen_ids]
            state.accumulated_chunks.extend(new_unique)

            result: SufficiencyResult = assess_sufficiency(
                state.original_question, state.accumulated_chunks
            )
            state.loop_cost += result.cost_usd
            state.sufficiency_history.append(result)

            iter_span.update(
                output={
                    "query_used": state.current_query,
                    "chunks_retrieved": len(new_chunks),
                    "new_unique_chunks_added": len(new_unique),
                    "is_sufficient": result.is_sufficient,
                    "total_chunks": len(state.accumulated_chunks),
                    "missing_aspects": result.missing_aspects,
                }
            )

            if result.is_sufficient:
                state.loop_terminated_by = "sufficiency_reached"
                break
    log.info(
        "agent loop completed",
        iteration=state.iteration,
        query_history=state.query_history,
        sufficiency=result.is_sufficient,
    )
    return state
