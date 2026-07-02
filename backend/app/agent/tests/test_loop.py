from unittest.mock import patch

from app.agent.state import AgentLoopState, SufficiencyResult
from app.retrieval.searcher import RetrievedChunk


def _make_chunk(chunk_id: str, text: str = "some text") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, doc_id="d1", doc_title="Doc", text=text,
        score=0.9, source_path="f.pdf", chunk_index=0,
    )


def _sufficient(missing: list[str] | None = None) -> SufficiencyResult:
    return SufficiencyResult(
        is_sufficient=True, reasoning="ok", missing_aspects=missing or [], confidence="high"
    )


def _insufficient(missing: list[str]) -> SufficiencyResult:
    return SufficiencyResult(
        is_sufficient=False, reasoning="missing stuff",
        missing_aspects=missing, confidence="high",
    )


def test_loop_stops_on_first_iteration_when_sufficient():
    from app.agent.loop import run_agent_loop

    chunks = [_make_chunk("c1"), _make_chunk("c2")]

    with patch("app.agent.loop._retrieve", return_value=chunks) as mock_retrieve, \
         patch("app.agent.loop.assess_sufficiency", return_value=_sufficient()) as mock_assess:
        state = run_agent_loop("What is RAG?", top_k=5)

    assert state.iteration == 1
    assert mock_retrieve.call_count == 1
    assert mock_assess.call_count == 1
    assert len(state.accumulated_chunks) == 2
    assert len(state.sufficiency_history) == 1
    assert state.sufficiency_history[0].is_sufficient is True


def test_loop_reformulates_and_runs_second_iteration():
    from app.agent.loop import run_agent_loop

    chunks_iter1 = [_make_chunk("c1")]
    chunks_iter2 = [_make_chunk("c2")]

    with patch("app.agent.loop._retrieve", side_effect=[chunks_iter1, chunks_iter2]), \
         patch("app.agent.loop.assess_sufficiency", side_effect=[
             _insufficient(["reranking mechanism"]),
             _sufficient(),
         ]), \
         patch("app.agent.loop.reformulate_query", return_value=("reranking mechanism RAG", 0.0)) as mock_reform:
        state = run_agent_loop("How does reranking work?")

    assert state.iteration == 2
    assert len(state.accumulated_chunks) == 2
    assert state.current_query == "reranking mechanism RAG"
    mock_reform.assert_called_once_with(
        "How does reranking work?", ["reranking mechanism"]
    )


def test_loop_deduplicates_chunks_across_iterations():
    from app.agent.loop import run_agent_loop

    shared_chunk = _make_chunk("c1", "shared text")
    new_chunk = _make_chunk("c2", "new text")

    with patch("app.agent.loop._retrieve", side_effect=[[shared_chunk], [shared_chunk, new_chunk]]), \
         patch("app.agent.loop.assess_sufficiency", side_effect=[
             _insufficient(["more info"]),
             _sufficient(),
         ]), \
         patch("app.agent.loop.reformulate_query", return_value=("more info query", 0.0)):
        state = run_agent_loop("question")

    chunk_ids = [c.chunk_id for c in state.accumulated_chunks]
    assert chunk_ids.count("c1") == 1, "c1 must appear exactly once despite appearing in both iterations"
    assert "c2" in chunk_ids
    assert len(state.accumulated_chunks) == 2


def test_loop_stops_at_max_iterations_even_when_always_insufficient():
    from app.agent.loop import MAX_ITERATIONS, run_agent_loop

    chunks = [_make_chunk("c1")]

    with patch("app.agent.loop._retrieve", return_value=chunks), \
         patch("app.agent.loop.assess_sufficiency",
               return_value=_insufficient(["always missing"])), \
         patch("app.agent.loop.reformulate_query", return_value=("new query", 0.0)):
        state = run_agent_loop("impossible question")

    assert state.iteration == MAX_ITERATIONS
    assert len(state.sufficiency_history) == MAX_ITERATIONS
    assert all(not r.is_sufficient for r in state.sufficiency_history)


def test_loop_preserves_original_question_across_reformulations():
    from app.agent.loop import run_agent_loop

    with patch("app.agent.loop._retrieve", return_value=[_make_chunk("c1")]), \
         patch("app.agent.loop.assess_sufficiency", return_value=_sufficient()), \
         patch("app.agent.loop.reformulate_query", return_value=("reformulated", 0.0)):
        state = run_agent_loop("original question")

    assert state.original_question == "original question"


def test_loop_terminated_by_sufficiency_reached():
    from app.agent.loop import run_agent_loop

    with patch("app.agent.loop._retrieve", return_value=[_make_chunk("c1")]), \
         patch("app.agent.loop.assess_sufficiency", return_value=_sufficient()):
        state = run_agent_loop("question")

    assert state.loop_terminated_by == "sufficiency_reached"


def test_loop_terminated_by_cap_reached():
    from app.agent.loop import MAX_ITERATIONS, run_agent_loop

    with patch("app.agent.loop._retrieve", return_value=[_make_chunk("c1")]), \
         patch("app.agent.loop.assess_sufficiency",
               return_value=_insufficient(["always missing"])), \
         patch("app.agent.loop.reformulate_query", return_value=("new query", 0.0)):
        state = run_agent_loop("impossible question")

    assert state.loop_terminated_by == "cap_reached"
    assert state.iteration == MAX_ITERATIONS