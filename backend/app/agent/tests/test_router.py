from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.agent.router import _finalize_chunks
from app.agent.state import AgentLoopState, SufficiencyResult
from app.generation.generator import GenerationResult
from app.retrieval.searcher import RetrievedChunk
from main import app

client = TestClient(app)

FAKE_VECTOR = [0.1] * 1536


def _make_chunk(chunk_id: str = "c1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, doc_id="doc", doc_title="Doc",
        text="some text", score=0.9, source_path="f.pdf", chunk_index=0,
    )


def _make_state(chunks: list[RetrievedChunk], iteration: int = 1, loop_cost: float = 0.0) -> AgentLoopState:
    state = AgentLoopState(original_question="q", current_query="q")
    state.accumulated_chunks = chunks
    state.iteration = iteration
    state.loop_cost = loop_cost
    state.loop_terminated_by = "sufficiency_reached"
    state.sufficiency_history = [
        SufficiencyResult(is_sufficient=True, reasoning="ok", missing_aspects=[], confidence="high", cost_usd=0.0)
    ]
    return state


@patch("app.agent.router.embed_query", return_value=FAKE_VECTOR)
@patch("app.agent.router.get_semantic_cache")
def test_cache_hit_returns_cached_answer(mock_get_cache, mock_embed, monkeypatch):
    import app.agent.router as router_module
    from app.caching.cache import SemanticCache
    from app.caching.schema import CachedResponse
    from app.caching.tests.fakes import FakeRedis

    monkeypatch.setattr(router_module.settings, "enable_semantic_cache", True)

    cache = SemanticCache(client=FakeRedis(), similarity_threshold=0.95)
    cache.write(
        "What is RAG?", FAKE_VECTOR,
        CachedResponse(answer="cached answer", sources=[], cost_usd=0.01),
        "hybrid_rerank", "text-embedding-3-small",
    )
    mock_get_cache.return_value = cache

    resp = client.post("/agent/query", json={"question": "What is RAG?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["cache_hit"] is True
    assert body["answer"] == "cached answer"
    assert body["cost_usd"] == 0.0
    assert body["iterations_used"] == 0
    assert body["loop_terminated_by"] == "cache_hit"


@patch("app.agent.router.rerank")
@patch("app.agent.router.generate_answer")
@patch("app.agent.router.run_agent_loop")
@patch("app.agent.router.embed_query", return_value=FAKE_VECTOR)
@patch("app.agent.router.get_semantic_cache")
def test_cache_miss_returns_agentic_answer(
    mock_get_cache, mock_embed, mock_loop, mock_gen, mock_rerank
):
    from app.caching.cache import SemanticCache
    from app.caching.tests.fakes import FakeRedis

    mock_get_cache.return_value = SemanticCache(client=FakeRedis(), similarity_threshold=0.95)

    chunk = _make_chunk()
    state = _make_state([chunk], iteration=2, loop_cost=0.003)
    mock_loop.return_value = state
    mock_rerank.return_value = [chunk]
    mock_gen.return_value = GenerationResult(
        answer="agentic answer", sources=[chunk],
        prompt_tokens=100, completion_tokens=20, cost_usd=0.005,
    )

    resp = client.post("/agent/query", json={"question": "What is RAG?", "top_k": 3})

    assert resp.status_code == 200
    body = resp.json()
    assert body["cache_hit"] is False
    assert body["answer"] == "agentic answer"
    assert body["iterations_used"] == 2
    assert body["loop_terminated_by"] == "sufficiency_reached"
    assert abs(body["cost_usd"] - 0.008) < 1e-9  # 0.005 + 0.003
    assert len(body["sources"]) == 1
    assert body["sources"][0]["chunk_id"] == "c1"
    mock_rerank.assert_called_once_with("What is RAG?", [chunk], 3)


@patch("app.agent.router.run_agent_loop")
@patch("app.agent.router.embed_query", return_value=FAKE_VECTOR)
@patch("app.agent.router.get_semantic_cache")
def test_no_chunks_returns_404(mock_get_cache, mock_embed, mock_loop):
    from app.caching.cache import SemanticCache
    from app.caching.tests.fakes import FakeRedis

    mock_get_cache.return_value = SemanticCache(client=FakeRedis(), similarity_threshold=0.95)

    empty_state = AgentLoopState(original_question="q", current_query="q")
    empty_state.loop_cost = 0.0
    mock_loop.return_value = empty_state

    resp = client.post("/agent/query", json={"question": "unknowable"})

    assert resp.status_code == 404


@patch("app.agent.router.rerank")
def test_finalize_chunks_reranks_when_enabled(mock_rerank, monkeypatch):
    import app.agent.router as router_module

    monkeypatch.setattr(router_module.settings, "use_reranker", True)
    chunk = _make_chunk()
    reranked = [_make_chunk("c2")]
    mock_rerank.return_value = reranked

    result = _finalize_chunks("q", [chunk], 3)

    mock_rerank.assert_called_once_with("q", [chunk], 3)
    assert result == reranked


@patch("app.agent.router.rerank")
def test_finalize_chunks_skips_rerank_when_disabled(mock_rerank, monkeypatch):
    import app.agent.router as router_module

    monkeypatch.setattr(router_module.settings, "use_reranker", False)
    chunk = _make_chunk()

    result = _finalize_chunks("q", [chunk], 3)

    mock_rerank.assert_not_called()
    assert result == [chunk]
