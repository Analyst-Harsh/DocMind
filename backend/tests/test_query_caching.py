from unittest.mock import patch

from fastapi.testclient import TestClient

from app.caching.cache import SemanticCache
from app.caching.schema import CachedResponse
from app.caching.tests.fakes import FakeRedis
from app.config import get_settings
from app.generation.generator import GenerationResult
from app.retrieval.searcher import RetrievedChunk
from main import app

client = TestClient(app)

FAKE_EMBEDDING = [1.0, 0.0]


def make_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c0",
        doc_id="doc",
        doc_title="Doc",
        text="hello",
        score=0.9,
        source_path="doc.md",
        chunk_index=0,
    )


@patch("main.generate_answer")
@patch("main.retrieve")
@patch("main.embed_query", return_value=FAKE_EMBEDDING)
@patch("main.get_semantic_cache")
def test_cache_miss_runs_full_pipeline_and_writes_cache(
    mock_get_cache, mock_embed_query, mock_retrieve, mock_generate_answer
):
    fake_cache = SemanticCache(client=FakeRedis(), similarity_threshold=0.95)
    mock_get_cache.return_value = fake_cache
    mock_retrieve.return_value = [make_chunk()]
    mock_generate_answer.return_value = GenerationResult(
        answer="RAG combines retrieval and generation.",
        sources=[make_chunk()],
        prompt_tokens=100,
        completion_tokens=20,
        cost_usd=0.002,
    )

    response = client.post("/query", json={"question": "What is RAG?"})

    assert response.status_code == 200
    body = response.json()
    assert body["cache_hit"] is False
    assert body["answer"] == "RAG combines retrieval and generation."
    mock_retrieve.assert_called_once()
    mock_generate_answer.assert_called_once()

    embedding_model = get_settings().embedding_model
    lookup = fake_cache.check(
        FAKE_EMBEDDING, retrieval_mode="dense", embedding_model=embedding_model
    )
    assert lookup.hit is not None
    assert lookup.hit.response.answer == "RAG combines retrieval and generation."


@patch("main.generate_answer")
@patch("main.retrieve")
@patch("main.embed_query", return_value=FAKE_EMBEDDING)
@patch("main.get_semantic_cache")
def test_cache_hit_skips_retrieval_and_generation(
    mock_get_cache, mock_embed_query, mock_retrieve, mock_generate_answer
):
    embedding_model = get_settings().embedding_model
    fake_cache = SemanticCache(client=FakeRedis(), similarity_threshold=0.95)
    fake_cache.write(
        "What is RAG?",
        FAKE_EMBEDDING,
        CachedResponse(answer="cached answer", sources=[], cost_usd=0.002),
        retrieval_mode="dense",
        embedding_model=embedding_model,
    )
    mock_get_cache.return_value = fake_cache

    response = client.post("/query", json={"question": "What is RAG, basically?"})

    assert response.status_code == 200
    body = response.json()
    assert body["cache_hit"] is True
    assert body["answer"] == "cached answer"
    # The cached entry's original cost was 0.002 -- this request must
    # report 0.0, since serving it from cache made no LLM call.
    assert body["cost_usd"] == 0.0
    mock_retrieve.assert_not_called()
    mock_generate_answer.assert_not_called()
