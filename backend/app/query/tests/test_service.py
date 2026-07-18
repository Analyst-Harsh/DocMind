from unittest.mock import MagicMock, patch

import pytest

from app.caching.cache import SemanticCache
from app.caching.schema import CachedResponse
from app.caching.tests.fakes import FakeRedis
from app.config import get_settings
from app.generation.generator import GenerationResult
from app.query.service import (
    NoRelevantChunksError,
    RepoNotIngestedError,
    run_query,
)
from app.retrieval.searcher import RetrievedChunk

FAKE_VECTOR = [0.1] * 1536


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1", doc_id="src/main.py", doc_title="src/main.py",
        text="def main(): ...", score=0.9, source_path="src/main.py",
        chunk_index=0,
    )


def _fake_cache():
    return SemanticCache(client=FakeRedis(), similarity_threshold=0.95)


@patch("app.query.service.generate_answer")
@patch("app.query.service.retrieve_reranked")
@patch("app.query.service.get_qdrant_client")
@patch("app.query.service.embed_query", return_value=FAKE_VECTOR)
@patch("app.query.service.get_semantic_cache")
def test_query_with_repo_routes_to_repo_collection(
    mock_cache, mock_embed, mock_get_client, mock_retrieve, mock_gen
):
    mock_cache.return_value = _fake_cache()

    fake_collection = MagicMock()
    fake_collection.name = "docmind_repo_octo-hello_text-embedding-3-small_hybrid"
    mock_get_client.return_value.get_collections.return_value.collections = [
        fake_collection
    ]

    chunk = _chunk()
    mock_retrieve.return_value = [chunk]
    mock_gen.return_value = GenerationResult(
        answer="it's in src/main.py", sources=[chunk],
        prompt_tokens=10, completion_tokens=5, cost_usd=0.001,
    )

    result = run_query(question="where is main?", repo="octo/hello")

    assert result.answer == "it's in src/main.py"
    mock_retrieve.assert_called_once()
    _, kwargs = mock_retrieve.call_args
    assert (
        kwargs["collection_name"]
        == "docmind_repo_octo-hello_text-embedding-3-small_hybrid"
    )
    assert kwargs["embedding_model"] == "text-embedding-3-small"


@patch("app.query.service.get_qdrant_client")
@patch("app.query.service.embed_query", return_value=FAKE_VECTOR)
@patch("app.query.service.get_semantic_cache")
def test_query_with_uningested_repo_raises_repo_not_ingested(
    mock_cache, mock_embed, mock_get_client
):
    mock_cache.return_value = _fake_cache()
    mock_get_client.return_value.get_collections.return_value.collections = []

    with pytest.raises(RepoNotIngestedError) as exc_info:
        run_query(question="where is main?", repo="octo/missing")

    assert "octo/missing" in str(exc_info.value)


@patch("app.query.service.generate_answer")
@patch("app.query.service.retrieve_reranked")
@patch("app.query.service.get_qdrant_client")
@patch("app.query.service.embed_query", return_value=FAKE_VECTOR)
@patch("app.query.service.get_semantic_cache")
def test_query_repo_flag_forces_hybrid_even_if_hybrid_false(
    mock_cache, mock_embed, mock_get_client, mock_retrieve, mock_gen
):
    # repo collections are hybrid-only -- setting repo must force hybrid
    # retrieval regardless of the (docs-corpus-only) hybrid flag.
    mock_cache.return_value = _fake_cache()
    fake_collection = MagicMock()
    fake_collection.name = "docmind_repo_octo-hello_text-embedding-3-small_hybrid"
    mock_get_client.return_value.get_collections.return_value.collections = [
        fake_collection
    ]
    chunk = _chunk()
    mock_retrieve.return_value = [chunk]
    mock_gen.return_value = GenerationResult(
        answer="answer", sources=[chunk],
        prompt_tokens=1, completion_tokens=1, cost_usd=0.0001,
    )

    result = run_query(question="q", repo="octo/hello", hybrid=False)

    assert result.answer == "answer"
    mock_retrieve.assert_called_once()


@patch("app.query.service.retrieve_reranked")
@patch("app.query.service.get_semantic_cache")
@patch("app.query.service.embed_query", return_value=FAKE_VECTOR)
def test_query_raises_no_relevant_chunks_when_retrieval_empty(
    mock_embed, mock_cache, mock_retrieve
):
    mock_cache.return_value = _fake_cache()
    mock_retrieve.return_value = []

    with pytest.raises(NoRelevantChunksError):
        run_query(question="anything")


@patch("app.query.service.generate_answer")
@patch("app.query.service.embed_query", return_value=FAKE_VECTOR)
@patch("app.query.service.get_semantic_cache")
def test_query_cache_hit_skips_generation(
    mock_get_cache, mock_embed, mock_gen, monkeypatch
):
    monkeypatch.setattr(get_settings(), "enable_semantic_cache", True)

    cache = _fake_cache()
    cache.write(
        "What is RAG?", FAKE_VECTOR,
        CachedResponse(answer="cached answer", sources=[], cost_usd=0.01),
        "dense", get_settings().embedding_model,
    )
    mock_get_cache.return_value = cache

    result = run_query(question="What is RAG?", hybrid=False)

    assert result.cache_hit is True
    assert result.answer == "cached answer"
    assert result.cost_usd == 0.0
    mock_gen.assert_not_called()
