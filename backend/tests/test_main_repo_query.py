from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.caching.cache import SemanticCache
from app.caching.tests.fakes import FakeRedis
from app.generation.generator import GenerationResult
from app.retrieval.searcher import RetrievedChunk
from main import app

client = TestClient(app)

FAKE_VECTOR = [0.1] * 1536


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1", doc_id="src/main.py", doc_title="src/main.py",
        text="def main(): ...", score=0.9, source_path="src/main.py",
        chunk_index=0,
    )


def _fake_cache() -> SemanticCache:
    return SemanticCache(client=FakeRedis(), similarity_threshold=0.95)


@patch("main.generate_answer")
@patch("main.retrieve_reranked")
@patch("main.get_qdrant_client")
@patch("main.embed_query", return_value=FAKE_VECTOR)
@patch("main.get_semantic_cache")
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

    resp = client.post(
        "/query", json={"question": "where is main?", "repo": "octo/hello"}
    )

    assert resp.status_code == 200
    mock_retrieve.assert_called_once()
    _, kwargs = mock_retrieve.call_args
    assert (
        kwargs["collection_name"]
        == "docmind_repo_octo-hello_text-embedding-3-small_hybrid"
    )
    assert kwargs["embedding_model"] == "text-embedding-3-small"


@patch("main.get_qdrant_client")
@patch("main.embed_query", return_value=FAKE_VECTOR)
@patch("main.get_semantic_cache")
def test_query_with_uningested_repo_returns_404(
    mock_cache, mock_embed, mock_get_client
):
    mock_cache.return_value = _fake_cache()
    mock_get_client.return_value.get_collections.return_value.collections = []

    resp = client.post(
        "/query", json={"question": "where is main?", "repo": "octo/missing"}
    )

    assert resp.status_code == 404
    assert "octo/missing" in resp.json()["detail"]


@patch("main.generate_answer")
@patch("main.retrieve_reranked")
@patch("main.get_qdrant_client")
@patch("main.embed_query", return_value=FAKE_VECTOR)
@patch("main.get_semantic_cache")
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

    resp = client.post(
        "/query",
        json={"question": "q", "repo": "octo/hello", "hybrid": False},
    )

    assert resp.status_code == 200
    mock_retrieve.assert_called_once()
