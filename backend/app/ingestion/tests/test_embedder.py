from unittest.mock import MagicMock, patch

import pytest

from app.ingestion.chunker.base_chunker import Chunk
from app.ingestion.embedder import (
    _load_local_model,
    embed_chunks,
    embed_query,
    get_embedding_dim,
)


@pytest.fixture(autouse=True)
def clear_local_model_cache():
    _load_local_model.cache_clear()
    yield
    _load_local_model.cache_clear()


def make_chunk(text: str, chunk_id: str = "doc_0") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc",
        doc_title="Doc",
        text=text,
        token_count=len(text.split()),
        chunk_index=0,
        doc_type="markdown",
        source_path="doc.md",
        tags=[],
        chunking_strategy="recursive",
    )


def test_get_embedding_dim_known_models():
    assert get_embedding_dim("text-embedding-3-small") == 1536
    assert get_embedding_dim("text-embedding-3-large") == 3072
    assert get_embedding_dim("BAAI/bge-large-en-v1.5") == 1024


def test_get_embedding_dim_unknown_model_raises():
    with pytest.raises(ValueError, match="unknown-model"):
        get_embedding_dim("unknown-model")


@patch("app.ingestion.embedder.SentenceTransformer")
def test_embed_chunks_routes_to_local_model(mock_st_cls):
    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]
    mock_st_cls.return_value = mock_model

    chunks = [make_chunk("hello"), make_chunk("world", "doc_1")]
    results = embed_chunks(chunks, model="BAAI/bge-large-en-v1.5")

    mock_st_cls.assert_called_once_with("BAAI/bge-large-en-v1.5")
    mock_model.encode.assert_called_once()
    assert [c.text for c, _ in results] == ["hello", "world"]
    assert [v for _, v in results] == [[0.1, 0.2], [0.3, 0.4]]


@patch("app.ingestion.embedder.SentenceTransformer")
def test_embed_query_routes_to_local_model_with_bge_instruction(mock_st_cls):
    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.5, 0.6]]
    mock_st_cls.return_value = mock_model

    vector = embed_query("what is RAG?", model="BAAI/bge-large-en-v1.5")

    encoded_input = mock_model.encode.call_args[0][0]
    assert encoded_input == [
        "Represent this sentence for searching relevant passages: what is RAG?"
    ]
    assert vector == [0.5, 0.6]


@patch("app.ingestion.embedder.client")
def test_embed_query_routes_to_openai_for_default_model(mock_client):
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.9, 0.8])]
    mock_client.embeddings.create.return_value = mock_response

    vector = embed_query("what is RAG?", model="text-embedding-3-small")

    mock_client.embeddings.create.assert_called_once_with(
        input=["what is RAG?"], model="text-embedding-3-small"
    )
    assert vector == [0.9, 0.8]
