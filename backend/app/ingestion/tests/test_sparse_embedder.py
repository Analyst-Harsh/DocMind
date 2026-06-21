from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from qdrant_client import models

from app.ingestion.chunker.base_chunker import Chunk
from app.ingestion.sparse_embedder import (
    _load_sparse_model,
    embed_chunks_sparse,
    embed_query_sparse,
)


@pytest.fixture(autouse=True)
def clear_sparse_model_cache():
    _load_sparse_model.cache_clear()
    yield
    _load_sparse_model.cache_clear()


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


class FakeSparseEmbedding:
    def __init__(self, indices, values):
        self.indices = np.array(indices)
        self.values = np.array(values)


@patch("app.ingestion.sparse_embedder.SparseTextEmbedding")
def test_embed_chunks_sparse_returns_sparse_vectors(mock_cls):
    mock_model = MagicMock()
    mock_model.embed.return_value = [
        FakeSparseEmbedding([1, 2], [0.5, 0.5]),
        FakeSparseEmbedding([3], [1.0]),
    ]
    mock_cls.return_value = mock_model

    chunks = [make_chunk("hello world"), make_chunk("bm25 search", "doc_1")]
    results = embed_chunks_sparse(chunks)

    mock_model.embed.assert_called_once_with(["hello world", "bm25 search"])
    assert [c.text for c, _ in results] == ["hello world", "bm25 search"]
    sparse_vectors = [v for _, v in results]
    assert all(isinstance(v, models.SparseVector) for v in sparse_vectors)
    assert sparse_vectors[0].indices == [1, 2]
    assert sparse_vectors[0].values == [0.5, 0.5]
    assert sparse_vectors[1].indices == [3]
    assert sparse_vectors[1].values == [1.0]


@patch("app.ingestion.sparse_embedder.SparseTextEmbedding")
def test_embed_query_sparse_uses_query_embed_mode(mock_cls):
    mock_model = MagicMock()
    mock_model.query_embed.return_value = [FakeSparseEmbedding([7, 8], [1, 1])]
    mock_cls.return_value = mock_model

    vector = embed_query_sparse("hybrid search query")

    mock_model.query_embed.assert_called_once_with(["hybrid search query"])
    mock_model.embed.assert_not_called()
    assert isinstance(vector, models.SparseVector)
    assert vector.indices == [7, 8]
    assert vector.values == [1, 1]
