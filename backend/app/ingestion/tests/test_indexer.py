from unittest.mock import MagicMock

from qdrant_client import models

from app.ingestion.chunker.base_chunker import Chunk
from app.ingestion.indexer import (
    collection_name_for,
    ensure_hybrid_collection,
    upsert_chunks_hybrid,
)


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


def test_collection_name_for_default_unchanged():
    assert (
        collection_name_for("recursive", "text-embedding-3-small")
        == "docmind_recursive_text-embedding-3-small"
    )


def test_collection_name_for_hybrid_suffix():
    name = collection_name_for(
        "recursive", "text-embedding-3-small", hybrid=True
    )
    assert name == "docmind_recursive_text-embedding-3-small_hybrid"


def test_ensure_hybrid_collection_creates_named_vectors():
    client = MagicMock()
    client.get_collections.return_value.collections = []

    ensure_hybrid_collection(client, "docmind_recursive_test_hybrid", 1536)

    client.create_collection.assert_called_once()
    _, kwargs = client.create_collection.call_args
    assert kwargs["collection_name"] == "docmind_recursive_test_hybrid"
    assert "dense" in kwargs["vectors_config"]
    assert kwargs["vectors_config"]["dense"].size == 1536
    assert "bm25" in kwargs["sparse_vectors_config"]
    assert (
        kwargs["sparse_vectors_config"]["bm25"].modifier == models.Modifier.IDF
    )


def test_ensure_hybrid_collection_idempotent():
    client = MagicMock()
    existing = MagicMock()
    existing.name = "docmind_recursive_test_hybrid"
    client.get_collections.return_value.collections = [existing]

    ensure_hybrid_collection(client, "docmind_recursive_test_hybrid", 1536)

    client.create_collection.assert_not_called()


def test_upsert_chunks_hybrid_builds_named_vector_points():
    client = MagicMock()
    chunk = make_chunk("hello world")
    chunk_embeddings = [(chunk, [0.1, 0.2])]
    sparse_embeddings = [
        (chunk, models.SparseVector(indices=[1, 2], values=[0.5, 0.5]))
    ]

    upsert_chunks_hybrid(
        client, "docmind_recursive_test_hybrid", chunk_embeddings, sparse_embeddings
    )

    client.upsert.assert_called_once()
    _, kwargs = client.upsert.call_args
    points = kwargs["points"]
    assert len(points) == 1
    point = points[0]
    assert point.vector["dense"] == [0.1, 0.2]
    assert point.vector["bm25"].indices == [1, 2]
    assert point.vector["bm25"].values == [0.5, 0.5]
    assert point.payload["chunk_id"] == "doc_0"
    assert point.payload["text"] == "hello world"
