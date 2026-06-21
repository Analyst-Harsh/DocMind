# app/ingestion/sparse_embedder.py
from functools import lru_cache

from fastembed import SparseTextEmbedding
from qdrant_client import models

from app.ingestion.chunker import Chunk

SPARSE_MODEL_NAME = "Qdrant/bm25"


@lru_cache(maxsize=1)
def _load_sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


def embed_chunks_sparse(
    chunks: list[Chunk],
) -> list[tuple[Chunk, models.SparseVector]]:
    """
    Document-mode BM25 sparse encoding (term-frequency weighted).
    Returns list of (chunk, sparse_vector) pairs, same order as input.
    """
    model = _load_sparse_model()
    texts = [c.text for c in chunks]
    sparse_embeddings = list(model.embed(texts))
    return [
        (
            chunk,
            models.SparseVector(
                indices=se.indices.tolist(), values=se.values.tolist()
            ),
        )
        for chunk, se in zip(chunks, sparse_embeddings, strict=True)
    ]


def embed_query_sparse(text: str) -> models.SparseVector:
    """Query-mode BM25 sparse encoding (binary term presence)."""
    model = _load_sparse_model()
    se = next(iter(model.query_embed([text])))
    return models.SparseVector(
        indices=se.indices.tolist(), values=se.values.tolist()
    )
