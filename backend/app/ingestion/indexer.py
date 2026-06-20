# app/ingestion/indexer.py
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from structlog import get_logger

from app.config import get_settings
from app.ingestion.chunker import Chunk

log = get_logger(__name__)
settings = get_settings()


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )


def collection_name_for(strategy: str, embedding_model: str) -> str:
    """
    One Qdrant collection per (strategy, embedding model) pair, e.g.
    docmind_chunks_fixed_size_text-embedding-3-small. Trying a new
    embedding model just creates new collections — prior models' data
    is never overwritten.

    Qdrant's REST API takes the collection name as a URL path segment, so
    "/" (as in HuggingFace model ids like "BAAI/bge-large-en-v1.5") must be
    sanitized or every request 404s.
    """
    safe_model = embedding_model.replace("/", "-")
    return f"{settings.qdrant_collection}_{strategy}_{safe_model}"


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int):
    """
    Create the Qdrant collection if it doesn't exist.
    Safe to call multiple times — won't overwrite existing data.
    """
    collections = [c.name for c in client.get_collections().collections]

    if collection_name not in collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        log.info(f"Created collection: {collection_name}")
    else:
        log.info(f"Collection already exists: {collection_name}")


def upsert_chunks(
    client: QdrantClient,
    collection_name: str,
    chunk_embeddings: list[tuple[Chunk, list[float]]],
):
    """
    Upsert chunks + vectors into Qdrant.
    The payload (metadata) stored alongside the vector is what
    comes back at retrieval time to build citations.
    """
    points = []
    for chunk, vector in chunk_embeddings:
        points.append(
            PointStruct(
                # Qdrant requires integer or UUID ids
                # We hash the chunk_id string to an int
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id)),
                vector=vector,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "doc_title": chunk.doc_title,
                    "text": chunk.text,
                    "token_count": chunk.token_count,
                    "chunk_index": chunk.chunk_index,
                    "doc_type": chunk.doc_type,
                    "source_path": chunk.source_path,
                    "tags": chunk.tags,
                },
            )
        )

    # Upsert in batches of 100
    for i in range(0, len(points), 100):
        batch = points[i : i + 100]
        client.upsert(
            collection_name=collection_name,
            points=batch,
        )
    log.info(f"Upserted {len(points)} points into {collection_name}")
