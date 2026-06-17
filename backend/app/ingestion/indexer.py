# app/ingestion/indexer.py
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)
from app.config import get_settings
from app.ingestion.chunker import Chunk

settings = get_settings()


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )


def ensure_collection(client: QdrantClient, vector_size: int = 1536):
    """
    Create the Qdrant collection if it doesn't exist.
    Safe to call multiple times — won't overwrite existing data.
    """
    collections = [c.name for c in client.get_collections().collections]

    if settings.qdrant_collection not in collections:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        print(f"Created collection: {settings.qdrant_collection}")
    else:
        print(f"Collection already exists: {settings.qdrant_collection}")


def upsert_chunks(
    client: QdrantClient,
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
                id=abs(hash(chunk.chunk_id)) % (2**63),
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
            collection_name=settings.qdrant_collection,
            points=batch,
        )
    print(f"Upserted {len(points)} points into Qdrant")
