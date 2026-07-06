# app/ingestion/indexer.py
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PointStruct,
    SparseVector,
    SparseVectorParams,
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


def collection_name_for(
    strategy: str, embedding_model: str, hybrid: bool = False
) -> str:
    """
    One Qdrant collection per (strategy, embedding model) pair, e.g.
    docmind_chunks_fixed_size_text-embedding-3-small. Trying a new
    embedding model just creates new collections — prior models' data
    is never overwritten.

    Qdrant's REST API takes the collection name as a URL path segment, so
    "/" (as in HuggingFace model ids like "BAAI/bge-large-en-v1.5") must be
    sanitized or every request 404s.

    hybrid=True suffixes "_hybrid" since hybrid collections use named
    dense+sparse vectors — a different schema from the dense-only
    collections, so they can't share a name.
    """
    safe_model = embedding_model.replace("/", "-")
    suffix = "_hybrid" if hybrid else ""
    return f"{settings.qdrant_collection}_{strategy}_{safe_model}{suffix}"


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


def multimodal_collection_name(embedding_model: str, hybrid: bool = False) -> str:
    """
    Collection name for table chunks, separate from the strategy-based text
    collections so the two pipelines can be compared directly in Qdrant.
    """
    safe_model = embedding_model.replace("/", "-")
    suffix = "_hybrid" if hybrid else ""
    return f"multimodal_{safe_model}{suffix}"


def upsert_table_chunks(
    client: QdrantClient,
    collection_name: str,
    chunk_embeddings: list[tuple[Chunk, list[float]]],
):
    """
    Upsert table chunks into Qdrant, including table-specific payload fields
    (table_markdown, table_headers, page_number, etc.) that upsert_chunks omits.
    """
    points = []
    for chunk, vector in chunk_embeddings:
        points.append(
            PointStruct(
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
                    "table_markdown": chunk.table_markdown,
                    "table_headers": chunk.table_headers,
                    "table_index": chunk.table_index,
                    "page_number": chunk.page_number,
                    "row_count": chunk.row_count,
                    "col_count": chunk.col_count,
                    "is_table": chunk.is_table,
                },
            )
        )

    for i in range(0, len(points), 100):
        client.upsert(collection_name=collection_name, points=points[i : i + 100])
    log.info(f"Upserted {len(points)} table points into {collection_name}")


def upsert_table_chunks_hybrid(
    client: QdrantClient,
    collection_name: str,
    chunk_embeddings: list[tuple[Chunk, list[float]]],
    sparse_embeddings: list[tuple[Chunk, SparseVector]],
):
    """Hybrid variant of upsert_table_chunks with dense + BM25 sparse vectors."""
    points = []
    for (chunk, dense_vec), (_, sparse_vec) in zip(
        chunk_embeddings, sparse_embeddings, strict=True
    ):
        points.append(
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id)),
                vector={"dense": dense_vec, "bm25": sparse_vec},
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
                    "table_markdown": chunk.table_markdown,
                    "table_headers": chunk.table_headers,
                    "table_index": chunk.table_index,
                    "page_number": chunk.page_number,
                    "row_count": chunk.row_count,
                    "col_count": chunk.col_count,
                    "is_table": chunk.is_table,
                },
            )
        )

    for i in range(0, len(points), 100):
        client.upsert(collection_name=collection_name, points=points[i : i + 100])
    log.info(f"Upserted {len(points)} hybrid table points into {collection_name}")


def ensure_hybrid_collection(
    client: QdrantClient, collection_name: str, vector_size: int
):
    """
    Create a hybrid (dense + BM25 sparse) collection if it doesn't exist.
    Named vectors "dense" and "bm25" let one collection serve both
    dense similarity search and sparse lexical search, fused via RRF
    at query time.

    Modifier.IDF tells Qdrant to compute IDF server-side from the
    collection's document-frequency stats — fastembed's BM25 sparse
    vectors only carry term-frequency weights, not corpus-wide IDF.
    """
    collections = [c.name for c in client.get_collections().collections]

    if collection_name not in collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(size=vector_size, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "bm25": SparseVectorParams(modifier=Modifier.IDF)
            },
        )
        log.info(f"Created hybrid collection: {collection_name}")
    else:
        log.info(f"Collection already exists: {collection_name}")


def upsert_chunks_hybrid(
    client: QdrantClient,
    collection_name: str,
    chunk_embeddings: list[tuple[Chunk, list[float]]],
    sparse_embeddings: list[tuple[Chunk, SparseVector]],
):
    """
    Upsert chunks with both dense and BM25 sparse vectors into a hybrid
    collection. chunk_embeddings and sparse_embeddings must cover the
    same chunks in the same order.
    """
    points = []
    for (chunk, dense_vec), (_, sparse_vec) in zip(
        chunk_embeddings, sparse_embeddings, strict=True
    ):
        points.append(
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id)),
                vector={"dense": dense_vec, "bm25": sparse_vec},
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

    for i in range(0, len(points), 100):
        batch = points[i : i + 100]
        client.upsert(collection_name=collection_name, points=batch)
    log.info(f"Upserted {len(points)} hybrid points into {collection_name}")
