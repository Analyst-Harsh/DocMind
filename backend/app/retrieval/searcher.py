# app/retrieval/searcher.py
from qdrant_client import QdrantClient
from app.ingestion.indexer import get_qdrant_client
from app.ingestion.embedder import embed_query
from app.config import get_settings
from dataclasses import dataclass

settings = get_settings()


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    score: float
    source_path: str
    chunk_index: int


def retrieve(
    query: str,
    top_k: int = 5,
    client: QdrantClient | None = None,
) -> list[RetrievedChunk]:
    """
    Embed the query and find the top_k most similar chunks in Qdrant.
    Returns chunks sorted by relevance score descending.
    """
    if client is None:
        client = get_qdrant_client()

    query_vector = embed_query(query)

    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=top_k,
        with_payload=True,  # return metadata alongside vectors
    )

    return [
        RetrievedChunk(
            chunk_id=r.payload.get("chunk_id", "") if r.payload is not None else "",
            doc_id=r.payload.get("doc_id", "") if r.payload is not None else "",
            doc_title=r.payload.get("doc_title", "") if r.payload is not None else "",
            text=r.payload.get("text", "") if r.payload is not None else "",
            score=r.score,
            source_path=r.payload.get("source_path", "") if r.payload is not None else "",
            chunk_index=r.payload.get("chunk_index", "") if r.payload is not None else 0,
        )
        for r in results.points
    ]
