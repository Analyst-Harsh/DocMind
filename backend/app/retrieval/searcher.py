from dataclasses import dataclass

from qdrant_client import QdrantClient, models
from structlog import get_logger

from app.config import get_settings
from app.ingestion.embedder import embed_query
from app.ingestion.indexer import get_qdrant_client
from app.ingestion.sparse_embedder import embed_query_sparse
from app.retrieval.reranker import rerank
from app.tracing.spans import traced_span

log = get_logger(__name__)
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
    collection_name: str | None = None,
    embedding_model: str | None = None,
) -> list[RetrievedChunk]:
    """
    Embed the query and find the top_k most similar chunks in Qdrant.
    Returns chunks sorted by relevance score descending.
    collection_name defaults to settings.qdrant_collection; embedding_model
    defaults to the configured model (must match the model the collection
    was ingested with).
    """
    if client is None:
        client = get_qdrant_client()

    query_vector = embed_query(query, model=embedding_model)

    results = client.query_points(
        collection_name=collection_name or settings.qdrant_collection,
        query=query_vector,
        limit=top_k,
        with_payload=True,  # return metadata alongside vectors
    )

    return _points_to_chunks(results.points)


def retrieve_hybrid(
    query: str,
    top_k: int = 5,
    client: QdrantClient | None = None,
    collection_name: str | None = None,
    embedding_model: str | None = None,
    prefetch_limit: int = 20,
) -> list[RetrievedChunk]:
    """
    Hybrid dense + BM25 sparse retrieval against a collection built with
    named "dense"/"bm25" vectors (see indexer.ensure_hybrid_collection).
    Fuses both result sets server-side via Reciprocal Rank Fusion (RRF).
    """
    if client is None:
        client = get_qdrant_client()

    with traced_span("dense-embedding", as_type="embedding") as span:
        dense_vector = embed_query(query, model=embedding_model)
        span.update(output={"dim": len(dense_vector)})

    with traced_span("sparse-embedding", as_type="embedding") as span:
        sparse_vector = embed_query_sparse(query)
        span.update(output={"num_terms": len(sparse_vector.indices)})

    with traced_span(
        "hybrid-search",
        as_type="retriever",
        input={
            "prefetch_limit": prefetch_limit,
            "top_k": top_k,
            "fusion": "RRF",
        },
    ) as span:
        results = client.query_points(
            collection_name=collection_name or settings.qdrant_collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vector, using="dense", limit=prefetch_limit
                ),
                models.Prefetch(
                    query=sparse_vector, using="bm25", limit=prefetch_limit
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        chunks = _points_to_chunks(results.points)
        span.update(output={"num_results": len(chunks), "top_chunks": chunks})

    return chunks


def retrieve_reranked(
    query: str,
    top_k: int = 5,
    client: QdrantClient | None = None,
    collection_name: str | None = None,
    embedding_model: str | None = None,
    candidate_pool_size: int = 20,
) -> list[RetrievedChunk]:
    """
    Hybrid retrieval over a wide candidate pool, re-scored by a
    cross-encoder reranker and truncated to top_k. The candidate pool's
    RRF score/order is discarded -- the reranker fully re-scores and
    re-sorts, so only candidate-set membership from retrieve_hybrid
    matters here.
    """
    candidates = retrieve_hybrid(
        query,
        top_k=int(candidate_pool_size / 2),
        client=client,
        collection_name=collection_name,
        embedding_model=embedding_model,
        prefetch_limit=candidate_pool_size,
    )
    log.info(f"retrieved {len(candidates)} candidates for query '{query}'")
    return rerank(query, candidates, top_k)


def _points_to_chunks(points) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=r.payload.get("chunk_id", "")
            if r.payload is not None
            else "",
            doc_id=r.payload.get("doc_id", "") if r.payload is not None else "",
            doc_title=r.payload.get("doc_title", "")
            if r.payload is not None
            else "",
            text=r.payload.get("text", "") if r.payload is not None else "",
            score=r.score,
            source_path=r.payload.get("source_path", "")
            if r.payload is not None
            else "",
            chunk_index=r.payload.get("chunk_index", "")
            if r.payload is not None
            else 0,
        )
        for r in points
    ]
