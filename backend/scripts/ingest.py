# scripts/ingest.py
"""
Chunk + embed + index the corpus into per-strategy Qdrant collections.

Usage:
  python -m scripts.ingest --strategy fixed_size
  python -m scripts.ingest --strategy all
"""

import argparse

from qdrant_client import QdrantClient

from app.config import get_settings
from app.ingestion.chunker import ChunkStrategy, get_chunker
from app.ingestion.embedder import embed_chunks, get_embedding_dim
from app.ingestion.indexer import (
    collection_name_for,
    ensure_collection,
    ensure_hybrid_collection,
    get_qdrant_client,
    upsert_chunks,
    upsert_chunks_hybrid,
)
from app.ingestion.loader import load_all_documents
from app.ingestion.sparse_embedder import embed_chunks_sparse


def resolve_targets(strategy: str) -> list[str]:
    all_strategies = [s.value for s in ChunkStrategy]
    if strategy == "all":
        return all_strategies
    if strategy not in all_strategies:
        raise ValueError(f"Unknown strategy: {strategy}")
    return [strategy]


def ingest_strategy(
    strategy: str, docs, client: QdrantClient, model: str, hybrid: bool = False
) -> None:
    print(f"\n=== Ingesting strategy: {strategy} (model: {model}) ===")
    chunker = get_chunker(
        ChunkStrategy(strategy), chunk_size=500, chunk_overlap=50
    )
    chunks = chunker.chunk_documents(docs)
    print(f"  Total chunks: {len(chunks)}")

    total_tokens = sum(c.token_count for c in chunks)
    print(
        f"  Tokens to embed: {total_tokens:,} "
        f"(~${(total_tokens / 1_000_000) * 0.02:.4f})"
    )
    chunk_embeddings = embed_chunks(chunks, model=model)
    if hybrid:
        print(f"  Building hybrid (dense+BM25) variant for {strategy}...")
        sparse_embeddings = embed_chunks_sparse(chunks)
        hybrid_collection = collection_name_for(strategy, model, hybrid=True)
        ensure_hybrid_collection(
            client, hybrid_collection, vector_size=get_embedding_dim(model)
        )
        upsert_chunks_hybrid(
            client, hybrid_collection, chunk_embeddings, sparse_embeddings
        )
        hybrid_info = client.get_collection(hybrid_collection)
        print(f"  {hybrid_collection}: {hybrid_info.points_count} points")
    else:
        collection = collection_name_for(strategy, model)
        ensure_collection(
            client, collection, vector_size=get_embedding_dim(model)
        )
        upsert_chunks(client, collection, chunk_embeddings)
        info = client.get_collection(collection)
        print(f"  {collection}: {info.points_count} points")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest corpus into Qdrant.")
    parser.add_argument(
        "--strategy",
        required=True,
        choices=[s.value for s in ChunkStrategy] + ["all"],
        help="Chunking strategy to ingest (or 'all').",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model to use (default: settings.embedding_model). "
        "Pass a local model like BAAI/bge-large-en-v1.5 to use "
        "sentence-transformers instead of OpenAI.",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Build a hybrid (dense+BM25 sparse) collection instead of the "
        "normal dense-only collection, for the given --strategy/"
        "--embedding-model.",
    )
    args = parser.parse_args()

    settings = get_settings()
    model = args.embedding_model or settings.embedding_model
    print("=== DocMind Ingestion ===")
    docs = load_all_documents()
    print(f"Loaded {len(docs)} documents")

    client = get_qdrant_client()
    for strategy in resolve_targets(args.strategy):
        ingest_strategy(strategy, docs, client, model, hybrid=args.hybrid)

    print("\n=== Ingestion complete ===")


if __name__ == "__main__":
    main()
