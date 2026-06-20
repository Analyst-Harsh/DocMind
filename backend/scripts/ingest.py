# scripts/ingest.py
"""
Chunk + embed + index the corpus into per-strategy Qdrant collections.

Usage:
  python -m scripts.ingest --strategy fixed_size
  python -m scripts.ingest --strategy all
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.loader import load_all_documents
from app.ingestion.chunker import get_chunker, ChunkStrategy
from app.ingestion.embedder import embed_chunks
from app.ingestion.indexer import (
    get_qdrant_client,
    ensure_collection,
    upsert_chunks,
    collection_name_for,
)
from app.config import get_settings


def resolve_targets(strategy: str) -> list[str]:
    all_strategies = [s.value for s in ChunkStrategy]
    if strategy == "all":
        return all_strategies
    if strategy not in all_strategies:
        raise ValueError(f"Unknown strategy: {strategy}")
    return [strategy]


def ingest_strategy(strategy: str, docs, client, settings) -> None:
    print(f"\n=== Ingesting strategy: {strategy} ===")
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
    chunk_embeddings = embed_chunks(chunks)

    collection = collection_name_for(strategy, settings.embedding_model)
    ensure_collection(client, collection)
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
    args = parser.parse_args()

    settings = get_settings()
    print("=== DocMind Ingestion ===")
    docs = load_all_documents()
    print(f"Loaded {len(docs)} documents")

    client = get_qdrant_client()
    for strategy in resolve_targets(args.strategy):
        ingest_strategy(strategy, docs, client, settings)

    print("\n=== Ingestion complete ===")


if __name__ == "__main__":
    main()
