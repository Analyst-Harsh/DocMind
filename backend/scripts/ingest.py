# scripts/ingest.py
"""
Run this once (or whenever corpus changes) to ingest all documents.
Usage: python -m scripts.ingest
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.loader import load_all_documents
from app.ingestion.chunker import FixedSizeChunker
from app.ingestion.embedder import embed_chunks
from app.ingestion.indexer import get_qdrant_client, ensure_collection, upsert_chunks
from app.config import get_settings


def main():
    settings = get_settings()
    print("=== DocMind Ingestion ===\n")

    # 1. Load raw documents
    print("Step 1: Loading documents...")
    docs = load_all_documents()
    print(f"  Loaded {len(docs)} documents\n")

    # 2. Chunk
    print("Step 2: Chunking...")
    chunker = FixedSizeChunker()
    chunks = chunker.chunk_documents(docs)
    print(f"  Total chunks: {len(chunks)}\n")

    # 3. Embed
    print("Step 3: Embedding (this costs money, be aware)...")
    total_tokens = sum(c.token_count for c in chunks)
    print(f"  Total tokens to embed: {total_tokens:,}")
    # text-embedding-3-small costs $0.02 per 1M tokens
    estimated_cost = (total_tokens / 1_000_000) * 0.02
    print(f"  Estimated cost: ${estimated_cost:.4f}")

    chunk_embeddings = embed_chunks(chunks)
    print(f"  Done.\n")

    # 4. Index into Qdrant
    print("Step 4: Indexing into Qdrant...")
    client = get_qdrant_client()
    ensure_collection(client)
    upsert_chunks(client, chunk_embeddings)

    # 5. Verify
    info = client.get_collection(settings.qdrant_collection)
    print(f"\n=== Ingestion complete ===")
    print(f"Collection: {settings.qdrant_collection}")
    print(f"Points in Qdrant: {info.points_count}")


if __name__ == "__main__":
    main()
