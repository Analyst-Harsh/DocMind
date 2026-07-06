# scripts/ingest_tables.py
"""
Extract tables from corpus PDFs, embed as KV text, and index into the
'multimodal' Qdrant collection for RAG comparison with text-based pipelines.

Usage (from backend/):
  python -m scripts.ingest_tables
  python -m scripts.ingest_tables --embedding-model BAAI/bge-large-en-v1.5
  python -m scripts.ingest_tables --hybrid
"""

import argparse

from app.config import get_settings
from app.ingestion.chunker.table_chunker import TableChunker
from app.ingestion.embedder import embed_chunks, get_embedding_dim
from app.ingestion.indexer import (
    ensure_collection,
    ensure_hybrid_collection,
    get_qdrant_client,
    multimodal_collection_name,
    upsert_table_chunks,
    upsert_table_chunks_hybrid,
)
from app.ingestion.loader import load_all_documents
from app.ingestion.sparse_embedder import embed_chunks_sparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract PDF tables and index into the multimodal Qdrant collection."
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model (default: settings.embedding_model). "
        "Pass BAAI/bge-large-en-v1.5 to use local sentence-transformers.",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Build a hybrid (dense+BM25 sparse) collection.",
    )
    args = parser.parse_args()

    settings = get_settings()
    model = args.embedding_model or settings.embedding_model

    print("=== DocMind Table Ingestion ===")
    docs = load_all_documents()
    pdf_docs = [d for d in docs if d.doc_type == "pdf"]
    print(
        f"Loaded {len(docs)} documents "
        f"({len(pdf_docs)} PDFs for table extraction)"
    )

    chunker = TableChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_documents(pdf_docs)

    table_n = sum(1 for c in chunks if c.is_table)
    text_n = len(chunks) - table_n
    print(f"\n  Table chunks : {table_n}")
    print(f"  Text chunks  : {text_n}")
    print(f"  Total        : {len(chunks)}")

    if not chunks:
        print("No content found in corpus PDFs. Exiting.")
        return

    total_tokens = sum(c.token_count for c in chunks)
    print(
        f"Tokens to embed: {total_tokens:,} "
        f"(~${(total_tokens / 1_000_000) * 0.02:.4f})"
    )

    client = get_qdrant_client()
    chunk_embeddings = embed_chunks(chunks, model=model)

    if args.hybrid:
        print("Building hybrid (dense+BM25) variant...")
        sparse_embeddings = embed_chunks_sparse(chunks)
        collection = multimodal_collection_name(model, hybrid=True)
        ensure_hybrid_collection(
            client, collection, vector_size=get_embedding_dim(model)
        )
        upsert_table_chunks_hybrid(
            client, collection, chunk_embeddings, sparse_embeddings
        )
    else:
        collection = multimodal_collection_name(model)
        ensure_collection(
            client, collection, vector_size=get_embedding_dim(model)
        )
        upsert_table_chunks(client, collection, chunk_embeddings)

    info = client.get_collection(collection)
    print(f"\n{collection}: {info.points_count} points")
    print("\n=== Table ingestion complete ===")


if __name__ == "__main__":
    main()
