# scripts/ingest_graph.py
"""
Extract entities/relationships from the corpus (LLM-based) and write them,
along with chunk text + embeddings, into Neo4j as a knowledge graph.

Usage:
  python -m scripts.ingest_graph
  python -m scripts.ingest_graph --limit 10
  python -m scripts.ingest_graph --embedding-model text-embedding-3-large
"""

import argparse

from app.config import get_settings
from app.graph.client import get_neo4j_driver
from app.graph.extractor import extract_entities_and_relations
from app.graph.schema import ensure_schema
from app.graph.writer import write_chunk_graph
from app.ingestion.chunker import ChunkStrategy, get_chunker
from app.ingestion.embedder import embed_chunks, get_embedding_dim
from app.ingestion.loader import load_all_documents

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Neo4j knowledge graph from the corpus."
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model for Chunk.embedding (default: "
        "settings.embedding_model). Must be an OpenAI model - the local "
        "sentence-transformers path isn't wired up for this script.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N chunks (for cheap smoke-testing "
        "before a full LLM-extraction run over the whole corpus).",
    )
    args = parser.parse_args()

    settings = get_settings()
    model = args.embedding_model or settings.embedding_model

    print("=== DocMind Knowledge Graph Ingestion ===")
    docs = load_all_documents()
    print(f"Loaded {len(docs)} documents")

    chunker = get_chunker(
        ChunkStrategy.RECURSIVE,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = chunker.chunk_documents(docs)
    if args.limit is not None:
        chunks = chunks[: args.limit]
    print(f"Chunks to process: {len(chunks)}")

    docs_by_id = {doc.doc_id: doc for doc in docs}

    driver = get_neo4j_driver()
    ensure_schema(vector_dim=get_embedding_dim(model), driver=driver)

    print("Embedding chunks...")
    chunk_embeddings = embed_chunks(chunks, model=model)

    total_entities = 0
    total_relations = 0
    for i, (chunk, embedding) in enumerate(chunk_embeddings, start=1):
        extraction = extract_entities_and_relations(chunk, model=settings.llm_model)
        write_chunk_graph(
            document=docs_by_id[chunk.doc_id],
            chunk=chunk,
            embedding=embedding,
            extraction=extraction,
            driver=driver,
        )
        total_entities += len(extraction.entities)
        total_relations += len(extraction.relations)
        print(
            f"  [{i}/{len(chunk_embeddings)}] {chunk.chunk_id}: "
            f"{len(extraction.entities)} entities, "
            f"{len(extraction.relations)} relations"
        )

    print(
        f"\n=== Ingestion complete: {len(chunk_embeddings)} chunks, "
        f"{total_entities} entity mentions, {total_relations} relations "
        "extracted ==="
    )


if __name__ == "__main__":
    main()
