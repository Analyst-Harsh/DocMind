# scripts/inspect_chunking.py
"""
Run all three chunking strategies over the corpus and print stats + sample
chunks, so chunk boundaries can be eyeballed before Day 2's eval harness.
Usage: python -m scripts.inspect_chunking
"""

from app.ingestion.chunker import CHUNKER_REGISTRY, get_chunker
from app.ingestion.loader import load_all_documents


def main() -> None:
    docs = load_all_documents()

    for strategy_name in CHUNKER_REGISTRY:
        print(f"\n{'=' * 60}\nStrategy: {strategy_name}\n{'=' * 60}")
        chunker = get_chunker(strategy_name, chunk_size=300, chunk_overlap=30)
        all_chunks = chunker.chunk_documents(docs)

        token_counts = [c.token_count for c in all_chunks]
        print(f"Total chunks: {len(all_chunks)}")
        print(f"Avg tokens/chunk: {sum(token_counts) / len(token_counts):.0f}")
        print(f"Min/Max tokens: {min(token_counts)}/{max(token_counts)}")

        seen_types = set()
        for chunk in all_chunks:
            if chunk.doc_type not in seen_types:
                seen_types.add(chunk.doc_type)
                print(
                    f"\n--- Sample [{chunk.doc_type}] from {chunk.doc_id} ---"
                )
                print(chunk.text)
                print("...")


if __name__ == "__main__":
    main()
