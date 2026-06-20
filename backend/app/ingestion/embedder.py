from openai import OpenAI
from app.config import get_settings
from app.ingestion.chunker import Chunk

settings = get_settings()
client = OpenAI()

# OpenAI can embed up to 2048 inputs per call
# but rate limits mean batching at 100 is safer for dev
BATCH_SIZE = 100


def embed_chunks(chunks: list[Chunk]) -> list[tuple[Chunk, list[float]]]:
    """
    Returns list of (chunk, embedding_vector) pairs.
    Batches calls to avoid rate limits.
    """
    results = []

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c.text for c in batch]

        response = client.embeddings.create(
            input=texts,
            model=settings.embedding_model,
        )
        # response.data is a list of Embedding objects, same order as input
        for chunk, embedding_obj in zip(batch, response.data):
            results.append((chunk, embedding_obj.embedding))

        print(
            f"  Embedded batch {i // BATCH_SIZE + 1} "
            f"({len(batch)} chunks, "
            f"{sum(c.token_count for c in batch)} tokens)"
        )

    return results


def embed_query(text: str, model: str | None = None) -> list[float]:
    """Embed a single query string. Defaults to the configured model."""
    response = client.embeddings.create(
        input=[text],
        model=model or settings.embedding_model,
    )
    return response.data[0].embedding
