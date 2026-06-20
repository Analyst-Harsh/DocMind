from functools import lru_cache

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.ingestion.chunker import Chunk

settings = get_settings()
client = OpenAI()

# OpenAI can embed up to 2048 inputs per call
# but rate limits mean batching at 100 is safer for dev
BATCH_SIZE = 100

EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "BAAI/bge-large-en-v1.5": 1024,
}

LOCAL_MODELS = {"BAAI/bge-large-en-v1.5"}

# BGE models are trained to expect this instruction prefixed onto queries
# (not passages/chunks) for retrieval tasks - see the model card.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def get_embedding_dim(model: str) -> int:
    try:
        return EMBEDDING_DIMENSIONS[model]
    except KeyError:
        raise ValueError(f"unknown-model: {model!r}") from None


@lru_cache(maxsize=4)
def _load_local_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def _embed_chunks_local(
    chunks: list[Chunk], model_name: str
) -> list[tuple[Chunk, list[float]]]:
    model = _load_local_model(model_name)
    texts = [c.text for c in chunks]
    vectors = model.encode(texts, batch_size=32, normalize_embeddings=True)
    return list(zip(chunks, [list(v) for v in vectors], strict=True))


def _embed_chunks_openai(
    chunks: list[Chunk], model_name: str
) -> list[tuple[Chunk, list[float]]]:
    results = []

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c.text for c in batch]

        response = client.embeddings.create(
            input=texts,
            model=model_name,
        )
        # response.data is a list of Embedding objects, same order as input
        for chunk, embedding_obj in zip(batch, response.data, strict=True):
            results.append((chunk, embedding_obj.embedding))

        print(
            f"  Embedded batch {i // BATCH_SIZE + 1} "
            f"({len(batch)} chunks, "
            f"{sum(c.token_count for c in batch)} tokens)"
        )

    return results


def embed_chunks(
    chunks: list[Chunk], model: str | None = None
) -> list[tuple[Chunk, list[float]]]:
    """
    Returns list of (chunk, embedding_vector) pairs.
    Batches calls to avoid rate limits.
    """
    model_name = model or settings.embedding_model
    if model_name in LOCAL_MODELS:
        return _embed_chunks_local(chunks, model_name)
    return _embed_chunks_openai(chunks, model_name)


def embed_query(text: str, model: str | None = None) -> list[float]:
    """Embed a single query string. Defaults to the configured model."""
    model_name = model or settings.embedding_model

    if model_name in LOCAL_MODELS:
        query_text = text
        if model_name in {"BAAI/bge-large-en-v1.5"}:
            query_text = BGE_QUERY_INSTRUCTION + text
        st_model = _load_local_model(model_name)
        vector = st_model.encode([query_text], normalize_embeddings=True)[0]
        return list(vector)

    response = client.embeddings.create(
        input=[text],
        model=model_name,
    )
    return response.data[0].embedding
