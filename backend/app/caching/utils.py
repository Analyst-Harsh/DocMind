import numpy as np


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    vec_a, vec_b = np.array(a), np.array(b)
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def build_key_prefix(embedding_model: str, retrieval_mode: str) -> str:
    """
    Redis key prefix for a (embedding_model, retrieval_mode) namespace,
    e.g. semcache:text-embedding-3-small:dense. Sanitizes "/" in
    HuggingFace-style model ids (e.g. BAAI/bge-large-en-v1.5) the same way
    collection_name_for does for Qdrant collection names.
    """
    safe_model = embedding_model.replace("/", "-")
    return f"semcache:{safe_model}:{retrieval_mode}"
