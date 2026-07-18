import numpy as np


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    vec_a, vec_b = np.array(a), np.array(b)
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def build_key_prefix(
    embedding_model: str, retrieval_mode: str, scope: str = "docs"
) -> str:
    """
    Redis key prefix for a (scope, embedding_model, retrieval_mode)
    namespace, e.g. semcache:docs:text-embedding-3-small:dense or
    semcache:octo-hello:text-embedding-3-small:hybrid for a repo-scoped
    query. scope defaults to "docs" (the fixed corpus) so an unscoped
    caller's behavior is unchanged; a repo query passes its repo slug so a
    docs-corpus cache entry can never be served for a repo question (or
    vice versa) even if their embeddings happen to be similar. Sanitizes
    "/" the same way collection_name_for does for Qdrant collection names.
    """
    safe_scope = scope.replace("/", "-")
    safe_model = embedding_model.replace("/", "-")
    return f"semcache:{safe_scope}:{safe_model}:{retrieval_mode}"
