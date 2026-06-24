import math

import pytest

from app.caching.cache import SemanticCache
from app.caching.schema import CachedResponse

MODEL = "text-embedding-3-small"
CACHED_EMBEDDING = [1.0, 0.0]


def _vector_at_cosine(similarity: float) -> list[float]:
    """A unit vector whose cosine similarity to [1.0, 0.0] is exactly `similarity`."""
    theta = math.acos(similarity)
    return [math.cos(theta), math.sin(theta)]


def test_paraphrase_pairs_above_threshold_hit(fake_redis):
    cache = SemanticCache(client=fake_redis, similarity_threshold=0.95)
    cache.write(
        "original question", CACHED_EMBEDDING,
        CachedResponse(answer="a", sources=[], cost_usd=0.0),
        retrieval_mode="dense", embedding_model=MODEL,
    )

    for similarity in (0.99, 0.97):
        result = cache.check(
            _vector_at_cosine(similarity), retrieval_mode="dense", embedding_model=MODEL
        )
        assert result.hit is not None, f"expected a hit at similarity={similarity}"


def test_non_paraphrase_pairs_below_threshold_miss(fake_redis):
    cache = SemanticCache(client=fake_redis, similarity_threshold=0.95)
    cache.write(
        "original question", CACHED_EMBEDDING,
        CachedResponse(answer="a", sources=[], cost_usd=0.0),
        retrieval_mode="dense", embedding_model=MODEL,
    )

    for similarity in (0.50, 0.30):
        result = cache.check(
            _vector_at_cosine(similarity), retrieval_mode="dense", embedding_model=MODEL
        )
        assert result.hit is None, f"expected a miss at similarity={similarity}"
        assert result.best_similarity == pytest.approx(similarity, abs=1e-6)
