from app.caching.cache import SemanticCache
from app.caching.schema import CachedResponse

MODEL = "text-embedding-3-small"


def test_write_then_check_returns_matching_entry(fake_redis):
    cache = SemanticCache(client=fake_redis, similarity_threshold=0.9)
    embedding = [1.0, 0.0, 0.0]
    response = CachedResponse(
        answer="RAG combines retrieval and generation.",
        sources=[{"chunk_id": "c0"}],
        cost_usd=0.0021,
    )

    cache.write(
        "What is RAG?", embedding, response, retrieval_mode="dense", embedding_model=MODEL
    )
    result = cache.check(embedding, retrieval_mode="dense", embedding_model=MODEL)

    assert result.hit is not None
    assert result.hit.query == "What is RAG?"
    assert result.hit.response == response
    assert result.best_similarity == 1.0


def test_check_returns_miss_on_empty_cache(fake_redis):
    cache = SemanticCache(client=fake_redis, similarity_threshold=0.9)

    result = cache.check([1.0, 0.0, 0.0], retrieval_mode="dense", embedding_model=MODEL)

    assert result.hit is None
    assert result.best_similarity == 0.0


def test_check_does_not_match_across_retrieval_modes(fake_redis):
    cache = SemanticCache(client=fake_redis, similarity_threshold=0.9)
    embedding = [1.0, 0.0, 0.0]
    cache.write(
        "What is RAG?",
        embedding,
        CachedResponse(answer="a", sources=[], cost_usd=0.0),
        retrieval_mode="dense",
        embedding_model=MODEL,
    )

    result = cache.check(embedding, retrieval_mode="hybrid_rerank", embedding_model=MODEL)

    assert result.hit is None


def test_flush_deletes_all_entries_and_returns_count(fake_redis):
    cache = SemanticCache(client=fake_redis, similarity_threshold=0.9)
    cache.write(
        "q1", [1.0, 0.0], CachedResponse(answer="a", sources=[], cost_usd=0.0),
        retrieval_mode="dense", embedding_model=MODEL,
    )
    cache.write(
        "q2", [0.0, 1.0], CachedResponse(answer="b", sources=[], cost_usd=0.0),
        retrieval_mode="hybrid", embedding_model=MODEL,
    )

    flushed = cache.flush()

    assert flushed == 2
    assert cache.check([1.0, 0.0], retrieval_mode="dense", embedding_model=MODEL).hit is None
