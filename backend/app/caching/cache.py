import json
import uuid
from dataclasses import asdict
from functools import lru_cache
from time import time

import redis

from app.caching.schema import CachedResponse, CacheEntry, CacheLookupResult
from app.caching.utils import _cosine_similarity, build_key_prefix
from app.config import get_settings


class SemanticCache:
    """
    Redis-backed semantic cache: write() stores a query embedding + full
    pipeline response with a TTL; check() scans every entry in the same
    (embedding_model, retrieval_mode) namespace and returns the closest
    match above the similarity threshold.

    O(n) scan over all entries in the namespace -- acceptable at this
    project's scale (dozens to low hundreds of entries). At production
    scale this would move to Redis's native vector index (RediSearch
    HNSW, FT.SEARCH ... KNN) instead of scanning + scoring in Python; the
    check()/write()/flush() interface here would stay the same, only the
    storage/query implementation underneath would change.
    """

    def __init__(
        self,
        client: redis.Redis | None = None,
        ttl_seconds: int | None = None,
        similarity_threshold: float | None = None,
    ) -> None:
        settings = get_settings()
        self._client = client or redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=True,
        )
        self._ttl_seconds = (
            ttl_seconds
            if ttl_seconds is not None
            else settings.semantic_cache_ttl_seconds
        )
        self._threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.semantic_cache_similarity_threshold
        )

    def ping(self) -> bool:
        return bool(self._client.ping())

    def write(
        self,
        query: str,
        embedding: list[float],
        response: CachedResponse,
        retrieval_mode: str,
        embedding_model: str,
    ) -> None:
        prefix = build_key_prefix(embedding_model, retrieval_mode)
        key = f"{prefix}:{uuid.uuid4()}"
        entry = CacheEntry(
            query=query, embedding=embedding, response=response, ts=time()
        )
        self._client.hset(
            key,
            mapping={
                "query": entry.query,
                "embedding": json.dumps(entry.embedding),
                "response": json.dumps(asdict(entry.response)),
                "ts": str(entry.ts),
            },
        )
        self._client.expire(key, self._ttl_seconds)

    def check(
        self,
        query_embedding: list[float],
        retrieval_mode: str,
        embedding_model: str,
    ) -> CacheLookupResult:
        prefix = build_key_prefix(embedding_model, retrieval_mode)
        best_entry: CacheEntry | None = None
        best_similarity = 0.0

        for key in self._client.scan_iter(match=f"{prefix}:*"):
            fields = self._client.hgetall(key)
            if not fields:
                continue  # expired between SCAN and HGETALL on a real server
            embedding = json.loads(fields["embedding"])
            similarity = _cosine_similarity(query_embedding, embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_entry = CacheEntry(
                    query=str(fields["query"]),
                    embedding=embedding,
                    response=CachedResponse(**json.loads(fields["response"])),
                    ts=float(fields["ts"]),
                )

        if best_entry is not None and best_similarity >= self._threshold:
            return CacheLookupResult(
                hit=best_entry, best_similarity=best_similarity
            )
        return CacheLookupResult(hit=None, best_similarity=best_similarity)

    def flush(self) -> int:
        keys = list(self._client.scan_iter(match="semcache:*"))
        if not keys:
            return 0
        return self._client.delete(*keys)


@lru_cache
def get_semantic_cache() -> SemanticCache:
    return SemanticCache()
