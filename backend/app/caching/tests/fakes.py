"""Test double for redis.Redis -- just the methods SemanticCache uses
(hset/hgetall/expire/scan_iter/delete/ping). Used in tests instead of a
real Redis server, matching this codebase's existing convention of mocking
every external service at the boundary (Qdrant, OpenAI, sentence-
transformers are all mocked in their respective tests too).
"""

import fnmatch
from collections.abc import Iterator


class FakeRedis:
    def __init__(self) -> None:
        self._hashes: dict[str, dict[str, str]] = {}

    def hset(self, key: str, mapping: dict[str, str]) -> None:
        self._hashes[key] = dict(mapping)

    def hgetall(self, key: str) -> dict[str, str]:
        return dict(self._hashes.get(key, {}))

    def expire(self, key: str, seconds: int) -> None:
        pass  # TTL expiry isn't simulated -- no test depends on it firing

    def scan_iter(self, match: str | None = None) -> Iterator[str]:
        for key in list(self._hashes.keys()):
            if match is None or fnmatch.fnmatch(key, match):
                yield key

    def delete(self, *keys: str) -> int:
        count = 0
        for key in keys:
            if key in self._hashes:
                del self._hashes[key]
                count += 1
        return count

    def ping(self) -> bool:
        return True
