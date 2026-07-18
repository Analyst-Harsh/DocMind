"""Test double for redis.Redis covering the extra primitives job_store.py
needs beyond app.caching.tests.fakes.FakeRedis (hset/hgetall/expire, plus
set/get/delete for locks and watermarks). Same rationale as that fake: this
codebase mocks every external service at the boundary rather than requiring
a live Redis for tests.
"""


class FakeRedis:
    def __init__(self) -> None:
        self._hashes: dict[str, dict[str, str]] = {}
        self._strings: dict[str, str] = {}

    def hset(self, key: str, mapping: dict[str, str]) -> None:
        self._hashes[key] = dict(mapping)

    def hgetall(self, key: str) -> dict[str, str]:
        return dict(self._hashes.get(key, {}))

    def expire(self, key: str, seconds: int) -> None:
        pass  # TTL expiry isn't simulated -- no test depends on it firing

    def set(
        self,
        name: str,
        value: str,
        nx: bool = False,
        ex: int | None = None,
    ) -> bool | None:
        if nx and name in self._strings:
            return None
        self._strings[name] = value
        return True

    def get(self, name: str) -> str | None:
        return self._strings.get(name)

    def delete(self, *keys: str) -> int:
        count = 0
        for key in keys:
            if key in self._strings:
                del self._strings[key]
                count += 1
            if key in self._hashes:
                del self._hashes[key]
                count += 1
        return count
