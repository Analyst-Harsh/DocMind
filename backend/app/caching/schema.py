from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CachedResponse:
    answer: str
    sources: list[dict[str, Any]]
    cost_usd: float


@dataclass
class CacheEntry:
    query: str
    embedding: list[float]
    response: CachedResponse
    ts: float


@dataclass
class CacheLookupResult:
    hit: CacheEntry | None
    best_similarity: float
