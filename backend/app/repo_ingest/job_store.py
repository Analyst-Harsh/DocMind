# app/repo_ingest/job_store.py
"""
Redis-backed store for repo-ingestion job records, plus the per-repo lock
and watermark that make ingestion safe to run concurrently and
incrementally. Job status must survive process restarts and work across
multiple uvicorn workers -- a module-level dict wouldn't -- so this follows
the same Redis-backed, injectable-client pattern as app.caching.cache's
SemanticCache.
"""

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import redis

from app.config import get_settings

# Crash backstop: if a job dies without releasing the lock, it expires
# instead of wedging the repo forever. Successful/failed runs release it
# explicitly well before this fires.
REPO_LOCK_TTL_SECONDS = 1800


@dataclass
class IngestJob:
    job_id: str
    job_type: str  # "full" | "incremental"
    repo: str
    ref: str
    commit_sha: str
    status: str = "pending"  # pending | running | completed | failed
    files_ingested: int = 0
    chunks_upserted: int = 0
    chunks_deleted: int = 0
    tokens_embedded: int = 0
    files_added: int = 0
    files_modified: int = 0
    files_removed: int = 0
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None


def _job_key(job_id: str) -> str:
    return f"ingest:job:{job_id}"


def _lock_key(repo: str) -> str:
    return f"ingest:lock:{repo}"


def _watermark_key(repo: str) -> str:
    return f"ingest:watermark:{repo}"


class JobStore:
    def __init__(
        self,
        client: redis.Redis | None = None,
        job_ttl_seconds: int | None = None,
    ) -> None:
        settings = get_settings()
        self._client = client or redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=True,
        )
        self._job_ttl_seconds = (
            job_ttl_seconds
            if job_ttl_seconds is not None
            else settings.ingest_job_ttl_seconds
        )

    # -- job lifecycle -------------------------------------------------

    def create_job(
        self,
        job_type: str,
        repo: str,
        ref: str,
        commit_sha: str,
        job_id: str | None = None,
    ) -> IngestJob:
        """
        job_id is normally auto-generated. The router passes one explicitly
        when it already had to mint an id to acquire the per-repo lock
        before creating the job record -- so a lock conflict (409) never
        leaves behind an orphan job that nothing will ever run.
        """
        job = IngestJob(
            job_id=job_id or uuid.uuid4().hex,
            job_type=job_type,
            repo=repo,
            ref=ref,
            commit_sha=commit_sha,
        )
        self._save(job)
        return job

    def get_job(self, job_id: str) -> IngestJob | None:
        raw = self._client.hgetall(_job_key(job_id))
        if not raw:
            return None
        return IngestJob(**json.loads(raw["data"]))

    def mark_running(self, job_id: str) -> None:
        job = self._require(job_id)
        job.status = "running"
        job.started_at = time.time()
        self._save(job)

    def mark_completed(self, job_id: str, **counters: int) -> None:
        """counters overwrites any matching IngestJob field, e.g.
        mark_completed(job_id, chunks_upserted=42, chunks_deleted=3)."""
        job = self._require(job_id)
        job.status = "completed"
        job.finished_at = time.time()
        for key, value in counters.items():
            setattr(job, key, value)
        self._save(job)

    def mark_failed(self, job_id: str, error: str) -> None:
        job = self._require(job_id)
        job.status = "failed"
        job.finished_at = time.time()
        job.error = error
        self._save(job)

    def _require(self, job_id: str) -> IngestJob:
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(f"unknown ingest job: {job_id}")
        return job

    def _save(self, job: IngestJob) -> None:
        key = _job_key(job.job_id)
        self._client.hset(key, mapping={"data": json.dumps(asdict(job))})
        self._client.expire(key, self._job_ttl_seconds)

    # -- per-repo lock ---------------------------------------------------
    # Serializes ingests for one repo: a bulk sweep racing an incremental
    # upsert (or two bulk runs racing each other) could delete the other's
    # fresh points, so only one ingest per repo may run at a time.

    def acquire_repo_lock(self, repo: str, job_id: str) -> str | None:
        """Returns None on success. On conflict, returns the job_id
        currently holding the lock, so the caller can report it (e.g. in a
        409 response)."""
        acquired = self._client.set(
            _lock_key(repo), job_id, nx=True, ex=REPO_LOCK_TTL_SECONDS
        )
        if acquired:
            return None
        holder = self._client.get(_lock_key(repo))
        return None if holder is None else str(holder)

    def release_repo_lock(self, repo: str, job_id: str) -> None:
        """Only releases the lock if this job still holds it -- otherwise a
        job that outlived its TTL could release a newer job's lock out from
        under it."""
        key = _lock_key(repo)
        if self._client.get(key) == job_id:
            self._client.delete(key)

    # -- watermark ---------------------------------------------------------
    # Last commit SHA successfully bulk-ingested for a repo. Written on
    # every successful full or incremental run; read by the incremental
    # path to compute a diff base (see app/repo_ingest/service.py).

    def get_watermark(self, repo: str) -> str | None:
        value = self._client.get(_watermark_key(repo))
        return None if value is None else str(value)

    def set_watermark(self, repo: str, sha: str) -> None:
        self._client.set(_watermark_key(repo), sha)


@lru_cache
def get_job_store() -> JobStore:
    return JobStore()
