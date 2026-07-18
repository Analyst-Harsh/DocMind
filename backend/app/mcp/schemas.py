# app/mcp/schemas.py
"""
Plain pydantic output models for the MCP tools. Deliberately not imported
from app.repo_ingest.router/main.py (those modules import FastAPI) -- this
is what keeps app/mcp decoupled from the HTTP layer, per the design.
"""

from pydantic import BaseModel

from app.repo_ingest.job_store import IngestJob


class IngestAccepted(BaseModel):
    job_id: str
    job_type: str
    repo: str
    ref: str
    commit_sha: str
    status: str


class JobStatus(BaseModel):
    job_id: str
    job_type: str
    status: str
    repo: str
    ref: str
    commit_sha: str
    files_ingested: int
    chunks_upserted: int
    chunks_deleted: int
    tokens_embedded: int
    files_added: int
    files_modified: int
    files_removed: int
    error: str | None
    created_at: float
    started_at: float | None
    finished_at: float | None

    @classmethod
    def from_job(cls, job: IngestJob) -> "JobStatus":
        # Listed field-by-field, not cls(**asdict(job)) -- IngestJob may
        # gain fields over time, and blindly forwarding all of them would
        # either crash (extra keys pydantic doesn't know) or silently leak
        # new internal fields into the tool's output. Explicit is safer.
        return cls(
            job_id=job.job_id,
            job_type=job.job_type,
            status=job.status,
            repo=job.repo,
            ref=job.ref,
            commit_sha=job.commit_sha,
            files_ingested=job.files_ingested,
            chunks_upserted=job.chunks_upserted,
            chunks_deleted=job.chunks_deleted,
            tokens_embedded=job.tokens_embedded,
            files_added=job.files_added,
            files_modified=job.files_modified,
            files_removed=job.files_removed,
            error=job.error,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
        )
