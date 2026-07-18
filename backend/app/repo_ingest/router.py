# app/repo_ingest/router.py
import re
import uuid
from collections.abc import Callable

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.repo_ingest import github
from app.repo_ingest.job_store import IngestJob, get_job_store
from app.repo_ingest.service import run_full_ingest, run_incremental_ingest

router = APIRouter(prefix="/ingest", tags=["ingest"])

REPO_PATTERN = re.compile(r"^[\w.-]+/[\w.-]+$")


class IngestRepoRequest(BaseModel):
    repo: str
    ref: str | None = None


class IngestJobAccepted(BaseModel):
    job_id: str
    job_type: str
    repo: str
    ref: str
    commit_sha: str
    status: str


class JobStatusResponse(BaseModel):
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
    error: str | None
    created_at: float
    started_at: float | None
    finished_at: float | None

    @classmethod
    def from_job(cls, job: IngestJob) -> "JobStatusResponse":
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
            error=job.error,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
        )


def _accept_ingest_job(
    request: IngestRepoRequest,
    background_tasks: BackgroundTasks,
    job_type: str,
    task_fn: Callable[[str, str, str, str], None],
) -> IngestJobAccepted:
    """
    Shared request handling for /ingest/repo and /ingest/files: validate
    repo format, resolve ref -> commit SHA, acquire the per-repo lock,
    create the job record, and schedule task_fn as the BackgroundTasks
    target. job_type only labels the job record -- task_fn (run_full_ingest
    or run_incremental_ingest) is what actually decides how the work runs.
    """
    if not REPO_PATTERN.match(request.repo):
        raise HTTPException(
            status_code=400, detail="repo must be in 'owner/name' form"
        )

    try:
        resolved_ref, commit_sha = github.resolve_commit_sha(
            request.repo, request.ref
        )
    except github.GithubNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except github.GithubAuthError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e
    except github.GithubRateLimitedError as e:
        raise HTTPException(status_code=429, detail=str(e)) from e

    job_store = get_job_store()

    # Mint the job id before creating the record, so a lock conflict never
    # leaves an orphan job that nothing will ever run (see
    # JobStore.create_job's job_id docstring).
    job_id = uuid.uuid4().hex
    holder = job_store.acquire_repo_lock(request.repo, job_id)
    if holder is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"An ingest is already running for {request.repo!r} "
                f"(job_id={holder!r})"
            ),
        )

    job = job_store.create_job(
        job_type, request.repo, resolved_ref, commit_sha, job_id=job_id
    )
    background_tasks.add_task(
        task_fn, job.job_id, request.repo, resolved_ref, commit_sha
    )

    return IngestJobAccepted(
        job_id=job.job_id,
        job_type=job.job_type,
        repo=job.repo,
        ref=job.ref,
        commit_sha=job.commit_sha,
        status=job.status,
    )


@router.post("/repo", response_model=IngestJobAccepted, status_code=202)
def ingest_repo(
    request: IngestRepoRequest, background_tasks: BackgroundTasks
) -> IngestJobAccepted:
    return _accept_ingest_job(request, background_tasks, "full", run_full_ingest)


@router.post("/files", response_model=IngestJobAccepted, status_code=202)
def ingest_files(
    request: IngestRepoRequest, background_tasks: BackgroundTasks
) -> IngestJobAccepted:
    """
    Incremental counterpart to /ingest/repo: same request/response shape,
    but the server computes its own diff against the last watermark
    (run_incremental_ingest) instead of trusting a forwarded webhook
    payload -- GitHub push payloads are lossy (capped commit lists, force
    pushes, retried/out-of-order deliveries), so DocMind re-derives the
    change set itself rather than relying on what the caller sent.
    """
    return _accept_ingest_job(
        request, background_tasks, "incremental", run_incremental_ingest
    )


@router.get("/status/{job_id}", response_model=JobStatusResponse)
def ingest_status(job_id: str) -> JobStatusResponse:
    job = get_job_store().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id!r}")
    return JobStatusResponse.from_job(job)
