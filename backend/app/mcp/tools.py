# app/mcp/tools.py
from functools import wraps

import structlog

from app.mcp.instance import mcp
from app.mcp.schemas import IngestAccepted, JobStatus
from app.mcp.tasks import fire_and_forget
from app.query.service import (
    NoRelevantChunksError,
    QueryResult,
    RepoNotIngestedError,
    run_query,
)
from app.repo_ingest import github
from app.repo_ingest.job_store import get_job_store
from app.repo_ingest.service import (
    InvalidRepoFormatError,
    RepoLockedError,
    prepare_ingest_job,
    run_full_ingest,
    run_incremental_ingest,
)

log = structlog.get_logger()

_DOMAIN_ERRORS = (
    InvalidRepoFormatError,
    github.GithubNotFoundError,
    github.GithubAuthError,
    github.GithubRateLimitedError,
    RepoLockedError,
    RepoNotIngestedError,
    NoRelevantChunksError,
)


def _tool(fn):
    """Wraps a tool implementation with structured logging and a
    defense-in-depth error boundary: known domain exceptions become a
    clean ValueError (safe to show an LLM/end user); anything unexpected
    is logged in full server-side and replaced with a generic message, so
    internal details (hostnames, ports, library internals) never leak
    into a tool's error content."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        log.info("mcp_tool_call", tool=fn.__name__, kwargs=kwargs)
        try:
            return fn(*args, **kwargs)
        except ValueError:
            # Already the target "client-safe" type (e.g. get_ingest_status's
            # hand-raised "unknown job_id" message) -- propagate unchanged
            # rather than falling into the generic catch-all below.
            raise
        except _DOMAIN_ERRORS as e:
            raise ValueError(str(e)) from e
        except Exception:
            log.exception(
                "mcp_tool_call_failed", tool=fn.__name__, kwargs=kwargs
            )
            raise RuntimeError(
                "Internal error while running this tool -- check server logs."
            ) from None

    return wrapper


def _accept_ingest(
    repo: str, ref: str | None, job_type: str, task_fn
) -> IngestAccepted:
    job = prepare_ingest_job(repo, ref, job_type, get_job_store)
    fire_and_forget(task_fn, job.job_id, repo, job.ref, job.commit_sha)
    return IngestAccepted(
        job_id=job.job_id,
        job_type=job.job_type,
        repo=job.repo,
        ref=job.ref,
        commit_sha=job.commit_sha,
        status=job.status,
    )


@mcp.tool
@_tool
def ingest_repo(repo: str, ref: str | None = None) -> IngestAccepted:
    """Kick off a FULL ingest of a GitHub repo ('owner/name') into its own
    Qdrant collection. Cost/latency warning: embeds every ingestable file
    via OpenAI -- can take minutes and real API spend for a large repo.
    Returns immediately with a job_id; poll get_ingest_status(job_id) for
    completion. If the repo is already ingested and you only need to catch
    up on recent commits, use sync_repo_incremental instead -- far cheaper.
    ref defaults to the repo's default branch if omitted."""
    return _accept_ingest(repo, ref, "full", run_full_ingest)


@mcp.tool
@_tool
def sync_repo_incremental(repo: str, ref: str | None = None) -> IngestAccepted:
    """Incrementally re-ingest a GitHub repo already ingested via
    ingest_repo, diffing against the last successfully ingested commit --
    much cheaper than ingest_repo for a repo that's already indexed. Falls
    back internally to a full re-ingest if there's no prior watermark or
    the diff can't be safely applied (e.g. force-push). Returns
    immediately with a job_id; poll get_ingest_status(job_id)."""
    return _accept_ingest(repo, ref, "incremental", run_incremental_ingest)


@mcp.tool
@_tool
def get_ingest_status(job_id: str) -> JobStatus:
    """Check the status of a job returned by ingest_repo or
    sync_repo_incremental. status is pending/running/completed/failed; on
    failure, `error` has the reason."""
    job = get_job_store().get_job(job_id)
    if job is None:
        raise ValueError(f"Unknown job_id: {job_id!r}")
    return JobStatus.from_job(job)


@mcp.tool
@_tool
def query_repo(repo: str, question: str, top_k: int = 5) -> QueryResult:
    """Ask a question against a GitHub repo already ingested via
    ingest_repo (call get_ingest_status first if unsure it finished).
    Always uses hybrid retrieval + reranking. Raises a clear error if the
    repo was never ingested or no relevant code chunks are found -- in
    either case, do not retry blindly; ingest the repo first or rephrase."""
    return run_query(question=question, top_k=top_k, hybrid=True, repo=repo)
