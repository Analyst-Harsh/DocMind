from unittest.mock import patch

import pytest

from app.mcp import tools
from app.mcp.schemas import IngestAccepted, JobStatus
from app.query.service import (
    NoRelevantChunksError,
    QueryResult,
    RepoNotIngestedError,
)
from app.repo_ingest import github
from app.repo_ingest.job_store import IngestJob
from app.repo_ingest.service import (
    InvalidRepoFormatError,
    RepoLockedError,
    run_full_ingest,
    run_incremental_ingest,
)


def _job(*, job_type: str = "full", status: str = "pending") -> IngestJob:
    return IngestJob(
        job_id="job-1",
        job_type=job_type,
        repo="octo/hello",
        ref="main",
        commit_sha="abc123",
        status=status,
    )


@patch("app.mcp.tools.fire_and_forget")
@patch("app.mcp.tools.prepare_ingest_job")
def test_ingest_repo_success(mock_prepare, mock_fire):
    mock_prepare.return_value = _job()

    result = tools.ingest_repo(repo="octo/hello")

    assert isinstance(result, IngestAccepted)
    assert result.job_id == "job-1"
    assert result.status == "pending"
    mock_fire.assert_called_once_with(
        run_full_ingest, "job-1", "octo/hello", "main", "abc123"
    )


@patch("app.mcp.tools.fire_and_forget")
@patch("app.mcp.tools.prepare_ingest_job")
def test_sync_repo_incremental_uses_run_incremental_ingest(mock_prepare, mock_fire):
    mock_prepare.return_value = _job(job_type="incremental")

    result = tools.sync_repo_incremental(repo="octo/hello")

    assert result.job_type == "incremental"
    mock_fire.assert_called_once_with(
        run_incremental_ingest, "job-1", "octo/hello", "main", "abc123"
    )


@patch("app.mcp.tools.prepare_ingest_job")
def test_ingest_repo_maps_invalid_repo_format(mock_prepare):
    mock_prepare.side_effect = InvalidRepoFormatError("not-a-repo")

    with pytest.raises(ValueError, match="owner/name"):
        tools.ingest_repo(repo="not-a-repo")


@patch("app.mcp.tools.prepare_ingest_job")
def test_ingest_repo_maps_github_not_found(mock_prepare):
    mock_prepare.side_effect = github.GithubNotFoundError("octo/missing not found")

    with pytest.raises(ValueError, match="not found"):
        tools.ingest_repo(repo="octo/missing")


@patch("app.mcp.tools.prepare_ingest_job")
def test_ingest_repo_maps_repo_locked(mock_prepare):
    mock_prepare.side_effect = RepoLockedError("octo/hello", "other-job")

    with pytest.raises(ValueError, match="other-job"):
        tools.ingest_repo(repo="octo/hello")


@patch("app.mcp.tools.get_job_store")
def test_get_ingest_status_returns_job_status(mock_get_store):
    mock_get_store.return_value.get_job.return_value = _job(status="completed")

    result = tools.get_ingest_status(job_id="job-1")

    assert isinstance(result, JobStatus)
    assert result.status == "completed"


@patch("app.mcp.tools.get_job_store")
def test_get_ingest_status_unknown_job_raises(mock_get_store):
    mock_get_store.return_value.get_job.return_value = None

    with pytest.raises(ValueError, match="job-missing"):
        tools.get_ingest_status(job_id="job-missing")


@patch("app.mcp.tools.run_query")
def test_query_repo_success(mock_run_query):
    mock_run_query.return_value = QueryResult(
        answer="it's in src/main.py",
        sources=[{"chunk_id": "c1"}],
        cost_usd=0.001,
        latency_ms=10,
        trace_id="trace-1",
        cache_hit=False,
    )

    result = tools.query_repo(repo="octo/hello", question="where is main?")

    assert isinstance(result, QueryResult)
    assert result.answer == "it's in src/main.py"
    mock_run_query.assert_called_once_with(
        question="where is main?", top_k=5, hybrid=True, repo="octo/hello"
    )


@patch("app.mcp.tools.run_query")
def test_query_repo_maps_repo_not_ingested(mock_run_query):
    mock_run_query.side_effect = RepoNotIngestedError("octo/missing")

    with pytest.raises(ValueError, match="octo/missing"):
        tools.query_repo(repo="octo/missing", question="anything")


@patch("app.mcp.tools.run_query")
def test_query_repo_maps_no_relevant_chunks(mock_run_query):
    mock_run_query.side_effect = NoRelevantChunksError()

    with pytest.raises(ValueError, match="No relevant documents found"):
        tools.query_repo(repo="octo/hello", question="anything")


def test_tool_decorator_sanitizes_unexpected_exceptions():
    @tools._tool
    def flaky():
        raise RuntimeError("qdrant host unreachable at 10.0.0.5:6333")

    with (
        patch.object(tools.log, "exception") as mock_log_exception,
        pytest.raises(RuntimeError) as exc_info,
    ):
        flaky()

    # the original exception's message must not leak into the client-facing
    # error -- only a generic, sanitized message does
    assert "10.0.0.5" not in str(exc_info.value)
    assert "check server logs" in str(exc_info.value)
    mock_log_exception.assert_called_once()


def test_tool_decorator_passes_through_domain_errors_as_value_error():
    @tools._tool
    def raises_domain_error():
        raise RepoNotIngestedError("octo/hello")

    with pytest.raises(ValueError, match="octo/hello"):
        raises_domain_error()


@patch("app.mcp.tools.fire_and_forget")
@patch("app.mcp.tools.prepare_ingest_job")
def test_tool_decorator_logs_every_call(mock_prepare, mock_fire):
    mock_prepare.return_value = _job()

    with patch.object(tools.log, "info") as mock_log_info:
        tools.ingest_repo(repo="octo/hello")

    mock_log_info.assert_called_once()
    args, kwargs = mock_log_info.call_args
    assert args[0] == "mcp_tool_call"
    assert kwargs["tool"] == "ingest_repo"
