from unittest.mock import patch

from fastapi.testclient import TestClient

from app.repo_ingest.job_store import JobStore
from app.repo_ingest.tests.fakes import FakeRedis
from main import app

client = TestClient(app)


def _job_store():
    return JobStore(client=FakeRedis(), job_ttl_seconds=3600)


@patch("app.repo_ingest.router.run_full_ingest")
@patch("app.repo_ingest.router.get_job_store")
@patch("app.repo_ingest.router.github.resolve_commit_sha")
def test_ingest_repo_returns_202(mock_resolve, mock_get_store, mock_run):
    mock_resolve.return_value = ("main", "abc123")
    store = _job_store()
    mock_get_store.return_value = store

    resp = client.post("/ingest/repo", json={"repo": "octo/hello", "ref": "main"})

    assert resp.status_code == 202
    body = resp.json()
    assert body["repo"] == "octo/hello"
    assert body["ref"] == "main"
    assert body["commit_sha"] == "abc123"
    assert body["status"] == "pending"
    assert body["job_type"] == "full"
    mock_run.assert_called_once_with(body["job_id"], "octo/hello", "main", "abc123")


@patch("app.repo_ingest.router.run_full_ingest")
@patch("app.repo_ingest.router.get_job_store")
@patch("app.repo_ingest.router.github.resolve_commit_sha")
def test_ingest_repo_defaults_ref_to_resolved_default_branch(
    mock_resolve, mock_get_store, mock_run
):
    mock_resolve.return_value = ("trunk", "def456")
    mock_get_store.return_value = _job_store()

    resp = client.post("/ingest/repo", json={"repo": "octo/hello"})

    assert resp.status_code == 202
    assert resp.json()["ref"] == "trunk"
    mock_resolve.assert_called_once_with("octo/hello", None)


def test_ingest_repo_rejects_malformed_repo():
    resp = client.post("/ingest/repo", json={"repo": "not-a-valid-repo"})
    assert resp.status_code == 400


@patch("app.repo_ingest.router.github.resolve_commit_sha")
def test_ingest_repo_github_not_found_maps_to_404(mock_resolve):
    from app.repo_ingest.github import GithubNotFoundError

    mock_resolve.side_effect = GithubNotFoundError("nope")
    resp = client.post("/ingest/repo", json={"repo": "octo/missing"})
    assert resp.status_code == 404


@patch("app.repo_ingest.router.github.resolve_commit_sha")
def test_ingest_repo_github_auth_error_maps_to_401(mock_resolve):
    from app.repo_ingest.github import GithubAuthError

    mock_resolve.side_effect = GithubAuthError("bad token")
    resp = client.post("/ingest/repo", json={"repo": "octo/private"})
    assert resp.status_code == 401


@patch("app.repo_ingest.router.github.resolve_commit_sha")
def test_ingest_repo_github_rate_limited_maps_to_429(mock_resolve):
    from app.repo_ingest.github import GithubRateLimitedError

    mock_resolve.side_effect = GithubRateLimitedError("slow down")
    resp = client.post("/ingest/repo", json={"repo": "octo/hello"})
    assert resp.status_code == 429


@patch("app.repo_ingest.router.run_full_ingest")
@patch("app.repo_ingest.router.get_job_store")
@patch("app.repo_ingest.router.github.resolve_commit_sha")
def test_ingest_repo_conflict_when_already_locked(
    mock_resolve, mock_get_store, mock_run
):
    mock_resolve.return_value = ("main", "abc123")
    store = _job_store()
    store.acquire_repo_lock("octo/hello", "existing-job-id")
    mock_get_store.return_value = store

    resp = client.post("/ingest/repo", json={"repo": "octo/hello", "ref": "main"})

    assert resp.status_code == 409
    assert "existing-job-id" in resp.json()["detail"]
    mock_run.assert_not_called()


@patch("app.repo_ingest.router.get_job_store")
def test_ingest_status_returns_job(mock_get_store):
    store = _job_store()
    job = store.create_job("full", "octo/hello", "main", "abc123")
    store.mark_completed(job.job_id, files_ingested=5, chunks_upserted=20)
    mock_get_store.return_value = store

    resp = client.get(f"/ingest/status/{job.job_id}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["files_ingested"] == 5
    assert body["chunks_upserted"] == 20


@patch("app.repo_ingest.router.get_job_store")
def test_ingest_status_unknown_job_returns_404(mock_get_store):
    mock_get_store.return_value = _job_store()
    resp = client.get("/ingest/status/does-not-exist")
    assert resp.status_code == 404


@patch("app.repo_ingest.router.run_incremental_ingest")
@patch("app.repo_ingest.router.get_job_store")
@patch("app.repo_ingest.router.github.resolve_commit_sha")
def test_ingest_files_returns_202_with_incremental_job_type(
    mock_resolve, mock_get_store, mock_run
):
    mock_resolve.return_value = ("main", "abc123")
    store = _job_store()
    mock_get_store.return_value = store

    resp = client.post("/ingest/files", json={"repo": "octo/hello", "ref": "main"})

    assert resp.status_code == 202
    body = resp.json()
    assert body["job_type"] == "incremental"
    mock_run.assert_called_once_with(body["job_id"], "octo/hello", "main", "abc123")


@patch("app.repo_ingest.router.run_full_ingest")
@patch("app.repo_ingest.router.run_incremental_ingest")
@patch("app.repo_ingest.router.get_job_store")
@patch("app.repo_ingest.router.github.resolve_commit_sha")
def test_ingest_files_conflicts_with_running_full_ingest_on_same_repo(
    mock_resolve, mock_get_store, mock_incremental, mock_full
):
    # the lock is per-repo, not per-endpoint -- a bulk ingest holding it
    # must block a concurrent incremental request for the same repo too
    mock_resolve.return_value = ("main", "abc123")
    store = _job_store()
    store.acquire_repo_lock("octo/hello", "running-full-job")
    mock_get_store.return_value = store

    resp = client.post("/ingest/files", json={"repo": "octo/hello", "ref": "main"})

    assert resp.status_code == 409
    mock_incremental.assert_not_called()
    mock_full.assert_not_called()


def test_ingest_files_rejects_malformed_repo():
    resp = client.post("/ingest/files", json={"repo": "not-a-valid-repo"})
    assert resp.status_code == 400
