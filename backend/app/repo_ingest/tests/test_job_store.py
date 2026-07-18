import pytest

from app.repo_ingest.job_store import JobStore


@pytest.fixture
def store(fake_redis) -> JobStore:
    return JobStore(client=fake_redis, job_ttl_seconds=3600)


def test_create_job_starts_pending(store):
    job = store.create_job("full", "octo/hello", "main", "abc123")
    assert job.status == "pending"
    assert job.repo == "octo/hello"
    assert job.ref == "main"
    assert job.commit_sha == "abc123"
    assert job.job_type == "full"


def test_get_job_round_trips(store):
    created = store.create_job("full", "octo/hello", "main", "abc123")
    fetched = store.get_job(created.job_id)
    assert fetched == created


def test_get_unknown_job_returns_none(store):
    assert store.get_job("does-not-exist") is None


def test_mark_running_sets_status_and_started_at(store):
    job = store.create_job("full", "octo/hello", "main", "abc123")
    store.mark_running(job.job_id)
    updated = store.get_job(job.job_id)
    assert updated.status == "running"
    assert updated.started_at is not None


def test_mark_completed_sets_counters(store):
    job = store.create_job("full", "octo/hello", "main", "abc123")
    store.mark_running(job.job_id)
    store.mark_completed(
        job.job_id, files_ingested=10, chunks_upserted=42, chunks_deleted=3
    )
    updated = store.get_job(job.job_id)
    assert updated.status == "completed"
    assert updated.files_ingested == 10
    assert updated.chunks_upserted == 42
    assert updated.chunks_deleted == 3
    assert updated.finished_at is not None


def test_mark_failed_records_error(store):
    job = store.create_job("full", "octo/hello", "main", "abc123")
    store.mark_failed(job.job_id, "boom")
    updated = store.get_job(job.job_id)
    assert updated.status == "failed"
    assert updated.error == "boom"
    assert updated.finished_at is not None


def test_mark_running_on_unknown_job_raises(store):
    with pytest.raises(KeyError):
        store.mark_running("nope")


def test_acquire_repo_lock_succeeds_when_free(store):
    result = store.acquire_repo_lock("octo/hello", "job-1")
    assert result is None


def test_acquire_repo_lock_conflict_returns_holder(store):
    store.acquire_repo_lock("octo/hello", "job-1")
    holder = store.acquire_repo_lock("octo/hello", "job-2")
    assert holder == "job-1"


def test_release_repo_lock_only_releases_own_lock(store):
    store.acquire_repo_lock("octo/hello", "job-1")
    # job-2 never held the lock -- releasing must be a no-op
    store.release_repo_lock("octo/hello", "job-2")
    holder = store.acquire_repo_lock("octo/hello", "job-3")
    assert holder == "job-1"

    store.release_repo_lock("octo/hello", "job-1")
    result = store.acquire_repo_lock("octo/hello", "job-4")
    assert result is None


def test_watermark_round_trip(store):
    assert store.get_watermark("octo/hello") is None
    store.set_watermark("octo/hello", "sha-1")
    assert store.get_watermark("octo/hello") == "sha-1"
    store.set_watermark("octo/hello", "sha-2")
    assert store.get_watermark("octo/hello") == "sha-2"
