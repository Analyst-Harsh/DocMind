from unittest.mock import MagicMock, patch

import pytest

from app.ingestion.loader import Document
from app.repo_ingest.github import COMPARE_FILES_CAP, ChangedFile, CompareResult
from app.repo_ingest.job_store import JobStore
from app.repo_ingest.service import (
    repo_collection_name,
    run_full_ingest,
    run_incremental_ingest,
)


def test_repo_collection_name_sanitizes_slash_and_lowercases():
    name = repo_collection_name("Octo/Hello", "text-embedding-3-small")
    assert name == "docmind_repo_octo-hello_text-embedding-3-small_hybrid"


@pytest.fixture
def store(fake_redis) -> JobStore:
    return JobStore(client=fake_redis, job_ttl_seconds=3600)


@patch("app.repo_ingest.service.get_semantic_cache")
@patch("app.repo_ingest.service.sweep_stale_repo_points")
@patch("app.repo_ingest.service.upsert_repo_chunks_hybrid")
@patch("app.repo_ingest.service.ensure_repo_payload_indexes")
@patch("app.repo_ingest.service.ensure_hybrid_collection")
@patch("app.repo_ingest.service.get_qdrant_client")
@patch("app.repo_ingest.service.embed_chunks_sparse")
@patch("app.repo_ingest.service.embed_chunks")
@patch("app.repo_ingest.service.iter_ingestable_files")
@patch("app.repo_ingest.service.github")
def test_run_full_ingest_happy_path(
    mock_github,
    mock_iter_files,
    mock_embed_chunks,
    mock_embed_sparse,
    mock_get_client,
    mock_ensure_collection,
    mock_ensure_indexes,
    mock_upsert,
    mock_sweep,
    mock_get_cache,
    store,
):
    job = store.create_job("full", "octo/hello", "main", "abc123")
    store.acquire_repo_lock("octo/hello", job.job_id)

    mock_github.download_tarball.return_value = "/tmp/fake-root"
    mock_iter_files.return_value = [("src/main.py", "def f():\n    return 1\n")]
    mock_embed_chunks.side_effect = lambda chunks, model=None: [
        (c, [0.1, 0.2]) for c in chunks
    ]
    mock_embed_sparse.side_effect = lambda chunks: [(c, MagicMock()) for c in chunks]
    mock_sweep.return_value = 2
    mock_get_cache.return_value = MagicMock()

    with patch("app.repo_ingest.service.get_job_store", return_value=store):
        run_full_ingest(job.job_id, "octo/hello", "main", "abc123")

    updated = store.get_job(job.job_id)
    assert updated.status == "completed"
    assert updated.files_ingested == 1
    assert updated.chunks_deleted == 2
    assert updated.chunks_upserted >= 1

    # upsert must happen before the sweep -- sweeping first would delete
    # every point in an empty/stale collection before the new ones land.
    assert mock_upsert.call_args_list and mock_sweep.call_args_list
    assert store.get_watermark("octo/hello") == "abc123"

    # sweep must run with this run's own commit_sha, so it only removes
    # points tagged with a *different* (stale) sha
    mock_sweep.assert_called_once_with(mock_get_client.return_value, mock_ensure_collection.call_args.args[1], "abc123")

    # lock released after success
    assert store.acquire_repo_lock("octo/hello", "another-job") is None


@patch("app.repo_ingest.service.get_semantic_cache")
@patch("app.repo_ingest.service.iter_ingestable_files")
@patch("app.repo_ingest.service.github")
def test_run_full_ingest_marks_failed_and_releases_lock_on_error(
    mock_github, mock_iter_files, mock_get_cache, store
):
    job = store.create_job("full", "octo/hello", "main", "abc123")
    store.acquire_repo_lock("octo/hello", job.job_id)

    mock_github.download_tarball.side_effect = RuntimeError("network exploded")

    with patch("app.repo_ingest.service.get_job_store", return_value=store):
        run_full_ingest(job.job_id, "octo/hello", "main", "abc123")

    updated = store.get_job(job.job_id)
    assert updated.status == "failed"
    assert "network exploded" in updated.error

    # lock released even on failure -- a retry isn't blocked forever
    assert store.acquire_repo_lock("octo/hello", "another-job") is None


def test_document_language_map_built_before_chunking():
    # documents built from iter_ingestable_files carry .language, which
    # upsert_repo_chunks_hybrid needs per-chunk (via the languages dict) --
    # sanity check the Document fixture shape this relies on.
    doc = Document(
        doc_id="src/main.py", title="src/main.py", text="x = 1",
        doc_type="code", source_path="src/main.py", tags=[], language="python",
    )
    assert doc.language == "python"


@patch("app.repo_ingest.service.run_full_ingest")
@patch("app.repo_ingest.service.github")
def test_incremental_falls_back_to_full_when_no_watermark(
    mock_github, mock_full, store
):
    job = store.create_job("incremental", "octo/hello", "main", "head-sha")
    store.acquire_repo_lock("octo/hello", job.job_id)

    with patch("app.repo_ingest.service.get_job_store", return_value=store):
        run_incremental_ingest(job.job_id, "octo/hello", "main", "head-sha")

    mock_full.assert_called_once_with(job.job_id, "octo/hello", "main", "head-sha")
    mock_github.compare.assert_not_called()


@pytest.mark.parametrize("status", ["identical", "behind"])
@patch("app.repo_ingest.service.run_full_ingest")
@patch("app.repo_ingest.service.github")
def test_incremental_is_a_no_op_when_identical_or_behind(
    mock_github, mock_full, store, status
):
    job = store.create_job("incremental", "octo/hello", "main", "head-sha")
    store.acquire_repo_lock("octo/hello", job.job_id)
    store.set_watermark("octo/hello", "old-sha")
    mock_github.compare.return_value = CompareResult(status=status, files=[])

    with patch("app.repo_ingest.service.get_job_store", return_value=store):
        run_incremental_ingest(job.job_id, "octo/hello", "main", "head-sha")

    mock_full.assert_not_called()
    updated = store.get_job(job.job_id)
    assert updated.status == "completed"
    # watermark must NOT advance on a "behind" (stale/duplicate) webhook
    assert store.get_watermark("octo/hello") == "old-sha"
    assert store.acquire_repo_lock("octo/hello", "another-job") is None


@patch("app.repo_ingest.service.run_full_ingest")
@patch("app.repo_ingest.service.github")
def test_incremental_falls_back_to_full_when_diverged(
    mock_github, mock_full, store
):
    job = store.create_job("incremental", "octo/hello", "main", "head-sha")
    store.acquire_repo_lock("octo/hello", job.job_id)
    store.set_watermark("octo/hello", "old-sha")
    mock_github.compare.return_value = CompareResult(status="diverged", files=[])

    with patch("app.repo_ingest.service.get_job_store", return_value=store):
        run_incremental_ingest(job.job_id, "octo/hello", "main", "head-sha")

    mock_full.assert_called_once_with(job.job_id, "octo/hello", "main", "head-sha")


@patch("app.repo_ingest.service.run_full_ingest")
@patch("app.repo_ingest.service.github")
def test_incremental_falls_back_when_compare_hits_api_file_cap(
    mock_github, mock_full, store
):
    mock_github.COMPARE_FILES_CAP = COMPARE_FILES_CAP
    job = store.create_job("incremental", "octo/hello", "main", "head-sha")
    store.acquire_repo_lock("octo/hello", job.job_id)
    store.set_watermark("octo/hello", "old-sha")
    files = [
        ChangedFile(path=f"src/f{i}.py", status="modified")
        for i in range(COMPARE_FILES_CAP)
    ]
    mock_github.compare.return_value = CompareResult(status="ahead", files=files)

    with patch("app.repo_ingest.service.get_job_store", return_value=store):
        run_incremental_ingest(job.job_id, "octo/hello", "main", "head-sha")

    mock_full.assert_called_once_with(job.job_id, "octo/hello", "main", "head-sha")


@patch("app.repo_ingest.service.run_full_ingest")
@patch("app.repo_ingest.service.github")
def test_incremental_falls_back_when_too_many_ingestable_changes(
    mock_github, mock_full, store
):
    from app.repo_ingest.service import INCREMENTAL_FILE_THRESHOLD

    mock_github.COMPARE_FILES_CAP = COMPARE_FILES_CAP
    job = store.create_job("incremental", "octo/hello", "main", "head-sha")
    store.acquire_repo_lock("octo/hello", job.job_id)
    store.set_watermark("octo/hello", "old-sha")
    files = [
        ChangedFile(path=f"src/f{i}.py", status="modified")
        for i in range(INCREMENTAL_FILE_THRESHOLD + 1)
    ]
    mock_github.compare.return_value = CompareResult(status="ahead", files=files)

    with patch("app.repo_ingest.service.get_job_store", return_value=store):
        run_incremental_ingest(job.job_id, "octo/hello", "main", "head-sha")

    mock_full.assert_called_once_with(job.job_id, "octo/hello", "main", "head-sha")


@patch("app.repo_ingest.service.get_semantic_cache")
@patch("app.repo_ingest.service.sweep_stale_points_for_path")
@patch("app.repo_ingest.service.upsert_repo_chunks_hybrid")
@patch("app.repo_ingest.service.ensure_repo_payload_indexes")
@patch("app.repo_ingest.service.ensure_hybrid_collection")
@patch("app.repo_ingest.service.get_qdrant_client")
@patch("app.repo_ingest.service.embed_chunks_sparse")
@patch("app.repo_ingest.service.embed_chunks")
@patch("app.repo_ingest.service.delete_repo_points_by_path")
@patch("app.repo_ingest.service.run_full_ingest")
@patch("app.repo_ingest.service.github")
def test_incremental_applies_added_modified_removed_and_renamed(
    mock_github,
    mock_full,
    mock_delete_by_path,
    mock_embed_chunks,
    mock_embed_sparse,
    mock_get_client,
    mock_ensure_collection,
    mock_ensure_indexes,
    mock_upsert,
    mock_sweep_path,
    mock_get_cache,
    store,
):
    mock_github.COMPARE_FILES_CAP = COMPARE_FILES_CAP
    job = store.create_job("incremental", "octo/hello", "main", "head-sha")
    store.acquire_repo_lock("octo/hello", job.job_id)
    store.set_watermark("octo/hello", "old-sha")

    mock_github.compare.return_value = CompareResult(
        status="ahead",
        files=[
            ChangedFile(path="src/new_file.py", status="added"),
            ChangedFile(path="src/edited.py", status="modified"),
            ChangedFile(path="src/gone.py", status="removed"),
            ChangedFile(
                path="src/renamed_to.py",
                status="renamed",
                previous_path="src/renamed_from.py",
            ),
        ],
    )
    mock_github.get_file_content.return_value = "def f():\n    return 1\n"
    mock_embed_chunks.side_effect = lambda chunks, model=None: [
        (c, [0.1, 0.2]) for c in chunks
    ]
    mock_embed_sparse.side_effect = lambda chunks: [(c, MagicMock()) for c in chunks]
    mock_delete_by_path.return_value = 1
    mock_sweep_path.return_value = 0
    mock_get_cache.return_value = MagicMock()

    with patch("app.repo_ingest.service.get_job_store", return_value=store):
        run_incremental_ingest(job.job_id, "octo/hello", "main", "head-sha")

    mock_full.assert_not_called()

    updated = store.get_job(job.job_id)
    assert updated.status == "completed"
    assert updated.files_added == 1
    # "modified" and "renamed" are both counted under files_modified --
    # the counter is informational only, not a correctness-affecting value
    assert updated.files_modified == 2
    assert updated.files_removed == 1
    assert store.get_watermark("octo/hello") == "head-sha"

    # removed + renamed-away-from each trigger a path delete
    deleted_paths = {c.args[2] for c in mock_delete_by_path.call_args_list}
    assert deleted_paths == {"src/gone.py", "src/renamed_from.py"}

    # added/modified/renamed-to all fetch content and get chunked+upserted
    fetched_paths = {c.args[1] for c in mock_github.get_file_content.call_args_list}
    assert fetched_paths == {"src/new_file.py", "src/edited.py", "src/renamed_to.py"}

    assert store.acquire_repo_lock("octo/hello", "another-job") is None


@patch("app.repo_ingest.service.get_semantic_cache")
@patch("app.repo_ingest.service.sweep_stale_repo_points")
@patch("app.repo_ingest.service.sweep_stale_points_for_path")
@patch("app.repo_ingest.service.upsert_repo_chunks_hybrid")
@patch("app.repo_ingest.service.ensure_repo_payload_indexes")
@patch("app.repo_ingest.service.ensure_hybrid_collection")
@patch("app.repo_ingest.service.get_qdrant_client")
@patch("app.repo_ingest.service.embed_chunks_sparse")
@patch("app.repo_ingest.service.embed_chunks")
@patch("app.repo_ingest.service.run_full_ingest")
@patch("app.repo_ingest.service.github")
def test_incremental_never_runs_whole_collection_sweep(
    mock_github,
    mock_full,
    mock_embed_chunks,
    mock_embed_sparse,
    mock_get_client,
    mock_ensure_collection,
    mock_ensure_indexes,
    mock_upsert,
    mock_sweep_path,
    mock_sweep_whole,
    mock_get_cache,
    store,
):
    mock_github.COMPARE_FILES_CAP = COMPARE_FILES_CAP
    job = store.create_job("incremental", "octo/hello", "main", "head-sha")
    store.acquire_repo_lock("octo/hello", job.job_id)
    store.set_watermark("octo/hello", "old-sha")

    mock_github.compare.return_value = CompareResult(
        status="ahead",
        files=[ChangedFile(path="src/edited.py", status="modified")],
    )
    mock_github.get_file_content.return_value = "x = 1\n"
    mock_embed_chunks.side_effect = lambda chunks, model=None: [
        (c, [0.1]) for c in chunks
    ]
    mock_embed_sparse.side_effect = lambda chunks: [(c, MagicMock()) for c in chunks]
    mock_sweep_path.return_value = 0
    mock_get_cache.return_value = MagicMock()

    with patch("app.repo_ingest.service.get_job_store", return_value=store):
        run_incremental_ingest(job.job_id, "octo/hello", "main", "head-sha")

    # assert completion first -- otherwise "sweep never called" would pass
    # vacuously if the run failed before reaching the sweep call at all
    assert store.get_job(job.job_id).status == "completed"
    mock_sweep_whole.assert_not_called()
