# app/repo_ingest/service.py
"""
Orchestrates GitHub repo ingestion: download -> filter -> chunk -> embed ->
upsert -> sweep. run_full_ingest is the FastAPI BackgroundTasks target for
POST /ingest/repo -- it owns its own job-status updates and error handling,
since nothing upstream is watching for exceptions once the request has
already returned 202.
"""

import re
import tempfile
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from structlog import get_logger

from app.caching.cache import get_semantic_cache
from app.config import get_settings
from app.ingestion.chunker import ChunkStrategy, get_chunker
from app.ingestion.embedder import embed_chunks, get_embedding_dim
from app.ingestion.indexer import (
    HYBRID_MODEL,
    delete_repo_points_by_path,
    ensure_hybrid_collection,
    ensure_repo_payload_indexes,
    get_qdrant_client,
    sweep_stale_points_for_path,
    sweep_stale_repo_points,
    upsert_repo_chunks_hybrid,
)
from app.ingestion.loader import Document
from app.ingestion.sparse_embedder import embed_chunks_sparse
from app.repo_ingest import github
from app.repo_ingest.filters import (
    document_for,
    is_ingestable_path,
    iter_ingestable_files,
)
from app.repo_ingest.job_store import IngestJob, JobStore, get_job_store

log = get_logger(__name__)
settings = get_settings()

# GitHub's compare API caps the files array itself (see
# github.COMPARE_FILES_CAP); this is a second, tighter threshold on how
# many *ingestable* changes are worth applying one-by-one via the contents
# API before a single tarball fetch is cheaper.
INCREMENTAL_FILE_THRESHOLD = 50

REPO_PATTERN = re.compile(r"^[\w.-]+/[\w.-]+$")


class InvalidRepoFormatError(ValueError):
    def __init__(self, repo: str):
        self.repo = repo
        super().__init__("repo must be in 'owner/name' form")


class RepoLockedError(Exception):
    def __init__(self, repo: str, holder_job_id: str):
        self.repo = repo
        self.holder_job_id = holder_job_id
        super().__init__(
            f"An ingest is already running for {repo!r} "
            f"(job_id={holder_job_id!r})"
        )


def repo_collection_name(repo: str, model: str) -> str:
    """
    One hybrid collection per repo, e.g.
    docmind_repo_octo-hello_text-embedding-3-small_hybrid. "/" in the repo
    slug and HuggingFace-style model ids both get sanitized, same reason as
    app.ingestion.indexer.collection_name_for -- Qdrant's REST API takes
    the collection name as a URL path segment.
    """
    safe_repo = repo.replace("/", "-").lower()
    safe_model = model.replace("/", "-")
    return f"{settings.qdrant_collection}_repo_{safe_repo}_{safe_model}_hybrid"


def prepare_ingest_job(
    repo: str,
    ref: str | None,
    job_type: str,
    job_store_factory: Callable[[], JobStore],
) -> IngestJob:
    """
    Shared setup for POST /ingest/repo, POST /ingest/files, and the MCP
    ingest_repo/sync_repo_incremental tools: validates repo format,
    resolves ref -> commit SHA, acquires the per-repo lock, and creates
    the job record. Does NOT schedule the actual ingestion work -- callers
    decide how (FastAPI BackgroundTasks vs. the MCP server's own bounded
    thread pool).

    job_store_factory is called lazily, only after the repo-format check
    passes -- same evaluation order the original router-level
    _accept_ingest_job used, so a malformed repo never even constructs a
    JobStore.

    Raises InvalidRepoFormatError, github.GithubNotFoundError,
    github.GithubAuthError, github.GithubRateLimitedError, or
    RepoLockedError. Callers map these to their own error surface
    (HTTPException for the router, a plain exception for MCP tools).
    """
    if not REPO_PATTERN.match(repo):
        raise InvalidRepoFormatError(repo)

    resolved_ref, commit_sha = github.resolve_commit_sha(repo, ref)

    job_store = job_store_factory()

    # Mint the job id before creating the record, so a lock conflict never
    # leaves an orphan job that nothing will ever run (see
    # JobStore.create_job's job_id docstring).
    job_id = uuid.uuid4().hex
    holder = job_store.acquire_repo_lock(repo, job_id)
    if holder is not None:
        raise RepoLockedError(repo, holder)

    return job_store.create_job(
        job_type, repo, resolved_ref, commit_sha, job_id=job_id
    )


def run_full_ingest(job_id: str, repo: str, ref: str, commit_sha: str) -> None:
    """
    Downloads the repo tarball pinned to commit_sha, chunks/embeds every
    ingestable file, and upserts into repo_collection_name(repo,
    HYBRID_MODEL) -- then sweeps every point whose commit_sha doesn't match
    this run, so deleted/shrunk files don't linger (mark-and-sweep; see
    app.ingestion.indexer.sweep_stale_repo_points). Point IDs are
    deterministic (uuid5 of chunk_id, which embeds the file path and chunk
    index), so re-running this for the same repo/commit_sha is idempotent.
    Any failure marks the job failed and leaves already-upserted points as
    they are -- they're consistent under their own commit_sha, and a re-run
    heals the rest.
    """
    job_store = get_job_store()
    try:
        job_store.mark_running(job_id)

        with tempfile.TemporaryDirectory(prefix="docmind-repo-ingest-") as tmp:
            tmp_dir = Path(tmp)
            root = github.download_tarball(repo, commit_sha, tmp_dir)

            documents: list[Document] = [
                document_for(path, text)
                for path, text in iter_ingestable_files(root)
            ]

            log.info(
                "repo ingest: files selected",
                job_id=job_id,
                repo=repo,
                files=len(documents),
            )

            languages = {doc.doc_id: doc.language for doc in documents}

            chunker = get_chunker(
                ChunkStrategy.CODE, chunk_size=500, chunk_overlap=50
            )
            chunks = chunker.chunk_documents(documents)

            chunk_embeddings = embed_chunks(chunks, model=HYBRID_MODEL)
            sparse_embeddings = embed_chunks_sparse(chunks)

            client = get_qdrant_client()
            collection = repo_collection_name(repo, HYBRID_MODEL)
            ensure_hybrid_collection(
                client, collection, vector_size=get_embedding_dim(HYBRID_MODEL)
            )
            ensure_repo_payload_indexes(client, collection)

            upsert_repo_chunks_hybrid(
                client,
                collection,
                chunk_embeddings,
                sparse_embeddings,
                repo=repo,
                ref=ref,
                commit_sha=commit_sha,
                ingested_at=datetime.now(UTC).isoformat(),
                languages=languages,
            )

            chunks_deleted = sweep_stale_repo_points(
                client, collection, commit_sha
            )

        job_store.set_watermark(repo, commit_sha)

        try:
            get_semantic_cache().flush()
        except Exception:
            # The repo is already searchable at this point -- a cache-flush
            # failure shouldn't fail the whole ingest, just risk a stale
            # cached answer until the cache entry's TTL expires. Same
            # tradeoff as app.documents.service.ingest_uploaded_document.
            log.warning(
                "semantic cache flush failed after repo ingest",
                exc_info=True,
                job_id=job_id,
                repo=repo,
            )

        job_store.mark_completed(
            job_id,
            files_ingested=len(documents),
            chunks_upserted=len(chunks),
            chunks_deleted=chunks_deleted,
            tokens_embedded=sum(c.token_count for c in chunks),
        )
        log.info("repo ingest: completed", job_id=job_id, repo=repo)
    except Exception as exc:
        log.exception("repo ingest: failed", job_id=job_id, repo=repo)
        job_store.mark_failed(job_id, str(exc))
    finally:
        job_store.release_repo_lock(repo, job_id)


def run_incremental_ingest(
    job_id: str, repo: str, ref: str, head_sha: str
) -> None:
    """
    Diffs against the last successfully ingested commit (the watermark
    JobStore.get_watermark tracks) via GitHub's compare API, and applies
    only the changed files instead of re-downloading the whole repo.
    Falls back to run_full_ingest -- which is idempotent and safe to
    delegate to directly, since it owns its own status/lock handling end
    to end -- whenever the diff isn't trustworthy or cheap enough to apply
    file-by-file: no watermark yet, a force-push ("diverged"), or too many
    changed files (either GitHub's own compare-API cap, or this module's
    INCREMENTAL_FILE_THRESHOLD). "identical" and "behind" are no-ops --
    the latter specifically guards against a late or duplicate webhook
    rolling the index backward.

    Unlike run_full_ingest, this never runs the whole-collection sweep:
    unchanged files legitimately keep an older commit_sha (it means "SHA
    when last written", not "current repo SHA"). Cleanup here is always
    scoped to the specific paths that changed.
    """
    job_store = get_job_store()
    try:
        job_store.mark_running(job_id)

        watermark = job_store.get_watermark(repo)
        if watermark is None:
            log.info(
                "repo incremental ingest: no watermark, falling back to full ingest",
                job_id=job_id,
                repo=repo,
            )
            run_full_ingest(job_id, repo, ref, head_sha)
            return

        comparison = github.compare(repo, watermark, head_sha)

        if comparison.status in ("identical", "behind"):
            job_store.mark_completed(job_id)
            log.info(
                "repo incremental ingest: no-op",
                job_id=job_id,
                repo=repo,
                status=comparison.status,
            )
            return

        too_large_to_diff = (
            comparison.status == "diverged"
            or len(comparison.files) >= github.COMPARE_FILES_CAP
        )
        if too_large_to_diff:
            log.info(
                "repo incremental ingest: falling back to full ingest",
                job_id=job_id,
                repo=repo,
                status=comparison.status,
                changed_files=len(comparison.files),
            )
            run_full_ingest(job_id, repo, ref, head_sha)
            return

        changes = [
            f
            for f in comparison.files
            if is_ingestable_path(f.path)
            or (
                f.previous_path is not None
                and is_ingestable_path(f.previous_path)
            )
        ]
        if len(changes) > INCREMENTAL_FILE_THRESHOLD:
            log.info(
                "repo incremental ingest: too many ingestable changes, "
                "falling back to full ingest",
                job_id=job_id,
                repo=repo,
                changed_files=len(changes),
            )
            run_full_ingest(job_id, repo, ref, head_sha)
            return

        client = get_qdrant_client()
        collection = repo_collection_name(repo, HYBRID_MODEL)
        ensure_hybrid_collection(
            client, collection, vector_size=get_embedding_dim(HYBRID_MODEL)
        )
        ensure_repo_payload_indexes(client, collection)

        chunker = get_chunker(
            ChunkStrategy.CODE, chunk_size=500, chunk_overlap=50
        )
        ingested_at = datetime.now(UTC).isoformat()

        added = modified = removed = 0
        chunks_upserted = chunks_deleted = tokens_embedded = 0

        for change in changes:
            if change.status == "renamed" and change.previous_path:
                chunks_deleted += delete_repo_points_by_path(
                    client, collection, change.previous_path
                )

            if change.status == "removed":
                chunks_deleted += delete_repo_points_by_path(
                    client, collection, change.path
                )
                removed += 1
                continue

            if not is_ingestable_path(change.path):
                # renamed away from an ingestable extension -- the old
                # path's points were already cleaned up above
                continue

            text = github.get_file_content(repo, change.path, head_sha)
            document = document_for(change.path, text)
            file_chunks = chunker.chunk_document(document)

            chunk_embeddings = embed_chunks(file_chunks, model=HYBRID_MODEL)
            sparse_embeddings = embed_chunks_sparse(file_chunks)
            upsert_repo_chunks_hybrid(
                client,
                collection,
                chunk_embeddings,
                sparse_embeddings,
                repo=repo,
                ref=ref,
                commit_sha=head_sha,
                ingested_at=ingested_at,
                languages={document.doc_id: document.language},
            )
            chunks_upserted += len(file_chunks)
            tokens_embedded += sum(c.token_count for c in file_chunks)

            # A modified file may now produce fewer chunks than before --
            # clean up any leftover points for this path still tagged with
            # an older commit_sha. Mirrors sweep_stale_repo_points, but
            # scoped to this one path instead of the whole collection
            # (this path never runs the whole-collection sweep).
            chunks_deleted += sweep_stale_points_for_path(
                client, collection, change.path, head_sha
            )

            if change.status == "added":
                added += 1
            else:
                modified += 1

        job_store.set_watermark(repo, head_sha)

        try:
            get_semantic_cache().flush()
        except Exception:
            log.warning(
                "semantic cache flush failed after incremental repo ingest",
                exc_info=True,
                job_id=job_id,
                repo=repo,
            )

        job_store.mark_completed(
            job_id,
            files_ingested=added + modified,
            files_added=added,
            files_modified=modified,
            files_removed=removed,
            chunks_upserted=chunks_upserted,
            chunks_deleted=chunks_deleted,
            tokens_embedded=tokens_embedded,
        )
        log.info(
            "repo incremental ingest: completed",
            job_id=job_id,
            repo=repo,
            added=added,
            modified=modified,
            removed=removed,
        )
    except Exception as exc:
        log.exception(
            "repo incremental ingest: failed", job_id=job_id, repo=repo
        )
        job_store.mark_failed(job_id, str(exc))
    finally:
        job_store.release_repo_lock(repo, job_id)
