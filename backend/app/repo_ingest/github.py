# app/repo_ingest/github.py
"""
Minimal GitHub REST client for repo ingestion: resolve a ref to a commit
SHA, download+extract the tarball for that SHA. httpx.Client is injectable
on every function (same optional-client convention as
app.ingestion.indexer.get_qdrant_client) so tests can pass one built on
httpx.MockTransport instead of hitting the network.
"""

import tarfile
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from structlog import get_logger

from app.config import get_settings

log = get_logger(__name__)

GITHUB_API_BASE = "https://api.github.com"

# GitHub's compare API caps the files array at 300 entries and doesn't
# paginate it -- beyond that, the incremental path falls back to a full
# re-ingest rather than working off a truncated diff (see
# app/repo_ingest/service.py's run_incremental_ingest).
COMPARE_FILES_CAP = 300


class GithubError(Exception):
    """Base class for GitHub API failures during repo ingestion."""


class GithubNotFoundError(GithubError):
    """Repo or ref doesn't exist (or isn't visible with the current token)."""


class GithubAuthError(GithubError):
    """GITHUB_TOKEN is missing or was rejected for a private repo."""


class GithubRateLimitedError(GithubError):
    """GitHub API rate limit exhausted."""


@dataclass
class ChangedFile:
    path: str
    status: str  # "added" | "modified" | "removed" | "renamed" | "copied" | "changed" | "unchanged"
    previous_path: str | None = None  # set only when status == "renamed"


@dataclass
class CompareResult:
    status: str  # "identical" | "ahead" | "behind" | "diverged"
    files: list[ChangedFile] = field(default_factory=list)


def _headers() -> dict[str, str]:
    settings = get_settings()
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    return headers


def _check_response(response: httpx.Response, repo: str) -> None:
    if response.status_code == 404:
        raise GithubNotFoundError(
            f"{repo!r} not found, or the ref/sha doesn't exist "
            "(or isn't visible with the configured token)"
        )
    if response.status_code == 401:
        raise GithubAuthError("GitHub rejected the configured GITHUB_TOKEN")
    if (
        response.status_code == 403
        and response.headers.get("x-ratelimit-remaining") == "0"
    ):
        raise GithubRateLimitedError("GitHub API rate limit exhausted")
    response.raise_for_status()


def resolve_commit_sha(
    repo: str, ref: str | None, client: httpx.Client | None = None
) -> tuple[str, str]:
    """
    Resolves ref (branch/tag/sha, or None for the repo's default branch) to
    (ref_used, commit_sha). Two calls when ref is omitted: one to read the
    default branch, one to resolve it to a SHA.
    """
    owned = client is None
    http_client = client or httpx.Client(timeout=30.0)
    try:
        resolved_ref = ref
        if resolved_ref is None:
            resp = http_client.get(
                f"{GITHUB_API_BASE}/repos/{repo}", headers=_headers()
            )
            _check_response(resp, repo)
            resolved_ref = resp.json()["default_branch"]

        resp = http_client.get(
            f"{GITHUB_API_BASE}/repos/{repo}/commits/{resolved_ref}",
            headers=_headers(),
        )
        _check_response(resp, repo)
        return resolved_ref, resp.json()["sha"]
    finally:
        if owned:
            http_client.close()


def download_tarball(
    repo: str, sha: str, dest_dir: Path, client: httpx.Client | None = None
) -> Path:
    """
    Streams the tarball for `sha` into dest_dir and extracts it. Pinned to
    the commit SHA (not a mutable ref) so the downloaded tree can never
    diverge from the SHA recorded on the ingestion job. Extraction uses
    tarfile's "data" filter (path-traversal-safe, Python 3.12+ default in
    3.14 but explicit here since we target 3.13). Returns the single
    top-level directory GitHub's tarball extracts to
    ("{owner}-{repo}-{shortsha}/").
    """
    owned = client is None
    http_client = client or httpx.Client(timeout=120.0, follow_redirects=True)
    tarball_path = dest_dir / "repo.tar.gz"
    try:
        with http_client.stream(
            "GET",
            f"{GITHUB_API_BASE}/repos/{repo}/tarball/{sha}",
            headers=_headers(),
        ) as response:
            _check_response(response, repo)
            with open(tarball_path, "wb") as f:
                for data in response.iter_bytes():
                    f.write(data)
    finally:
        if owned:
            http_client.close()

    extract_dir = dest_dir / "extracted"
    extract_dir.mkdir()
    with tarfile.open(tarball_path) as tar:
        tar.extractall(extract_dir, filter="data")

    roots = [p for p in extract_dir.iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise GithubError(
            f"expected exactly one root directory in {repo}@{sha} tarball, "
            f"found {len(roots)}"
        )
    return roots[0]


def compare(
    repo: str, base: str, head: str, client: httpx.Client | None = None
) -> CompareResult:
    """
    Diffs two commits via GitHub's compare API -- the basis for
    incremental ingestion (app/repo_ingest/service.py's
    run_incremental_ingest). status is "identical" (nothing changed),
    "ahead" (head is a descendant of base -- the normal case), "behind"
    (base is a descendant of head -- a stale/duplicate/out-of-order
    webhook), or "diverged" (base and head share no linear history, e.g.
    after a force-push -- the diff is meaningless and callers should fall
    back to a full re-ingest). files is capped at COMPARE_FILES_CAP by the
    API itself; callers must check for that and fall back too.
    """
    owned = client is None
    http_client = client or httpx.Client(timeout=30.0)
    try:
        resp = http_client.get(
            f"{GITHUB_API_BASE}/repos/{repo}/compare/{base}...{head}",
            headers=_headers(),
        )
        _check_response(resp, repo)
        data = resp.json()
        files = [
            ChangedFile(
                path=f["filename"],
                status=f["status"],
                previous_path=f.get("previous_filename"),
            )
            for f in data.get("files", [])
        ]
        return CompareResult(status=data["status"], files=files)
    finally:
        if owned:
            http_client.close()


def get_file_content(
    repo: str, path: str, sha: str, client: httpx.Client | None = None
) -> str:
    """
    Fetches one file's raw text content at a specific commit, for the
    incremental path's per-file add/modify handling. Uses the contents
    API's raw media type instead of the default JSON response, which
    would base64-encode the content and cap out at 1MB differently --
    raw avoids that decode step entirely.
    """
    owned = client is None
    http_client = client or httpx.Client(timeout=30.0)
    try:
        headers = _headers()
        headers["Accept"] = "application/vnd.github.raw+json"
        resp = http_client.get(
            f"{GITHUB_API_BASE}/repos/{repo}/contents/{path}",
            params={"ref": sha},
            headers=headers,
        )
        _check_response(resp, repo)
        return resp.text
    finally:
        if owned:
            http_client.close()
