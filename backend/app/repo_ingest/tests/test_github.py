import io
import tarfile

import httpx
import pytest

from app.repo_ingest.github import (
    GithubAuthError,
    GithubError,
    GithubNotFoundError,
    GithubRateLimitedError,
    compare,
    download_tarball,
    get_file_content,
    resolve_commit_sha,
)


def _client_for(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_resolve_commit_sha_with_explicit_ref():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/repos/octo/hello/commits/main"
        return httpx.Response(200, json={"sha": "abc123"})

    ref, sha = resolve_commit_sha("octo/hello", "main", client=_client_for(handler))
    assert ref == "main"
    assert sha == "abc123"


def test_resolve_commit_sha_falls_back_to_default_branch():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/repos/octo/hello":
            return httpx.Response(200, json={"default_branch": "trunk"})
        assert request.url.path == "/repos/octo/hello/commits/trunk"
        return httpx.Response(200, json={"sha": "def456"})

    ref, sha = resolve_commit_sha("octo/hello", None, client=_client_for(handler))
    assert ref == "trunk"
    assert sha == "def456"
    assert calls == ["/repos/octo/hello", "/repos/octo/hello/commits/trunk"]


def test_resolve_commit_sha_404_raises_not_found():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "Not Found"})

    with pytest.raises(GithubNotFoundError):
        resolve_commit_sha("octo/missing", "main", client=_client_for(handler))


def test_resolve_commit_sha_401_raises_auth_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "Bad credentials"})

    with pytest.raises(GithubAuthError):
        resolve_commit_sha("octo/private", "main", client=_client_for(handler))


def test_resolve_commit_sha_rate_limited():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            headers={"x-ratelimit-remaining": "0"},
            json={"message": "rate limited"},
        )

    with pytest.raises(GithubRateLimitedError):
        resolve_commit_sha("octo/hello", "main", client=_client_for(handler))


def test_resolve_commit_sha_other_403_raises_generic_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            headers={"x-ratelimit-remaining": "42"},
            json={"message": "forbidden"},
        )

    with pytest.raises(httpx.HTTPStatusError):
        resolve_commit_sha("octo/hello", "main", client=_client_for(handler))


def _make_tarball_bytes(root_name: str, files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel_path, content in files.items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=f"{root_name}/{rel_path}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_download_tarball_extracts_and_strips_root_prefix(tmp_path):
    tarball_bytes = _make_tarball_bytes(
        "octo-hello-abc123",
        {"README.md": "# hi", "src/main.py": "x = 1"},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/repos/octo/hello/tarball/abc123"
        return httpx.Response(
            200, content=tarball_bytes, headers={"content-type": "application/gzip"}
        )

    root = download_tarball(
        "octo/hello", "abc123", tmp_path, client=_client_for(handler)
    )
    assert root.name == "octo-hello-abc123"
    assert (root / "README.md").read_text() == "# hi"
    assert (root / "src" / "main.py").read_text() == "x = 1"


def test_download_tarball_404_raises_not_found(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "Not Found"})

    with pytest.raises(GithubNotFoundError):
        download_tarball("octo/missing", "deadbeef", tmp_path, client=_client_for(handler))


def test_download_tarball_rejects_unexpected_root_count(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for root_name in ("root-one", "root-two"):
            data = b"x"
            info = tarfile.TarInfo(name=f"{root_name}/file.txt")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    tarball_bytes = buf.getvalue()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=tarball_bytes)

    with pytest.raises(GithubError):
        download_tarball("octo/weird", "sha", tmp_path, client=_client_for(handler))


def test_compare_parses_status_and_files():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/repos/octo/hello/compare/base-sha...head-sha"
        return httpx.Response(
            200,
            json={
                "status": "ahead",
                "files": [
                    {"filename": "src/a.py", "status": "added"},
                    {"filename": "src/b.py", "status": "modified"},
                    {"filename": "src/c.py", "status": "removed"},
                    {
                        "filename": "src/new_name.py",
                        "status": "renamed",
                        "previous_filename": "src/old_name.py",
                    },
                ],
            },
        )

    result = compare(
        "octo/hello", "base-sha", "head-sha", client=_client_for(handler)
    )

    assert result.status == "ahead"
    assert len(result.files) == 4
    assert result.files[0].path == "src/a.py"
    assert result.files[0].status == "added"
    assert result.files[3].previous_path == "src/old_name.py"


def test_compare_identical_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "identical", "files": []})

    result = compare("octo/hello", "sha1", "sha1", client=_client_for(handler))
    assert result.status == "identical"
    assert result.files == []


def test_compare_404_raises_not_found():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "Not Found"})

    with pytest.raises(GithubNotFoundError):
        compare("octo/hello", "bad-sha", "head-sha", client=_client_for(handler))


def test_get_file_content_returns_raw_text():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/repos/octo/hello/contents/src/main.py"
        assert request.url.params["ref"] == "abc123"
        assert request.headers["accept"] == "application/vnd.github.raw+json"
        return httpx.Response(200, text="def f():\n    return 1\n")

    text = get_file_content(
        "octo/hello", "src/main.py", "abc123", client=_client_for(handler)
    )
    assert text == "def f():\n    return 1\n"


def test_get_file_content_404_raises_not_found():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "Not Found"})

    with pytest.raises(GithubNotFoundError):
        get_file_content(
            "octo/hello", "missing.py", "abc123", client=_client_for(handler)
        )
