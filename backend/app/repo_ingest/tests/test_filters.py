from pathlib import Path

from app.repo_ingest.filters import (
    document_for,
    is_ingestable_path,
    iter_ingestable_files,
)


def _touch(path: Path, content: str = "content") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_finds_code_and_markdown_files(tmp_path):
    _touch(tmp_path / "src" / "main.py", "def f(): pass")
    _touch(tmp_path / "README.md", "# hi")

    found = dict(iter_ingestable_files(tmp_path))
    assert "src/main.py" in found
    assert "README.md" in found


def test_skips_vendored_and_generated_dirs(tmp_path):
    _touch(tmp_path / "node_modules" / "pkg" / "index.js", "module.exports = {}")
    _touch(tmp_path / "src" / "app.py", "x = 1")
    _touch(tmp_path / "nested" / "vendor" / "lib" / "code.go", "package lib")

    found = dict(iter_ingestable_files(tmp_path))
    assert "src/app.py" in found
    assert not any("node_modules" in p for p in found)
    assert not any("vendor" in p for p in found)


def test_skips_lockfiles(tmp_path):
    _touch(tmp_path / "package-lock.json", "{}")
    _touch(tmp_path / "uv.lock", "")
    _touch(tmp_path / "app.py", "x = 1")

    found = dict(iter_ingestable_files(tmp_path))
    assert "app.py" in found
    assert "package-lock.json" not in found
    assert "uv.lock" not in found


def test_skips_unlisted_extensions(tmp_path):
    _touch(tmp_path / "image.png", "binary-ish")
    _touch(tmp_path / "app.py", "x = 1")

    found = dict(iter_ingestable_files(tmp_path))
    assert "app.py" in found
    assert "image.png" not in found


def test_skips_files_over_size_cap(tmp_path):
    big = tmp_path / "big.py"
    _touch(big, "x = 1\n" * 500_000)  # well over MAX_FILE_BYTES
    small = tmp_path / "small.py"
    _touch(small, "x = 1")

    found = dict(iter_ingestable_files(tmp_path))
    assert "small.py" in found
    assert "big.py" not in found


def test_skips_binary_files_that_fail_utf8_decode(tmp_path):
    binary_path = tmp_path / "data.py"
    binary_path.write_bytes(b"\xff\xfe\x00\x01binary garbage")

    found = dict(iter_ingestable_files(tmp_path))
    assert "data.py" not in found


def test_skips_empty_files(tmp_path):
    _touch(tmp_path / "empty.py", "   \n\n  ")

    found = dict(iter_ingestable_files(tmp_path))
    assert "empty.py" not in found


def test_document_for_maps_markdown_extensions():
    doc = document_for("docs/guide.md", "# Guide")
    assert doc.doc_type == "markdown"
    assert doc.language == "markdown"
    assert doc.doc_id == "docs/guide.md"


def test_document_for_maps_code_language():
    doc = document_for("src/main.go", "package main")
    assert doc.doc_type == "code"
    assert doc.language == "go"


def test_document_for_config_file_is_code_type_with_language():
    doc = document_for("charts/values.yaml", "key: value")
    assert doc.doc_type == "code"
    assert doc.language == "yaml"


def test_is_ingestable_path_accepts_code_and_docs():
    assert is_ingestable_path("src/main.py") is True
    assert is_ingestable_path("README.md") is True


def test_is_ingestable_path_rejects_unlisted_extension():
    assert is_ingestable_path("image.png") is False


def test_is_ingestable_path_rejects_skipped_dirs():
    assert is_ingestable_path("node_modules/pkg/index.js") is False
    assert is_ingestable_path("nested/vendor/lib/code.go") is False


def test_is_ingestable_path_rejects_lockfiles():
    assert is_ingestable_path("package-lock.json") is False
