# app/repo_ingest/filters.py
"""
Decides which files in a downloaded repo tree are worth chunking/embedding,
and builds the Document each surviving file becomes -- everything
downstream (CodeChunker, embedder, indexer, citation payloads) then works
unchanged, exactly as it does for the docs corpus.
"""

from collections.abc import Iterator
from pathlib import Path

from app.ingestion.loader import Document

MAX_FILE_BYTES = 1_000_000

CODE_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".sh": "shell",
    ".sql": "sql",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
}

DOC_EXTENSIONS: dict[str, str] = {
    ".md": "markdown",
    ".rst": "text",
    ".txt": "text",
}

CONFIG_EXTENSIONS: dict[str, str] = {
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
}

# Extension -> language, consumed by CodeChunker to pick a separator
# hierarchy (app/ingestion/chunker/code_chunker.py's LANGUAGE_SEPARATORS).
EXTENSION_LANGUAGE: dict[str, str] = {
    **CODE_EXTENSIONS,
    **DOC_EXTENSIONS,
    **CONFIG_EXTENSIONS,
}

# Well-known filenames that carry no extension -- GitHub convention predates
# ".md" becoming universal (e.g. octocat/Hello-World's only file is
# literally named "README", not "README.md"), and Dockerfile/Makefile never
# had one. Keyed by lowercased basename since GitHub itself treats these
# case-insensitively. Mirrors DOC_EXTENSIONS/CODE_EXTENSIONS's split so
# document_for can decide doc_type the same way for both.
DOC_FILENAMES: dict[str, str] = {
    "readme": "markdown",
    "changelog": "markdown",
    "contributing": "markdown",
    "license": "text",
    "licence": "text",
    "authors": "text",
    "notice": "text",
}

CODE_FILENAMES: dict[str, str] = {
    "dockerfile": "dockerfile",
    "makefile": "makefile",
}

FILENAME_LANGUAGE: dict[str, str] = {**DOC_FILENAMES, **CODE_FILENAMES}

# Matched against any path segment, not just the immediate parent, so a
# nested vendored dir (e.g. "foo/node_modules/bar/baz.js") is still skipped.
SKIP_DIR_NAMES = {
    ".git",
    "node_modules",
    "vendor",
    "dist",
    "build",
    "target",
    "__pycache__",
    ".venv",
    "venv",
    ".next",
    ".idea",
    "coverage",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}

SKIP_FILE_NAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "uv.lock",
    "poetry.lock",
    "Cargo.lock",
    "go.sum",
    "composer.lock",
    "Gemfile.lock",
}

MARKDOWN_LIKE_EXTENSIONS = {".md", ".rst", ".txt"}


def _is_in_skipped_dir(relative_dir: Path) -> bool:
    return any(part in SKIP_DIR_NAMES for part in relative_dir.parts)


def is_ingestable_path(path: str) -> bool:
    """
    The static (content-independent) half of the filter -- extension/dir/
    filename checks that don't require reading the file. Shared by
    iter_ingestable_files' bulk walk and the incremental diff path
    (app/repo_ingest/service.py's run_incremental_ingest), which uses this
    to decide whether a changed path is worth an API call to fetch its
    content at all, before size/binary checks (which do need content) even
    apply.
    """
    p = Path(path)
    if _is_in_skipped_dir(p.parent):
        return False
    if p.name in SKIP_FILE_NAMES:
        return False
    if p.suffix:
        return p.suffix.lower() in EXTENSION_LANGUAGE
    return p.name.lower() in FILENAME_LANGUAGE


def iter_ingestable_files(root: Path) -> Iterator[tuple[str, str]]:
    """
    Walks root (the single directory a GitHub tarball extracts to),
    yielding (repo_relative_path, text) for every file that passes the
    extension/dir/size/binary filters. Paths use forward slashes
    regardless of platform, since they become doc_id/chunk_id and are
    compared against GitHub API paths (which are always POSIX-style).
    """
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        rel = path.relative_to(root)
        if not is_ingestable_path(rel.as_posix()):
            continue

        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable -- skip rather than fail the run

        strip_text = text.strip()
        if not strip_text:
            continue

        yield rel.as_posix(), strip_text


def document_for(path: str, text: str) -> Document:
    """
    Builds the Document a repo-relative path becomes. doc_id is the path
    itself -- unique within a repo's collection, which is all that's
    needed since each repo gets its own collection.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext:
        language = EXTENSION_LANGUAGE.get(ext)
        doc_type = "markdown" if ext in MARKDOWN_LIKE_EXTENSIONS else "code"
    else:
        name = p.name.lower()
        language = FILENAME_LANGUAGE.get(name)
        doc_type = "markdown" if name in DOC_FILENAMES else "code"
    return Document(
        doc_id=path,
        title=path,
        text=text,
        doc_type=doc_type,
        source_path=path,
        tags=[],
        language=language,
    )
