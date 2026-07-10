"""
Downloads all corpus documents and writes manifest.yaml.
Run once: python -m scripts.download_corpus
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import time

import certifi
import requests
import yaml

CORPUS_DIR = Path(__file__).parent.parent / "corpus"

# ── Document definitions ────────────────────────────────────────────────────
DOCUMENTS = [
    # PDFs
    {
        "doc_id": "rag-paper",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "url": "https://arxiv.org/pdf/2005.11401",
        "dest": "pdfs/rag_lewis_2020.pdf",
        "type": "pdf",
        "tags": ["rag", "retrieval", "generation", "foundational"],
    },
    {
        "doc_id": "attention-paper",
        "title": "Attention Is All You Need",
        "url": "https://arxiv.org/pdf/1706.03762",
        "dest": "pdfs/attention_vaswani_2017.pdf",
        "type": "pdf",
        "tags": ["transformer", "attention", "architecture", "foundational"],
    },
    {
        "doc_id": "ragas-paper",
        "title": "RAGAS: Automated Evaluation of Retrieval Augmented Generation",
        "url": "https://arxiv.org/pdf/2309.15217",
        "dest": "pdfs/ragas_es_2023.pdf",
        "type": "pdf",
        "tags": ["evaluation", "rag", "faithfulness", "metrics"],
    },
    # Markdown READMEs
    {
        "doc_id": "qdrant-readme",
        "title": "Qdrant — Vector Database README",
        "url": "https://raw.githubusercontent.com/qdrant/qdrant/master/README.md",
        "dest": "repos/qdrant/README.md",
        "type": "markdown",
        "tags": ["qdrant", "vector-db", "similarity-search", "tooling"],
    },
    {
        "doc_id": "langfuse-readme",
        "title": "Langfuse — LLM Observability README",
        "url": "https://raw.githubusercontent.com/langfuse/langfuse/main/README.md",
        "dest": "repos/langfuse/README.md",
        "type": "markdown",
        "tags": [
            "langfuse",
            "observability",
            "tracing",
            "evaluation",
            "tooling",
        ],
    },
    {
        "doc_id": "fastapi-readme",
        "title": "FastAPI README",
        "url": "https://raw.githubusercontent.com/tiangolo/fastapi/master/README.md",
        "dest": "repos/fastapi/README.md",
        "type": "markdown",
        "tags": ["fastapi", "api", "python", "tooling"],
    },
    {
        "doc_id": "ragas-readme",
        "title": "RAGAS Library README",
        "url": "https://raw.githubusercontent.com/explodinggradients/ragas/main/README.md",
        "dest": "repos/ragas/README.md",
        "type": "markdown",
        "tags": ["ragas", "evaluation", "rag", "metrics", "tooling"],
    },
    {
        "doc_id": "tiktoken-readme",
        "title": "tiktoken README",
        "url": "https://raw.githubusercontent.com/openai/tiktoken/main/README.md",
        "dest": "repos/tiktoken/README.md",
        "type": "markdown",
        "tags": ["tokenization", "openai", "tokens", "tooling"],
    },
]


def download_file(url: str, dest_path: Path) -> bool:
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        print(f"  ✓ Already exists: {dest_path.name}")
        return True

    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DocMind/1.0)"},
            verify=certifi.where(),  # use certifi's up-to-date CA bundle
            timeout=30,
            allow_redirects=True,  # arXiv redirects /pdf/ID → actual PDF
        )
        response.raise_for_status()
        dest_path.write_bytes(response.content)
        size_kb = len(response.content) / 1024
        print(f"  ↓ Downloaded: {dest_path.name} ({size_kb:.0f} KB)")
        return True

    except Exception as e:
        print(f"  ✗ Failed: {dest_path.name} — {e}")
        return False


def write_manifest(documents: list[dict]) -> None:
    """
    Write manifest.yaml from the document definitions.
    Excludes the 'url' key — that's only needed for downloading.
    """
    manifest_entries = []
    for doc in documents:
        entry = {
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "path": doc["dest"],
            "type": doc["type"],
            "tags": doc["tags"],
        }
        manifest_entries.append(entry)

    manifest_path = CORPUS_DIR / "manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump(
            {"documents": manifest_entries},
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    print(f"\nWrote manifest.yaml with {len(manifest_entries)} entries")


def print_corpus_stats() -> None:
    """Print a summary of what's in the corpus directory."""
    total_size = 0
    total_files = 0
    for path in CORPUS_DIR.rglob("*"):
        if path.is_file() and path.suffix in (".pdf", ".md", ".py", ".ts"):
            total_size += path.stat().st_size
            total_files += 1

    print("\n── Corpus stats ──────────────────────────")
    print(f"  Files: {total_files}")
    print(f"  Total size: {total_size / 1024 / 1024:.1f} MB")
    print(f"  Location: {CORPUS_DIR}")
    print("───────────────────────────────────────────")


def main() -> None:
    print("=== DocMind Corpus Downloader ===\n")
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = []

    for doc in DOCUMENTS:
        print(f"[{doc['doc_id']}]")
        dest_path = CORPUS_DIR / doc["dest"]
        ok = download_file(doc["url"], dest_path)
        if ok:
            success += 1
        else:
            failed.append(doc["doc_id"])
        # be polite to servers — small delay between downloads
        time.sleep(0.5)

    write_manifest(DOCUMENTS)
    print_corpus_stats()

    print(f"\nDone: {success}/{len(DOCUMENTS)} documents downloaded")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        print("Re-run the script to retry failed downloads.")


if __name__ == "__main__":
    main()
