from pathlib import Path
from dataclasses import dataclass

import yaml
import fitz  # PyMuPDF

CORPUS_DIR = Path(__file__).parent.parent.parent / "corpus"


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    doc_type: str  # markdown, code, pdf
    source_path: str
    tags: list[str]
    language: str | None = None  # for code files


def load_manifest() -> list[dict]:
    manifest_path = CORPUS_DIR / "manifest.yaml"
    with open(manifest_path) as f:
        return yaml.safe_load(f)["documents"]


def load_document(entry: dict) -> Document:
    file_path = CORPUS_DIR / entry["path"]
    doc_type = entry["type"]

    if doc_type in ("markdown", "code"):
        text = file_path.read_text(encoding="utf-8")
    elif doc_type == "pdf":
        text = _extract_pdf_text(file_path)
    else:
        raise ValueError(f"Unknown document type: {doc_type}")

    return Document(
        doc_id=entry["doc_id"],
        title=entry["title"],
        text=text,
        doc_type=doc_type,
        source_path=str(file_path),
        tags=entry.get("tags", []),
        language=entry.get("language"),
    )


def _extract_pdf_text(path: Path) -> str:
    doc = fitz.open(path)
    pages = []
    for page in doc:
        # extracts text from the page and appends it to the pages list
        pages.append(page.get_text())
    return "\n\n".join(pages)


def load_all_documents() -> list[Document]:
    manifest = load_manifest()
    documents = []
    for entry in manifest:
        doc = load_document(entry)
        print(f"Loaded: {doc.doc_id} ({len(doc.text)} chars)")
        documents.append(doc)
    return documents
