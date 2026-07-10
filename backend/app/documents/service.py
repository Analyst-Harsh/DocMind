import re
from collections import Counter
from pathlib import Path

import yaml
from structlog import get_logger

from app.caching.cache import get_semantic_cache
from app.ingestion.indexer import (
    HYBRID_MODEL,
    HYBRID_STRATEGY,
    collection_name_for,
    get_qdrant_client,
)
from app.ingestion.loader import CORPUS_DIR, Document, load_manifest
from scripts.ingest import ingest_strategy

log = get_logger(__name__)

MANIFEST_PATH = CORPUS_DIR / "manifest.yaml"
UPLOADS_DIR = CORPUS_DIR / "uploads"


def slugify(stem: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    return slug or "document"


def is_duplicate_doc_id(doc_id: str) -> bool:
    return any(doc["doc_id"] == doc_id for doc in load_manifest())


def save_upload(filename: str, content: bytes) -> Path:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOADS_DIR / filename
    dest.write_bytes(content)
    return dest


def append_manifest_entry(entry: dict) -> None:
    """
    Reads the existing manifest and appends one entry, preserving every
    prior entry -- unlike download_corpus.py's write_manifest(), which
    overwrites the file from a hardcoded list.
    """
    manifest = load_manifest()
    if any(doc["doc_id"] == entry["doc_id"] for doc in manifest):
        raise ValueError(
            f"doc_id '{entry['doc_id']}' already exists in manifest"
        )
    manifest.append(entry)
    with open(MANIFEST_PATH, "w") as f:
        yaml.dump(
            {"documents": manifest},
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


def list_documents_with_chunk_counts() -> list[dict]:
    manifest = load_manifest()
    client = get_qdrant_client()
    collection = collection_name_for(HYBRID_STRATEGY, HYBRID_MODEL, hybrid=True)

    counts: Counter[str] = Counter()
    existing_collections = [c.name for c in client.get_collections().collections]
    if collection in existing_collections:
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection,
                limit=256,
                offset=offset,
                with_payload=["doc_id"],
                with_vectors=False,
            )
            counts.update(
                p.payload["doc_id"] for p in points if p.payload
            )
            if offset is None:
                break

    return [
        {
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "type": doc["type"],
            "tags": doc.get("tags", []),
            "chunk_count": counts.get(doc["doc_id"], 0),
        }
        for doc in manifest
    ]


def ingest_uploaded_document(document: Document) -> int:
    """
    Ingests one document into the same collection main.py's /query and
    /query/stream actually search -- see HYBRID_STRATEGY/HYBRID_MODEL in
    app/ingestion/indexer.py. Reuses scripts.ingest.ingest_strategy as-is;
    passing a single-document list only chunks/embeds/upserts that one
    doc, leaving every other document's points untouched.
    """
    client = get_qdrant_client()
    chunk_count = ingest_strategy(
        HYBRID_STRATEGY, [document], client, HYBRID_MODEL, hybrid=True
    )

    try:
        get_semantic_cache().flush()
    except Exception:
        # The doc is already searchable at this point -- a cache-flush
        # failure shouldn't fail the whole upload, just risk a stale
        # cached answer that predates this doc until the cache expires.
        log.warning(
            "semantic cache flush failed after upload ingest", exc_info=True
        )

    return chunk_count
