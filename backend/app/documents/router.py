from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.documents.service import (
    append_manifest_entry,
    ingest_uploaded_document,
    is_duplicate_doc_id,
    list_documents_with_chunk_counts,
    save_upload,
    slugify,
)
from app.ingestion.loader import load_document

router = APIRouter(prefix="/documents", tags=["documents"])

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
EXTENSION_TO_TYPE = {".pdf": "pdf", ".md": "markdown"}


class DocumentSummary(BaseModel):
    doc_id: str
    title: str
    type: str
    tags: list[str]
    chunk_count: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]


class UploadResponse(BaseModel):
    doc_id: str
    title: str
    type: str
    chunk_count: int


@router.get("", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    return DocumentListResponse(documents=list_documents_with_chunk_counts())


@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),  # noqa: B008 -- FastAPI's documented pattern
) -> UploadResponse:
    # Path(...).name strips any directory components a crafted filename
    # might carry, so this can never write outside corpus/uploads/.
    safe_filename = Path(file.filename or "").name
    suffix = Path(safe_filename).suffix.lower()
    doc_type = EXTENSION_TO_TYPE.get(suffix)
    if doc_type is None:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type — only .pdf and .md are accepted.",
        )

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large — max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    doc_id = slugify(Path(safe_filename).stem)
    if is_duplicate_doc_id(doc_id):
        raise HTTPException(
            status_code=409,
            detail=f"A document with doc_id '{doc_id}' already exists.",
        )

    save_upload(safe_filename, content)
    entry = {
        "doc_id": doc_id,
        "title": Path(safe_filename).stem,
        "path": f"uploads/{safe_filename}",
        "type": doc_type,
        "tags": [],
    }
    append_manifest_entry(entry)

    document = load_document(entry)
    try:
        chunk_count = ingest_uploaded_document(document)
    except Exception as e:
        # The doc is saved and cataloged (a re-run of scripts/ingest.py
        # would pick it up) even though this request reports failure --
        # no rollback, per the narrow scope of this endpoint.
        raise HTTPException(
            status_code=500,
            detail=f"Document saved and cataloged, but ingestion failed: {e}",
        ) from e

    return UploadResponse(
        doc_id=doc_id,
        title=entry["title"],
        type=doc_type,
        chunk_count=chunk_count,
    )
