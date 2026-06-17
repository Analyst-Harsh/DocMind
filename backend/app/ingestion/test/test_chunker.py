# tests/test_chunker.py
import pytest
from app.ingestion.chunker import FixedSizeChunker
from app.ingestion.loader import Document


def make_doc(text: str) -> Document:
    return Document(
        doc_id="test-doc",
        title="Test Document",
        text=text,
        doc_type="markdown",
        source_path="/fake/path.md",
        tags=["test"],
    )


def test_single_chunk_for_short_text():
    chunker = FixedSizeChunker(chunk_size=500, chunk_overlap=50)
    doc = make_doc("This is a short document.")
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "test-doc_0"
    assert "short document" in chunks[0].text


def test_chunk_count_scales_with_length():
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
    # ~300 tokens worth of text should produce 3+ chunks
    long_text = "word " * 300
    doc = make_doc(long_text)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 3


def test_overlap_means_boundary_text_in_two_chunks():
    chunker = FixedSizeChunker(chunk_size=20, chunk_overlap=5)
    doc = make_doc("word " * 50)
    chunks = chunker.chunk_document(doc)
    # Text near the boundary of chunk 0 and chunk 1 should appear in both
    if len(chunks) >= 2:
        # Last few tokens of chunk 0 should appear in chunk 1
        last_words_chunk0 = chunks[0].text.split()[-3:]
        first_words_chunk1 = chunks[1].text.split()[:8]
        overlap_found = any(w in first_words_chunk1 for w in last_words_chunk0)
        assert overlap_found


def test_chunk_ids_are_unique():
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
    doc = make_doc("word " * 500)
    chunks = chunker.chunk_document(doc)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_token_count_within_chunk_size():
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
    doc = make_doc("word " * 500)
    chunks = chunker.chunk_document(doc)
    for chunk in chunks:
        assert chunk.token_count <= 100
