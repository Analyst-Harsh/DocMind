from app.ingestion.chunker.fixed_size_chunker import FixedSizeChunker


def test_single_chunk_for_short_text(make_doc):
    chunker = FixedSizeChunker(chunk_size=500, chunk_overlap=50)
    doc = make_doc("This is a short document.")
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "test-doc_fixed_size_0"
    assert "short document" in chunks[0].text


def test_chunk_count_scales_with_length(make_doc):
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
    # ~300 tokens worth of text should produce 3+ chunks
    long_text = "word " * 300
    doc = make_doc(long_text)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 3


def test_overlap_means_boundary_text_in_two_chunks(make_doc):
    chunker = FixedSizeChunker(chunk_size=20, chunk_overlap=5)
    doc = make_doc("word " * 50)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2
    last_words_chunk0 = chunks[0].text.split()[-3:]
    first_words_chunk1 = chunks[1].text.split()[:8]
    overlap_found = any(w in first_words_chunk1 for w in last_words_chunk0)
    assert overlap_found


def test_chunk_ids_are_unique(make_doc):
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
    doc = make_doc("word " * 500)
    chunks = chunker.chunk_document(doc)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_token_count_within_chunk_size(make_doc):
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
    doc = make_doc("word " * 500)
    chunks = chunker.chunk_document(doc)
    for chunk in chunks:
        assert chunk.token_count <= 100


def test_empty_text_produces_no_chunks(make_doc):
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
    doc = make_doc("")
    chunks = chunker.chunk_document(doc)
    assert chunks == []


def test_chunk_metadata_matches_document(make_doc):
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
    doc = make_doc("word " * 50, doc_id="my-doc")
    chunks = chunker.chunk_document(doc)
    for chunk in chunks:
        assert chunk.doc_id == "my-doc"
        assert chunk.doc_title == "Test Document"
        assert chunk.doc_type == "markdown"
        assert chunk.chunking_strategy == "fixed_size"
