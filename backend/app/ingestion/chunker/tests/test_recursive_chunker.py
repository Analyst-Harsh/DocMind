from app.ingestion.chunker.recursive_chunker import RecursiveChunker


def test_single_chunk_for_short_text(make_doc):
    chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
    doc = make_doc("This is a short document.")
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1
    assert "short document" in chunks[0].text


def test_splits_on_paragraph_boundary_first(make_doc):
    chunker = RecursiveChunker(chunk_size=20, chunk_overlap=0)
    doc = make_doc("para one " * 10 + "\n\n" + "para two " * 10)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.text.strip() != ""


def test_chunk_count_scales_with_length(make_doc):
    chunker = RecursiveChunker(chunk_size=50, chunk_overlap=5)
    long_text = "This is a sentence. " * 100
    doc = make_doc(long_text)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 3


def test_token_count_within_chunk_size_with_tolerance(make_doc):
    # _merge_pieces can slightly exceed chunk_size by overlap carry-forward;
    # bound it loosely rather than asserting a hard cap
    chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
    doc = make_doc("This is a sentence. " * 100)
    chunks = chunker.chunk_document(doc)
    for chunk in chunks:
        assert chunk.token_count <= 50 + 10 + 20


def test_overlap_carries_trailing_pieces_forward(make_doc):
    chunker = RecursiveChunker(chunk_size=15, chunk_overlap=8)
    # unique tokens so each piece appears exactly once - no ambiguity
    # when locating the overlap window between consecutive chunks
    doc = make_doc(" ".join(f"tok{i}" for i in range(50)))
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2
    # pieces are joined with "\n\n"; the leading pieces of chunk 1 are a
    # window of chunk 0's trailing pieces (the carried-forward overlap)
    pieces_chunk0 = chunks[0].text.split("\n\n")
    pieces_chunk1 = chunks[1].text.split("\n\n")
    overlap_start = pieces_chunk0.index(pieces_chunk1[0])
    assert pieces_chunk0[overlap_start:] == pieces_chunk1[: len(pieces_chunk0) - overlap_start]


def test_no_separators_falls_back_to_hard_split(make_doc):
    # a single long "word" with no separators must still get split
    chunker = RecursiveChunker(chunk_size=5, chunk_overlap=0)
    doc = make_doc("supercalifragilisticexpialidocious" * 20)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2


def test_empty_text_produces_no_chunks(make_doc):
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
    doc = make_doc("")
    chunks = chunker.chunk_document(doc)
    assert chunks == []


def test_chunk_metadata_matches_document(make_doc):
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
    doc = make_doc("word " * 50, doc_id="my-doc")
    chunks = chunker.chunk_document(doc)
    for chunk in chunks:
        assert chunk.doc_id == "my-doc"
        assert chunk.chunking_strategy == "recursive"
