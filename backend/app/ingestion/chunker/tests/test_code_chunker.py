from app.ingestion.chunker.code_chunker import CodeChunker


def test_python_function_stays_intact_when_it_fits_budget(make_doc):
    chunker = CodeChunker(chunk_size=500, chunk_overlap=50)
    src = (
        "import os\n\n\n"
        "def foo():\n    return os.getcwd()\n\n\n"
        "def bar():\n    return 42\n"
    )
    doc = make_doc(src, doc_type="code", language="python")
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1
    assert "def foo" in chunks[0].text
    assert "def bar" in chunks[0].text


def test_python_splits_on_def_boundary_when_oversized(make_doc):
    chunker = CodeChunker(chunk_size=15, chunk_overlap=0)
    src = (
        "def alpha():\n    " + ("x = 1\n    " * 10) + "return x\n\n\n"
        "def beta():\n    " + ("y = 2\n    " * 10) + "return y\n"
    )
    doc = make_doc(src, doc_type="code", language="python")
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.text.strip() != ""


def test_javascript_uses_function_and_class_separators(make_doc):
    chunker = CodeChunker(chunk_size=10, chunk_overlap=0)
    src = "function one() {\n  return 1;\n}\n\nfunction two() {\n  return 2;\n}\n"
    doc = make_doc(src, doc_type="code", language="javascript")
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2
    joined = "\n".join(c.text for c in chunks)
    assert "function one" in joined
    assert "function two" in joined


def test_go_uses_func_type_separators(make_doc):
    chunker = CodeChunker(chunk_size=10, chunk_overlap=0)
    src = "func One() int {\n  return 1\n}\n\ntype Foo struct {\n  X int\n}\n"
    doc = make_doc(src, doc_type="code", language="go")
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2


def test_markdown_uses_heading_separators(make_doc):
    chunker = CodeChunker(chunk_size=10, chunk_overlap=0)
    src = "## Section One\nsome text here\n\n## Section Two\nmore text here\n"
    doc = make_doc(src, doc_type="markdown", language="markdown")
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2


def test_unknown_language_falls_back_to_default_separators(make_doc):
    chunker = CodeChunker(chunk_size=500, chunk_overlap=50)
    doc = make_doc("some plain text", doc_type="code", language="cobol")
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1
    assert "some plain text" in chunks[0].text


def test_no_language_falls_back_to_default_separators(make_doc):
    chunker = CodeChunker(chunk_size=500, chunk_overlap=50)
    doc = make_doc("plain text, no language set", doc_type="text", language=None)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1


def test_empty_text_produces_no_chunks(make_doc):
    chunker = CodeChunker(chunk_size=100, chunk_overlap=10)
    doc = make_doc("", doc_type="code", language="python")
    chunks = chunker.chunk_document(doc)
    assert chunks == []


def test_keyword_separator_is_reattached_not_consumed(make_doc):
    # Splitting on "\ndef " must keep "def " attached to the piece that
    # follows -- plain str.split() would strip it, leaving a chunk that
    # starts mid-signature ("foo():" instead of "def foo():").
    chunker = CodeChunker(chunk_size=8, chunk_overlap=0)
    src = "def foo():\n    return 1\n\n\ndef bar():\n    return 2\n"
    doc = make_doc(src, doc_type="code", language="python")
    chunks = chunker.chunk_document(doc)
    assert any(c.text.strip().startswith("def foo") for c in chunks)
    assert any(c.text.strip().startswith("def bar") for c in chunks)


def test_whitespace_separators_are_not_reattached(make_doc):
    # Blank-line/word-level separators carry no semantic keyword, so they
    # should behave exactly like the base RecursiveChunker (consumed, not
    # reattached) -- no leading whitespace artifacts on later pieces.
    chunker = CodeChunker(chunk_size=3, chunk_overlap=0)
    doc = make_doc("alpha beta gamma delta", doc_type="code", language="c")
    chunks = chunker.chunk_document(doc)
    for chunk in chunks:
        assert not chunk.text.startswith(" ")
        assert not chunk.text.startswith("\n")


def test_chunk_metadata_matches_document(make_doc):
    chunker = CodeChunker(chunk_size=100, chunk_overlap=10)
    doc = make_doc(
        "def f():\n    pass\n",
        doc_type="code",
        language="python",
        doc_id="src/app/main.py",
    )
    chunks = chunker.chunk_document(doc)
    for chunk in chunks:
        assert chunk.doc_id == "src/app/main.py"
        assert chunk.chunking_strategy == "code"
        assert chunk.chunk_id == f"src/app/main.py_code_{chunk.chunk_index}"
