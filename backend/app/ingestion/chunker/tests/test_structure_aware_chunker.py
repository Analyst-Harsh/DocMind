from app.ingestion.chunker.structure_aware_chunker import StructureAwareChunker


def test_markdown_splits_on_headers(make_doc):
    text = (
        "Intro text.\n\n"
        "# Getting Started\n"
        "Install the package.\n\n"
        "## Usage\n"
        "Run the CLI.\n\n"
        "# FAQ\n"
        "None yet.\n"
    )
    chunker = StructureAwareChunker(chunk_size=500, chunk_overlap=50)
    doc = make_doc(text, doc_type="markdown")
    chunks = chunker.chunk_document(doc)
    texts = [c.text for c in chunks]
    assert any("Intro text" in t for t in texts)
    assert any("# Getting Started" in t and "Install the package" in t for t in texts)
    assert any("## Usage" in t for t in texts)
    assert any("# FAQ" in t for t in texts)


def test_markdown_ignores_hash_inside_fenced_code_block(make_doc):
    text = (
        "# Real Header\n"
        "Some text.\n\n"
        "```python\n"
        "# this is just a comment, not a header\n"
        "print('hi')\n"
        "```\n\n"
        "## Another Real Header\n"
        "More text.\n"
    )
    chunker = StructureAwareChunker(chunk_size=500, chunk_overlap=50)
    doc = make_doc(text, doc_type="markdown")
    chunks = chunker.chunk_document(doc)
    texts = [c.text for c in chunks]
    # the fenced code block's '#' comment must not create its own section
    assert any("this is just a comment" in t for t in texts)
    assert not any(t.strip().startswith("# this is just a comment") for t in texts)


def test_markdown_with_no_headers_returns_single_section(make_doc):
    text = "Just a plain paragraph with no markdown headers at all."
    chunker = StructureAwareChunker(chunk_size=500, chunk_overlap=50)
    doc = make_doc(text, doc_type="markdown")
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].text.strip() == text


def test_python_splits_per_top_level_function_and_class(make_doc):
    text = (
        "import os\n\n"
        "CONFIG = {'x': 1}\n\n"
        "@cache\n"
        "def helper():\n"
        "    '''Helper docstring.'''\n"
        "    return 1\n\n"
        "class Foo:\n"
        "    def bar(self):\n"
        "        pass\n\n"
        "if __name__ == '__main__':\n"
        "    helper()\n"
    )
    chunker = StructureAwareChunker(chunk_size=500, chunk_overlap=50)
    doc = make_doc(text, doc_type="code", language="python")
    chunks = chunker.chunk_document(doc)
    texts = [c.text for c in chunks]
    assert any("import os" in t and "CONFIG" in t for t in texts)
    assert any("@cache" in t and "def helper" in t for t in texts)
    assert any("class Foo" in t and "def bar" in t for t in texts)
    assert any("__main__" in t for t in texts)


def test_python_decorator_stays_with_its_function(make_doc):
    text = "@decorator_one\n@decorator_two\ndef foo():\n    pass\n"
    chunker = StructureAwareChunker(chunk_size=500, chunk_overlap=50)
    doc = make_doc(text, doc_type="code", language="python")
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1
    assert "@decorator_one" in chunks[0].text
    assert "@decorator_two" in chunks[0].text
    assert "def foo" in chunks[0].text


def test_python_unparseable_falls_back_to_plain_text(make_doc):
    text = "def broken(:\n  this is not valid python\n"
    chunker = StructureAwareChunker(chunk_size=500, chunk_overlap=50)
    doc = make_doc(text, doc_type="code", language="python")
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_large_section_gets_recursively_split(make_doc):
    # a single markdown section far bigger than chunk_size must still
    # get broken down rather than emitted as one oversized chunk
    chunker = StructureAwareChunker(chunk_size=20, chunk_overlap=5)
    text = "# Header\n" + ("word " * 200)
    doc = make_doc(text, doc_type="markdown")
    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 1


def test_non_markdown_non_python_falls_back_to_recursive_split(make_doc):
    chunker = StructureAwareChunker(chunk_size=20, chunk_overlap=5)
    doc = make_doc("word " * 100, doc_type="pdf")
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2


def test_chunk_metadata_matches_document(make_doc):
    chunker = StructureAwareChunker(chunk_size=500, chunk_overlap=50)
    doc = make_doc("# Title\nSome content.", doc_type="markdown", doc_id="my-doc")
    chunks = chunker.chunk_document(doc)
    for chunk in chunks:
        assert chunk.doc_id == "my-doc"
        assert chunk.chunking_strategy == "structure_aware"
