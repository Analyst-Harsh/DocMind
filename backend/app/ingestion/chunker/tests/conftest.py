import pytest

from app.ingestion.loader import Document


@pytest.fixture
def make_doc():
    def _make_doc(
        text: str,
        doc_type: str = "markdown",
        language: str | None = None,
        doc_id: str = "test-doc",
    ) -> Document:
        return Document(
            doc_id=doc_id,
            title="Test Document",
            text=text,
            doc_type=doc_type,
            source_path="/fake/path",
            tags=["test"],
            language=language,
        )

    return _make_doc
