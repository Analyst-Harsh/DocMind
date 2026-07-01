import pytest
from app.generation.prompts import get_registry
from app.retrieval.searcher import RetrievedChunk


def _make_chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1",
        doc_id="d1",
        doc_title="Test Doc",
        text=text,
        score=0.9,
        source_path="test.pdf",
        chunk_index=0,
    )


def test_sufficiency_assessment_prompt_renders():
    result = get_registry().render(
        "sufficiency_assessment",
        question="What is RAG?",
        chunks=[_make_chunk("RAG stands for Retrieval-Augmented Generation.")],
    )
    assert "is_sufficient" in result
    assert "missing_aspects" in result
    assert "What is RAG?" in result


def test_sufficiency_assessment_empty_chunks():
    result = get_registry().render(
        "sufficiency_assessment",
        question="What is RAG?",
        chunks=[],
    )
    assert "is_sufficient" in result


def test_query_reformulation_prompt_renders():
    result = get_registry().render(
        "query_reformulation",
        original_question="What is RAG?",
        missing_aspects=["definition of retrieval", "how generation works"],
    )
    assert "definition of retrieval" in result
    assert "how generation works" in result
    assert "What is RAG?" in result


def test_query_reformulation_single_aspect():
    result = get_registry().render(
        "query_reformulation",
        original_question="Explain chunking",
        missing_aspects=["fixed-size vs recursive chunking comparison"],
    )
    assert "fixed-size vs recursive chunking comparison" in result
