from unittest.mock import MagicMock, patch

import pytest

from app.retrieval.reranker import _load_reranker_model, rerank
from app.retrieval.searcher import RetrievedChunk


@pytest.fixture(autouse=True)
def clear_reranker_cache():
    _load_reranker_model.cache_clear()
    yield
    _load_reranker_model.cache_clear()


def make_chunk(chunk_id: str, text: str, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="doc",
        doc_title="Doc",
        text=text,
        score=score,
        source_path="doc.md",
        chunk_index=0,
    )


@patch("app.retrieval.reranker.CrossEncoder")
def test_rerank_sorts_by_cross_encoder_score_descending(mock_ce_cls):
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.2, 0.9, 0.5]
    mock_ce_cls.return_value = mock_model

    chunks = [
        make_chunk("a", "low relevance", score=0.99),
        make_chunk("b", "high relevance", score=0.10),
        make_chunk("c", "mid relevance", score=0.50),
    ]

    result = rerank("query", chunks, top_k=3)

    assert [c.chunk_id for c in result] == ["b", "c", "a"]
    mock_model.predict.assert_called_once_with(
        [
            ("query", "low relevance"),
            ("query", "high relevance"),
            ("query", "mid relevance"),
        ]
    )


@patch("app.retrieval.reranker.CrossEncoder")
def test_rerank_truncates_to_top_k(mock_ce_cls):
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.1, 0.9, 0.5, 0.3]
    mock_ce_cls.return_value = mock_model

    chunks = [make_chunk(str(i), f"text {i}") for i in range(4)]

    result = rerank("query", chunks, top_k=2)

    assert [c.chunk_id for c in result] == ["1", "2"]


@patch("app.retrieval.reranker.CrossEncoder")
def test_rerank_overwrites_score_with_cross_encoder_score(mock_ce_cls):
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.42]
    mock_ce_cls.return_value = mock_model

    chunks = [make_chunk("a", "text", score=0.99)]

    result = rerank("query", chunks, top_k=1)

    assert result[0].score == pytest.approx(0.42)


def test_rerank_empty_chunks_returns_empty():
    assert rerank("query", [], top_k=5) == []
