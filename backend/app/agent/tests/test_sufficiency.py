import json
from unittest.mock import MagicMock, patch

import pytest

from app.agent.state import SufficiencyResult
from app.retrieval.searcher import RetrievedChunk


def _make_chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1", doc_id="d1", doc_title="Doc", text=text,
        score=0.9, source_path="f.pdf", chunk_index=0,
    )


def _mock_openai_response(payload: dict, prompt_tokens: int = 10, completion_tokens: int = 5) -> MagicMock:
    msg = MagicMock()
    msg.content = json.dumps(payload)
    choice = MagicMock()
    choice.message = msg
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def test_assess_sufficiency_returns_sufficient():
    from app.agent.sufficiency import assess_sufficiency

    payload = {
        "is_sufficient": True,
        "reasoning": "Context fully covers the question.",
        "missing_aspects": [],
        "confidence": "high",
    }
    with patch("app.agent.sufficiency.client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_openai_response(payload)
        result = assess_sufficiency("What is RAG?", [_make_chunk("RAG is ...")])

    assert isinstance(result, SufficiencyResult)
    assert result.is_sufficient is True
    assert result.confidence == "high"
    assert result.missing_aspects == []


def test_assess_sufficiency_returns_insufficient():
    from app.agent.sufficiency import assess_sufficiency

    payload = {
        "is_sufficient": False,
        "reasoning": "Missing how reranking works.",
        "missing_aspects": ["reranking mechanism", "cross-encoder scoring"],
        "confidence": "high",
    }
    with patch("app.agent.sufficiency.client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_openai_response(payload)
        result = assess_sufficiency("How does reranking work?", [_make_chunk("RAG is ...")])

    assert result.is_sufficient is False
    assert "reranking mechanism" in result.missing_aspects
    assert len(result.missing_aspects) == 2


def test_assess_sufficiency_raises_on_invalid_json():
    from app.agent.sufficiency import assess_sufficiency

    msg = MagicMock()
    msg.content = "not valid json at all"
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]

    with patch("app.agent.sufficiency.client") as mock_client:
        mock_client.chat.completions.create.return_value = resp
        with pytest.raises(json.JSONDecodeError):
            assess_sufficiency("question", [_make_chunk("text")])


def test_assess_sufficiency_calls_llm_with_temperature_zero():
    from app.agent.sufficiency import assess_sufficiency

    payload = {
        "is_sufficient": True,
        "reasoning": "ok",
        "missing_aspects": [],
        "confidence": "medium",
    }
    with patch("app.agent.sufficiency.client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_openai_response(payload)
        assess_sufficiency("q", [_make_chunk("text")])

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["temperature"] == 0.0


def test_assess_sufficiency_raises_on_none_content():
    from app.agent.sufficiency import assess_sufficiency

    msg = MagicMock()
    msg.content = None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    with patch("app.agent.sufficiency.client") as mock_client:
        mock_client.chat.completions.create.return_value = resp
        with pytest.raises(RuntimeError, match="missing content"):
            assess_sufficiency("q", [])
