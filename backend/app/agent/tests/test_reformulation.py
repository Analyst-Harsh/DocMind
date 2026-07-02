from unittest.mock import MagicMock, patch


def _mock_response(text: str, prompt_tokens: int = 10, completion_tokens: int = 5) -> MagicMock:
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def test_reformulate_query_returns_stripped_text():
    from app.agent.reformulation import reformulate_query

    with patch("app.agent.reformulation.client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_response(
            "  cross-encoder reranking mechanism RAG  "
        )
        query, cost = reformulate_query("How does reranking work?", ["cross-encoder scoring"], [])

    assert query == "cross-encoder reranking mechanism RAG"
    assert isinstance(cost, float)


def test_reformulate_query_joins_multiple_aspects():
    from app.agent.reformulation import reformulate_query

    with patch("app.agent.reformulation.client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_response("query text")
        reformulate_query("question", ["aspect one", "aspect two"], [])

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    prompt_text = call_kwargs["messages"][0]["content"]
    assert "aspect one" in prompt_text
    assert "aspect two" in prompt_text


def test_reformulate_query_uses_temperature_zero():
    from app.agent.reformulation import reformulate_query

    with patch("app.agent.reformulation.client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_response("result")
        reformulate_query("q", ["missing thing"], [])

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["temperature"] == 0.0


def test_reformulate_query_includes_original_question_in_prompt():
    from app.agent.reformulation import reformulate_query

    with patch("app.agent.reformulation.client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_response("new query")
        reformulate_query("What is chunking strategy?", ["fixed-size vs recursive"], [])

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    prompt_text = call_kwargs["messages"][0]["content"]
    assert "What is chunking strategy?" in prompt_text
