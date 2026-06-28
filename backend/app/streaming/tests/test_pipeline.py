import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from app.caching.schema import CachedResponse

SOURCES = [
    {"doc_title": "Test Doc", "chunk_index": 0, "score": 0.9, "doc_id": "doc1"}
]


@contextmanager
def _noop_span(*args, **kwargs):
    yield MagicMock()


def _make_openai_stream(
    tokens: list[str], prompt_tokens: int = 80, completion_tokens: int = 10
):
    """Build an iterator of mock OpenAI stream chunks ending with a usage chunk."""
    chunks = []
    for token in tokens:
        c = MagicMock()
        c.choices = [MagicMock()]
        c.choices[0].delta.content = token
        c.usage = None
        chunks.append(c)
    usage_chunk = MagicMock()
    usage_chunk.choices = []
    usage_chunk.usage = MagicMock()
    usage_chunk.usage.prompt_tokens = prompt_tokens
    usage_chunk.usage.completion_tokens = completion_tokens
    chunks.append(usage_chunk)
    return iter(chunks)


def _call_pipeline(**overrides):
    """Call stream_query_pipeline with safe defaults, collect all events."""
    from app.streaming.pipeline import stream_query_pipeline

    defaults = dict(
        question="What is RAG?",
        chunks=[],
        cache_hit=None,
        trace_id="test-trace-abc",
        start_time=0.0,
        query_vector=[0.0] * 1536,
        sources=SOURCES,
        resolved_model="text-embedding-3-small",
        retrieval_mode="hybrid_rerank",
    )
    defaults.update(overrides)
    return list(stream_query_pipeline(**defaults))


# ── Cache hit tests ──────────────────────────────────────────────────────────


@patch("app.streaming.pipeline.flush_traces")
@patch("app.streaming.pipeline.root_span", _noop_span)
def test_cache_hit_emits_token_done_metadata(mock_flush):
    cache_hit = CachedResponse(
        answer="Cached answer.", sources=SOURCES, cost_usd=0.001
    )
    events = _call_pipeline(cache_hit=cache_hit)

    assert len(events) == 3
    assert events[0] == "event: token\ndata: Cached answer.\n\n"
    assert events[1] == "event: done\ndata: \n\n"
    assert events[2].startswith("event: metadata\ndata: ")
    mock_flush.assert_called_once()


@patch("app.streaming.pipeline.flush_traces")
@patch("app.streaming.pipeline.root_span", _noop_span)
def test_cache_hit_metadata_fields(mock_flush):
    cache_hit = CachedResponse(
        answer="Cached answer.", sources=SOURCES, cost_usd=0.001
    )
    events = _call_pipeline(cache_hit=cache_hit, trace_id="trace-xyz")

    meta = json.loads(events[2].split("data: ", 1)[1])
    assert meta["cache_hit"] is True
    assert meta["cost_usd"] == 0.0
    assert meta["trace_id"] == "trace-xyz"
    assert meta["sources"] == SOURCES
    assert "latency_ms" in meta


# ── Cache miss tests ─────────────────────────────────────────────────────────


@patch("app.streaming.pipeline.build_qa_prompt", return_value="Test prompt")
@patch("app.streaming.pipeline.flush_traces")
@patch("app.streaming.pipeline.traced_span", _noop_span)
@patch("app.streaming.pipeline.root_span", _noop_span)
@patch("app.streaming.pipeline.settings")
@patch("app.streaming.pipeline.client")
def test_cache_miss_yields_token_events(
    mock_client, mock_settings, mock_flush, mock_prompt
):
    mock_settings.llm_model = "gpt-4o-mini"
    mock_settings.enable_semantic_cache = False
    mock_client.chat.completions.create.return_value = _make_openai_stream(
        ["Hello", " world", "!"]
    )

    events = _call_pipeline(chunks=[MagicMock()])

    token_events = [e for e in events if e.startswith("event: token")]
    assert token_events == [
        "event: token\ndata: Hello\n\n",
        "event: token\ndata:  world\n\n",
        "event: token\ndata: !\n\n",
    ]


@patch("app.streaming.pipeline.build_qa_prompt", return_value="Test prompt")
@patch("app.streaming.pipeline.flush_traces")
@patch("app.streaming.pipeline.traced_span", _noop_span)
@patch("app.streaming.pipeline.root_span", _noop_span)
@patch("app.streaming.pipeline.settings")
@patch("app.streaming.pipeline.client")
def test_cache_miss_yields_done_and_metadata(
    mock_client, mock_settings, mock_flush, mock_prompt
):
    mock_settings.llm_model = "gpt-4o-mini"
    mock_settings.enable_semantic_cache = False
    mock_client.chat.completions.create.return_value = _make_openai_stream(
        ["Hi"]
    )

    events = _call_pipeline(chunks=[MagicMock()], trace_id="t-456")

    done_events = [e for e in events if e.startswith("event: done")]
    meta_events = [e for e in events if e.startswith("event: metadata")]
    assert len(done_events) == 1
    assert len(meta_events) == 1

    meta = json.loads(meta_events[0].split("data: ", 1)[1])
    assert meta["cache_hit"] is False
    assert meta["trace_id"] == "t-456"
    assert meta["sources"] == SOURCES


@patch("app.streaming.pipeline.build_qa_prompt", return_value="Test prompt")
@patch("app.streaming.pipeline.flush_traces")
@patch("app.streaming.pipeline.traced_span", _noop_span)
@patch("app.streaming.pipeline.root_span", _noop_span)
@patch("app.streaming.pipeline.settings")
@patch("app.streaming.pipeline.client")
def test_cache_miss_computes_cost_from_usage(
    mock_client, mock_settings, mock_flush, mock_prompt
):
    mock_settings.llm_model = "gpt-4o-mini"
    mock_settings.enable_semantic_cache = False
    mock_client.chat.completions.create.return_value = _make_openai_stream(
        ["Hi"], prompt_tokens=100, completion_tokens=50
    )

    events = _call_pipeline(chunks=[MagicMock()])

    meta_events = [e for e in events if e.startswith("event: metadata")]
    meta = json.loads(meta_events[0].split("data: ", 1)[1])

    from app.generation.generator import (
        COST_PER_INPUT_TOKEN,
        COST_PER_OUTPUT_TOKEN,
    )

    expected = 100 * COST_PER_INPUT_TOKEN + 50 * COST_PER_OUTPUT_TOKEN
    assert abs(meta["cost_usd"] - expected) < 1e-10


@patch("app.streaming.pipeline.build_qa_prompt", return_value="Test prompt")
@patch("app.streaming.pipeline.flush_traces")
@patch("app.streaming.pipeline.traced_span", _noop_span)
@patch("app.streaming.pipeline.root_span", _noop_span)
@patch("app.streaming.pipeline.settings")
@patch("app.streaming.pipeline.client")
def test_none_delta_content_is_skipped(
    mock_client, mock_settings, mock_flush, mock_prompt
):
    """First chunk from OpenAI often has delta.content=None (role-only chunk)."""
    mock_settings.llm_model = "gpt-4o-mini"
    mock_settings.enable_semantic_cache = False

    none_chunk = MagicMock()
    none_chunk.choices = [MagicMock()]
    none_chunk.choices[0].delta.content = None
    none_chunk.usage = None

    real_chunk = MagicMock()
    real_chunk.choices = [MagicMock()]
    real_chunk.choices[0].delta.content = "Hello"
    real_chunk.usage = None

    usage_chunk = MagicMock()
    usage_chunk.choices = []
    usage_chunk.usage = MagicMock()
    usage_chunk.usage.prompt_tokens = 10
    usage_chunk.usage.completion_tokens = 1

    mock_client.chat.completions.create.return_value = iter(
        [none_chunk, real_chunk, usage_chunk]
    )

    events = _call_pipeline(chunks=[MagicMock()])
    token_events = [e for e in events if e.startswith("event: token")]
    assert token_events == ["event: token\ndata: Hello\n\n"]


# ── Event ordering ────────────────────────────────────────────────────────────


@patch("app.streaming.pipeline.build_qa_prompt", return_value="Test prompt")
@patch("app.streaming.pipeline.flush_traces")
@patch("app.streaming.pipeline.traced_span", _noop_span)
@patch("app.streaming.pipeline.root_span", _noop_span)
@patch("app.streaming.pipeline.settings")
@patch("app.streaming.pipeline.client")
def test_event_ordering_is_tokens_then_done_then_metadata(
    mock_client, mock_settings, mock_flush, mock_prompt
):
    mock_settings.llm_model = "gpt-4o-mini"
    mock_settings.enable_semantic_cache = False
    mock_client.chat.completions.create.return_value = _make_openai_stream(
        ["A", "B"]
    )

    events = _call_pipeline(chunks=[MagicMock()])

    types = []
    for e in events:
        if e.startswith("event: token"):
            types.append("token")
        elif e.startswith("event: done"):
            types.append("done")
        elif e.startswith("event: metadata"):
            types.append("metadata")

    assert types == ["token", "token", "done", "metadata"]
