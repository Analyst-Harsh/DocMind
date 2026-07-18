from unittest.mock import patch

from fastapi.testclient import TestClient

from app.query.service import (
    NoRelevantChunksError,
    QueryResult,
    RepoNotIngestedError,
)
from main import app

client = TestClient(app)


@patch("main.run_query")
def test_query_with_repo_returns_result_from_run_query(mock_run_query):
    mock_run_query.return_value = QueryResult(
        answer="it's in src/main.py",
        sources=[{"chunk_id": "c1", "doc_id": "src/main.py"}],
        cost_usd=0.001,
        latency_ms=42,
        trace_id="trace-1",
        cache_hit=False,
    )

    resp = client.post(
        "/query", json={"question": "where is main?", "repo": "octo/hello"}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "it's in src/main.py"
    mock_run_query.assert_called_once_with(
        question="where is main?", top_k=5, hybrid=True, repo="octo/hello"
    )


@patch("main.run_query")
def test_query_with_uningested_repo_returns_404(mock_run_query):
    mock_run_query.side_effect = RepoNotIngestedError("octo/missing")

    resp = client.post(
        "/query", json={"question": "where is main?", "repo": "octo/missing"}
    )

    assert resp.status_code == 404
    assert "octo/missing" in resp.json()["detail"]


@patch("main.run_query")
def test_query_with_no_relevant_chunks_returns_404(mock_run_query):
    mock_run_query.side_effect = NoRelevantChunksError()

    resp = client.post("/query", json={"question": "anything"})

    assert resp.status_code == 404
    assert resp.json()["detail"] == "No relevant documents found"
