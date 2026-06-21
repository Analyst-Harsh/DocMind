from unittest.mock import MagicMock, patch

from qdrant_client import models

from app.retrieval.searcher import retrieve_hybrid


def make_point(chunk_id="c0", doc_id="doc", text="hello", score=0.9):
    point = MagicMock()
    point.score = score
    point.payload = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "doc_title": "Doc",
        "text": text,
        "source_path": "doc.md",
        "chunk_index": 0,
    }
    return point


@patch("app.retrieval.searcher.embed_query_sparse")
@patch("app.retrieval.searcher.embed_query")
def test_retrieve_hybrid_builds_prefetch_and_fusion_query(
    mock_embed_query, mock_embed_query_sparse
):
    mock_embed_query.return_value = [0.1, 0.2]
    mock_embed_query_sparse.return_value = models.SparseVector(
        indices=[1, 2], values=[0.5, 0.5]
    )
    client = MagicMock()
    client.query_points.return_value.points = []

    retrieve_hybrid(
        "what is bm25?",
        top_k=5,
        client=client,
        collection_name="docmind_recursive_test_hybrid",
    )

    client.query_points.assert_called_once()
    _, kwargs = client.query_points.call_args
    assert kwargs["collection_name"] == "docmind_recursive_test_hybrid"
    prefetches = kwargs["prefetch"]
    assert {p.using for p in prefetches} == {"dense", "bm25"}
    assert kwargs["query"] == models.FusionQuery(fusion=models.Fusion.RRF)
    assert kwargs["limit"] == 5


@patch("app.retrieval.searcher.embed_query_sparse")
@patch("app.retrieval.searcher.embed_query")
def test_retrieve_hybrid_maps_points_to_retrieved_chunk(
    mock_embed_query, mock_embed_query_sparse
):
    mock_embed_query.return_value = [0.1, 0.2]
    mock_embed_query_sparse.return_value = models.SparseVector(
        indices=[1], values=[1.0]
    )
    client = MagicMock()
    client.query_points.return_value.points = [
        make_point(chunk_id="c0", text="hello world", score=0.95)
    ]

    results = retrieve_hybrid(
        "query", client=client, collection_name="coll"
    )

    assert len(results) == 1
    assert results[0].chunk_id == "c0"
    assert results[0].text == "hello world"
    assert results[0].score == 0.95
