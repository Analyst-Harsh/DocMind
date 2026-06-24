from unittest.mock import ANY, MagicMock, patch

from qdrant_client import models

from app.retrieval.searcher import retrieve, retrieve_hybrid, retrieve_reranked


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


@patch("app.retrieval.searcher.embed_query")
def test_retrieve_skips_embed_query_when_query_vector_given(mock_embed_query):
    client = MagicMock()
    client.query_points.return_value.points = []

    retrieve("query", client=client, collection_name="coll", query_vector=[0.1, 0.2])

    mock_embed_query.assert_not_called()
    _, kwargs = client.query_points.call_args
    assert kwargs["query"] == [0.1, 0.2]


@patch("app.retrieval.searcher.embed_query_sparse")
@patch("app.retrieval.searcher.embed_query")
def test_retrieve_hybrid_skips_embed_query_when_query_vector_given(
    mock_embed_query, mock_embed_query_sparse
):
    mock_embed_query_sparse.return_value = models.SparseVector(
        indices=[1], values=[1.0]
    )
    client = MagicMock()
    client.query_points.return_value.points = []

    retrieve_hybrid(
        "what is bm25?", client=client, collection_name="coll", query_vector=[0.5, 0.5]
    )

    mock_embed_query.assert_not_called()
    _, kwargs = client.query_points.call_args
    dense_prefetch = next(p for p in kwargs["prefetch"] if p.using == "dense")
    assert dense_prefetch.query == [0.5, 0.5]


@patch("app.retrieval.searcher.rerank")
@patch("app.retrieval.searcher.retrieve_hybrid")
def test_retrieve_reranked_composes_hybrid_and_rerank(
    mock_retrieve_hybrid, mock_rerank
):
    mock_retrieve_hybrid.return_value = ["candidate-chunks"]
    mock_rerank.return_value = ["final-chunks"]

    result = retrieve_reranked(
        "query",
        top_k=5,
        client=MagicMock(),
        collection_name="coll",
        candidate_pool_size=20,
    )

    mock_retrieve_hybrid.assert_called_once_with(
        "query",
        top_k=10,
        client=ANY,
        collection_name="coll",
        embedding_model=None,
        prefetch_limit=20,
        query_vector=None,
    )
    mock_rerank.assert_called_once_with("query", ["candidate-chunks"], 5)
    assert result == ["final-chunks"]
