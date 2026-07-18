from unittest.mock import MagicMock

from qdrant_client import models

from app.ingestion.chunker.base_chunker import Chunk
from app.ingestion.indexer import (
    delete_repo_points_by_path,
    ensure_repo_payload_indexes,
    sweep_stale_points_for_path,
    sweep_stale_repo_points,
    upsert_repo_chunks_hybrid,
)


def make_chunk(text: str, doc_id: str = "src/main.py", chunk_id: str = "src/main.py_code_0") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_title=doc_id,
        text=text,
        token_count=len(text.split()),
        chunk_index=0,
        doc_type="code",
        source_path=doc_id,
        tags=[],
        chunking_strategy="code",
    )


def test_ensure_repo_payload_indexes_creates_commit_sha_and_path():
    client = MagicMock()

    ensure_repo_payload_indexes(client, "docmind_repo_octo-hello_hybrid")

    assert client.create_payload_index.call_count == 2
    field_names = {
        call.kwargs["field_name"] for call in client.create_payload_index.call_args_list
    }
    assert field_names == {"commit_sha", "path"}


def test_upsert_repo_chunks_hybrid_sets_repo_payload_fields():
    client = MagicMock()
    chunk = make_chunk("def f(): pass")
    chunk_embeddings = [(chunk, [0.1, 0.2])]
    sparse_embeddings = [
        (chunk, models.SparseVector(indices=[1], values=[0.5]))
    ]

    upsert_repo_chunks_hybrid(
        client,
        "docmind_repo_octo-hello_hybrid",
        chunk_embeddings,
        sparse_embeddings,
        repo="octo/hello",
        ref="main",
        commit_sha="abc123",
        ingested_at="2026-07-17T00:00:00+00:00",
        languages={"src/main.py": "python"},
    )

    client.upsert.assert_called_once()
    _, kwargs = client.upsert.call_args
    point = kwargs["points"][0]
    assert point.payload["repo"] == "octo/hello"
    assert point.payload["path"] == "src/main.py"
    assert point.payload["ref"] == "main"
    assert point.payload["commit_sha"] == "abc123"
    assert point.payload["language"] == "python"
    assert point.payload["ingested_at"] == "2026-07-17T00:00:00+00:00"


def test_upsert_repo_chunks_hybrid_missing_language_is_none():
    client = MagicMock()
    chunk = make_chunk("plain text", doc_id="README.txt")
    chunk_embeddings = [(chunk, [0.1])]
    sparse_embeddings = [(chunk, models.SparseVector(indices=[], values=[]))]

    upsert_repo_chunks_hybrid(
        client,
        "coll",
        chunk_embeddings,
        sparse_embeddings,
        repo="octo/hello",
        ref="main",
        commit_sha="abc123",
        ingested_at="2026-07-17T00:00:00+00:00",
        languages={},
    )

    point = client.upsert.call_args.kwargs["points"][0]
    assert point.payload["language"] is None


def test_sweep_stale_repo_points_deletes_by_commit_sha_filter():
    client = MagicMock()
    before = MagicMock(points_count=10)
    after = MagicMock(points_count=6)
    client.get_collection.side_effect = [before, after]

    deleted = sweep_stale_repo_points(client, "coll", "new-sha")

    assert deleted == 4
    client.delete.assert_called_once()
    _, kwargs = client.delete.call_args
    assert kwargs["collection_name"] == "coll"
    selector = kwargs["points_selector"]
    assert selector.must_not[0].key == "commit_sha"
    assert selector.must_not[0].match.value == "new-sha"


def test_delete_repo_points_by_path_deletes_matching_points():
    client = MagicMock()
    client.count.return_value = MagicMock(count=3)

    deleted = delete_repo_points_by_path(client, "coll", "src/old.py")

    assert deleted == 3
    client.delete.assert_called_once()
    _, kwargs = client.delete.call_args
    assert kwargs["collection_name"] == "coll"
    assert kwargs["points_selector"].must[0].key == "path"
    assert kwargs["points_selector"].must[0].match.value == "src/old.py"


def test_delete_repo_points_by_path_skips_delete_when_no_matches():
    client = MagicMock()
    client.count.return_value = MagicMock(count=0)

    deleted = delete_repo_points_by_path(client, "coll", "src/never-existed.py")

    assert deleted == 0
    client.delete.assert_not_called()


def test_sweep_stale_points_for_path_filters_on_path_and_not_commit_sha():
    client = MagicMock()
    client.count.return_value = MagicMock(count=2)

    deleted = sweep_stale_points_for_path(client, "coll", "src/main.py", "new-sha")

    assert deleted == 2
    _, kwargs = client.delete.call_args
    selector = kwargs["points_selector"]
    assert selector.must[0].key == "path"
    assert selector.must[0].match.value == "src/main.py"
    assert selector.must_not[0].key == "commit_sha"
    assert selector.must_not[0].match.value == "new-sha"


def test_sweep_stale_points_for_path_skips_delete_when_no_matches():
    client = MagicMock()
    client.count.return_value = MagicMock(count=0)

    deleted = sweep_stale_points_for_path(client, "coll", "src/main.py", "new-sha")

    assert deleted == 0
    client.delete.assert_not_called()
