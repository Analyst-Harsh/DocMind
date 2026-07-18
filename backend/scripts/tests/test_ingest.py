from app.ingestion.chunker import ChunkStrategy
from scripts.ingest import resolve_targets


def test_all_excludes_code_strategy():
    targets = resolve_targets("all")
    assert ChunkStrategy.CODE.value not in targets
    assert ChunkStrategy.RECURSIVE.value in targets
    assert ChunkStrategy.FIXED_SIZE.value in targets
    assert ChunkStrategy.STRUCTURE_AWARE.value in targets


def test_single_strategy_still_resolves():
    assert resolve_targets("recursive") == ["recursive"]


def test_code_strategy_can_still_be_targeted_explicitly():
    assert resolve_targets("code") == ["code"]
