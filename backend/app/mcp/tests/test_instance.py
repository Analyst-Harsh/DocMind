import asyncio

from app.mcp import tools  # noqa: F401 -- import registers the 4 tools
from app.mcp.instance import mcp


def test_registers_exactly_the_four_scoped_tools():
    # mcp.list_tools() is async (SDK internals), but nothing else in this
    # codebase is -- asyncio.run() here avoids pulling in pytest-asyncio
    # as a new dev dependency for one smoke test.
    registered = asyncio.run(mcp.list_tools())
    assert {t.name for t in registered} == {
        "ingest_repo",
        "sync_repo_incremental",
        "get_ingest_status",
        "query_repo",
    }
