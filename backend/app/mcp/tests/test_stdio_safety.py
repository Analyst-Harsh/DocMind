"""Regression tests for app.mcp.instance's structlog-to-stderr fix.

MCP's stdio transport reads server stdout line-by-line and parses each line
as a JSON-RPC message (see mcp.client.stdio's stdout_reader). structlog's
default logger factory writes to stdout, which silently corrupts that
stream -- reproduced by spawning the real server and calling a tool, which
made the client log "Failed to parse JSONRPC message from server" before
this fix. These tests pin the fix at two levels: the global config, and an
actual tool call's real (unmocked) logging behavior.

Note: the second test patches structlog._output.stdout.write rather than
using pytest's capfd -- capfd is fd-level and empirically racy in this suite
whenever an earlier test has driven an asyncio event loop (e.g.
test_instance.py's asyncio.run), which can delay the flush past the next
test's readouterr() snapshot. Patching the write method sidesteps buffering
and event-loop timing entirely.
"""

from unittest.mock import patch

import pytest
import structlog

from app.mcp import (
    instance,  # noqa: F401 -- import triggers structlog.configure()
    tools,
)


def test_structlog_logger_factory_targets_stderr_not_stdout():
    factory = structlog.get_config()["logger_factory"]
    logger = factory()

    assert logger._file is structlog._output.stderr
    assert logger._file is not structlog._output.stdout


def test_tool_call_logging_never_writes_to_stdout():
    with (
        patch.object(structlog._output.stdout, "write") as mock_stdout_write,
        patch("app.mcp.tools.get_job_store") as mock_get_store,
    ):
        mock_get_store.return_value.get_job.return_value = None
        with pytest.raises(ValueError):
            tools.get_ingest_status(job_id="missing")

    mock_stdout_write.assert_not_called()
