# app/mcp/instance.py
"""
The shared FastMCP server instance. Kept in its own module (no tool logic
here) so tools.py can `from app.mcp.instance import mcp` to register
tools, and server.py can import both `mcp` and `tools` (for its
registration side effect) without a circular import between the three.
"""

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import structlog
from fastmcp import FastMCP

from app.config import get_settings

# stdio is the JSON-RPC wire for this server (see server.py) -- every line
# written to stdout must be a protocol message, or the client's line-based
# parser fails to decode it. structlog's default logger factory writes to
# stdout, which corrupts that stream; redirect it to stderr before any tool
# can log. Must run at import time, before app.mcp.tools registers a logger.
structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=sys.stderr))


@dataclass
class AppContext:
    """No shared mutable state is needed today -- get_settings(),
    get_job_store(), and get_qdrant_client() are already process-local,
    lru_cache'd singletons that any tool can call directly."""


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    # Settings() has several required fields with no defaults (e.g.
    # openai_api_key, langfuse_secret_key/public_key/base_url) -- call
    # get_settings() eagerly so a misconfigured .env fails loudly at
    # server startup, not on the first tool call.
    get_settings()
    yield AppContext()


mcp = FastMCP("docmind", lifespan=lifespan)
