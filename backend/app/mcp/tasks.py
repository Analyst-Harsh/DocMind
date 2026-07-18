# app/mcp/tasks.py
"""
Fire-and-forget scheduling for run_full_ingest/run_incremental_ingest,
called from the MCP tools instead of FastAPI's BackgroundTasks (which
needs a request/response cycle this server doesn't have).
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from app.config import get_settings

_executor = ThreadPoolExecutor(
    max_workers=get_settings().mcp_ingest_max_concurrency,
    thread_name_prefix="mcp-ingest",
)


def fire_and_forget(task_fn: Callable[..., None], *args: object) -> None:
    """Submits task_fn(*args) to a small bounded worker pool and returns
    immediately. run_full_ingest/run_incremental_ingest already catch
    Exception, mark the job failed, and release the lock in `finally` --
    nothing here needs to observe the future's result.

    A bounded ThreadPoolExecutor (not a raw Thread per call) caps how many
    repos can be embedding concurrently -- an MCP client can plausibly
    fire tool calls faster than a human hitting a REST endpoint, and each
    concurrent ingest is real, paid OpenAI spend. Unlike asyncio.create_task,
    the executor itself holds the strong reference to in-flight work, so
    there's no fire-and-forget-task-gc gotcha to guard against either.
    """
    _executor.submit(task_fn, *args)
