from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest


@contextmanager
def _noop_span(*args, **kwargs):
    """Yields a MagicMock so .update(**kw) calls are silently accepted."""
    yield MagicMock()


@pytest.fixture(autouse=True)
def disable_tracing(monkeypatch):
    """Patch traced_span in every agent module to avoid Langfuse connections."""
    # Wrap in try/except to handle modules that don't exist yet
    try:
        monkeypatch.setattr("app.agent.sufficiency.traced_span", _noop_span)
    except (AttributeError, ModuleNotFoundError, ImportError):
        pass

    try:
        monkeypatch.setattr("app.agent.reformulation.traced_span", _noop_span)
    except (AttributeError, ModuleNotFoundError, ImportError):
        pass

    try:
        monkeypatch.setattr("app.agent.loop.traced_span", _noop_span)
    except (AttributeError, ModuleNotFoundError, ImportError):
        pass
