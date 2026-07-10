from __future__ import annotations

from contextlib import contextmanager, suppress
from unittest.mock import MagicMock

import pytest


@contextmanager
def _noop_span(*args, **kwargs):
    """Yields a MagicMock so .update(**kw) calls are silently accepted."""
    yield MagicMock()


@pytest.fixture(autouse=True)
def disable_tracing(monkeypatch):
    """Patch traced_span in every agent module to avoid Langfuse connections."""
    # Suppress in case a module doesn't exist yet
    with suppress(AttributeError, ModuleNotFoundError, ImportError):
        monkeypatch.setattr("app.agent.sufficiency.traced_span", _noop_span)

    with suppress(AttributeError, ModuleNotFoundError, ImportError):
        monkeypatch.setattr("app.agent.reformulation.traced_span", _noop_span)

    with suppress(AttributeError, ModuleNotFoundError, ImportError):
        monkeypatch.setattr("app.agent.loop.traced_span", _noop_span)
