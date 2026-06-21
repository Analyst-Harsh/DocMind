from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from langfuse.types import TraceContext

from app.config import get_settings
from app.tracing.langfuse import get_langfuse


class NoOpObservation:
    """Stand-in observation used when tracing is disabled; .update() is a no-op."""

    def update(self, **_kwargs: Any) -> None:
        pass


@contextmanager
def root_span(name: str, trace_id: str, input: Any = None) -> Iterator[Any]:
    """Top-level span for a request, linked to trace_id. Caller still calls flush_traces() after."""
    if not get_settings().enable_tracing:
        yield NoOpObservation()
        return
    trace_context: TraceContext = {"trace_id": trace_id}
    with get_langfuse().start_as_current_observation(
        trace_context=trace_context, name=name, as_type="span", input=input
    ) as span:
        yield span


@contextmanager
def traced_span(name: str, as_type: str = "span", **kwargs: Any) -> Iterator[Any]:
    """Child span nested under whatever span is currently active (root_span or another traced_span)."""
    if not get_settings().enable_tracing:
        yield NoOpObservation()
        return
    # as_type is a plain str here by design (traced_span is generic over every
    # observation type); the SDK's overloads require a Literal per as_type value,
    # which a generic wrapper can never statically satisfy.
    with get_langfuse().start_as_current_observation(  # type: ignore[call-overload]
        name=name, as_type=as_type, **kwargs
    ) as span:
        yield span


def new_trace_id() -> str:
    if not get_settings().enable_tracing:
        return "tracing-disabled"
    return get_langfuse().create_trace_id()


def flush_traces() -> None:
    if get_settings().enable_tracing:
        get_langfuse().flush()
