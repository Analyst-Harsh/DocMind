"""Decide whether a retrieved chunk satisfies a golden item.

A chunk is relevant to a golden item when they share a doc_id and the
item's snippet fuzzily appears inside the chunk text. Both sides are
whitespace-normalized first so PDF line wraps / page breaks don't defeat
an otherwise-exact match. rapidfuzz.partial_ratio is used because it
scores the best-matching substring of the (longer) chunk text against
the (shorter) snippet — exactly the "is this snippet contained here?"
question.
"""

from typing import Protocol

from rapidfuzz import fuzz

# Calibrated via scripts/calibrate_match_threshold.py against the golden
# dataset: verbatim-substring matches always score 100, unrelated chunk
# noise tops out at ~91. 92 sits just above that noise ceiling while
# leaving headroom below 100 for snippets split across a chunk boundary.
MATCH_THRESHOLD = 90


class _HasDocAndSnippet(Protocol):
    doc_id: str
    snippet: str


class _HasDocAndText(Protocol):
    doc_id: str
    text: str


def normalize(text: str) -> str:
    """Lowercase and collapse all whitespace runs to single spaces."""
    return " ".join(text.lower().split())


def is_relevant(
    item: _HasDocAndSnippet,
    chunk: _HasDocAndText,
    threshold: int = MATCH_THRESHOLD,
) -> bool:
    """Return True if the chunk is relevant to the golden item."""
    if item.doc_id != chunk.doc_id:
        return False
    return (
        fuzz.partial_ratio(normalize(item.snippet), normalize(chunk.text))
        >= threshold
    )
