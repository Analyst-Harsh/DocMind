"""Golden query set: strategy-agnostic ground truth for chunking eval.

Ground truth is (doc_id, verbatim snippet). A snippet must be a
whitespace-normalized substring of its source document — enforced both
when the LLM drafts the set and when it's loaded, so a bad hand-edit or
hallucinated quote fails loudly instead of silently scoring as unfound.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from app.eval.matcher import normalize
from app.ingestion.loader import load_all_documents


@dataclass
class GoldenItem:
    doc_id: str
    snippet: str


@dataclass
class GoldenQuery:
    query: str
    items: list[GoldenItem]


def snippet_in_doc(snippet: str, doc_text: str) -> bool:
    """True if the snippet appears in the doc after whitespace normalization."""
    return normalize(snippet) in normalize(doc_text)


def load_corpus_texts() -> dict[str, str]:
    """Map every corpus doc_id to its raw extracted text."""
    return {doc.doc_id: doc.text for doc in load_all_documents()}


def load_golden_dataset(
    path: str | Path, doc_texts: dict[str, str]
) -> list[GoldenQuery]:
    """Load the golden query set from a YAML file, validating snippets."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    queries: list[GoldenQuery] = []
    normalized_docs = {
        doc_id: normalize(text) for doc_id, text in doc_texts.items()
    }

    for entry in raw["queries"]:
        query_text = entry["query"]
        items = [
            GoldenItem(doc_id=i["doc_id"], snippet=i["snippet"])
            for i in entry.get("items", [])
        ]
        if not items:
            raise ValueError(f"Golden query has no items: {query_text!r}")

        for item in items:
            if item.doc_id not in normalized_docs:
                raise ValueError(
                    f"Unknown doc_id {item.doc_id!r} in query {query_text!r}"
                )
            if not item.snippet.strip():
                raise ValueError(
                    f"Empty snippet for doc_id {item.doc_id!r} in query "
                    f"{query_text!r}"
                )
            if normalize(item.snippet) not in normalized_docs[item.doc_id]:
                raise ValueError(
                    f"Snippet not found in {item.doc_id!r} for query "
                    f"{query_text!r}: {item.snippet!r}"
                )

        queries.append(GoldenQuery(query=query_text, items=items))

    return queries
