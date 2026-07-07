"""RAGAS evaluation set: question + paraphrased reference answer pairs.

Unlike golden_dataset.py's verbatim (doc_id, snippet) ground truth used for
retrieval precision/recall, this dataset feeds generation-quality metrics
(faithfulness, answer relevancy, context precision/recall) which need a
reference answer phrased independently of the source text.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

REFUSAL_ANSWER = (
    "I don't have enough information in the provided documents to answer this."
)
ALLOWED_CATEGORIES = {
    "factual_single_doc",
    "multi_doc_synthesis",
    "not_in_corpus",
    "table_only",
    "diagram_only",
    "hybrid",
}
MIN_REFERENCE_WORDS = 8
MAX_REFERENCE_WORDS = 60


@dataclass
class RagasItem:
    question: str
    reference_answer: str
    source_docs: list[str]
    category: str


def load_ragas_dataset(
    path: str | Path, doc_texts: dict[str, str]
) -> list[RagasItem]:
    """Load the RAGAS query set from a YAML file, validating its structure."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    items: list[RagasItem] = []
    seen_questions: set[str] = set()

    for entry in raw["items"]:
        question = entry["question"]
        reference_answer = entry["reference_answer"]
        source_docs = entry.get("source_docs", [])
        category = entry["category"]

        if category not in ALLOWED_CATEGORIES:
            raise ValueError(
                f"Unknown category {category!r} for question {question!r}"
            )
        if question in seen_questions:
            raise ValueError(f"Duplicate question: {question!r}")
        seen_questions.add(question)

        for doc_id in source_docs:
            if doc_id not in doc_texts:
                raise ValueError(
                    f"Unknown doc_id {doc_id!r} in question {question!r}"
                )

        if category == "multi_doc_synthesis" and len(source_docs) < 2:
            raise ValueError(
                f"multi_doc_synthesis question needs >=2 source_docs: "
                f"{question!r}"
            )

        if category == "not_in_corpus":
            if source_docs:
                raise ValueError(
                    f"not_in_corpus question must have no source_docs: "
                    f"{question!r}"
                )
            if reference_answer != REFUSAL_ANSWER:
                raise ValueError(
                    f"not_in_corpus question must use the refusal answer: "
                    f"{question!r}"
                )
        else:
            word_count = len(reference_answer.split())
            if not (MIN_REFERENCE_WORDS <= word_count <= MAX_REFERENCE_WORDS):
                raise ValueError(
                    f"reference_answer for {question!r} has {word_count} "
                    f"words, expected {MIN_REFERENCE_WORDS}-"
                    f"{MAX_REFERENCE_WORDS}"
                )

        items.append(
            RagasItem(
                question=question,
                reference_answer=reference_answer,
                source_docs=source_docs,
                category=category,
            )
        )

    return items
