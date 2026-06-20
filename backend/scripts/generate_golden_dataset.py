# scripts/generate_golden_dataset.py
"""
LLM-assisted draft of the golden query set. For each corpus doc, ask the
model for Q&A pairs with verbatim supporting quotes, validate each quote
is actually in the doc, and write a draft YAML for human review.

Usage:
  python -m scripts.generate_golden_dataset
  python -m scripts.generate_golden_dataset --docs attention-paper,rag-paper --per-doc 5
"""

import argparse
import json
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI

from app.config import get_settings
from app.eval.golden_dataset import load_corpus_texts
from app.eval.matcher import normalize
from structlog import get_logger

log = get_logger()
client = OpenAI()

PROMPT = """You are building an evaluation set for a retrieval system.

Given the document below (id: {doc_id}), write {per_doc} diverse questions a
user might ask that are answerable ONLY from this document. For each question,
provide one short VERBATIM quote (a single sentence or phrase, copied exactly,
NOT spanning paragraph or page breaks) from the document that supports the
answer.

Return ONLY a JSON array, each element: {{"query": "...", "snippet": "..."}}.
The snippet must be copied character-for-character from the document text.

DOCUMENT TEXT:
{doc_text}
"""


@dataclass
class RawPair:
    query: str
    doc_id: str
    snippet: str


def partition_validated(
    pairs: list[RawPair], doc_texts: dict[str, str]
) -> tuple[list[RawPair], list[RawPair]]:
    normalized_docs = {
        doc_id: normalize(text) for doc_id, text in doc_texts.items()
    }
    valid, needs_review = [], []
    for p in pairs:
        if (
            p.doc_id in normalized_docs
            and normalize(p.snippet) in normalized_docs[p.doc_id]
        ):
            valid.append(p)
        else:
            needs_review.append(p)
    return valid, needs_review


def to_yaml(valid: list[RawPair], needs_review: list[RawPair]) -> str:
    grouped: "OrderedDict[str, list[RawPair]]" = OrderedDict()
    for p in valid:
        grouped.setdefault(p.query, []).append(p)

    data = {
        "queries": [
            {
                "query": query,
                "items": [
                    {"doc_id": it.doc_id, "snippet": it.snippet} for it in items
                ],
            }
            for query, items in grouped.items()
        ]
    }
    output = yaml.dump(data, sort_keys=False, allow_unicode=True)

    if needs_review:
        output += "\n# NEEDS REVIEW — quote did not validate against the source doc.\n"
        output += "# Fix the snippet (or discard) before moving these into 'queries'.\n"
        for p in needs_review:
            output += f"#   query: {p.query!r}\n"
            output += f"#     doc_id: {p.doc_id}\n"
            output += f"#     snippet: {p.snippet!r}\n"

    return output


def _draft_for_doc(
    doc_id: str, doc_text: str, per_doc: int, model: str
) -> list[RawPair]:
    """
    Ask the LLM to draft Q&A pairs for a single doc, returning only those that
    pass validation (snippet is actually in the doc). If the LLM output is not
    valid JSON, log a warning and return an empty list.
    """
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": PROMPT.format(
                    doc_id=doc_id, per_doc=per_doc, doc_text=doc_text
                ),
            }
        ],
    )
    content = response.choices[0].message.content or "[]"
    content = (
        content.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
        items = json.loads(content)
    except json.JSONDecodeError as e:
        log.warning(
            f"could not parse LLM output for {doc_id}, skipping", error=e
        )
        return []
    return [
        RawPair(query=i["query"], doc_id=doc_id, snippet=i["snippet"])
        for i in items
        if "query" in i and "snippet" in i
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Draft the golden query set.")
    parser.add_argument(
        "--docs", default="all", help="all | comma-separated doc_ids"
    )
    parser.add_argument(
        "--per-doc",
        type=int,
        default=5,
        help="number of Q&A pairs to draft per doc",
    )
    parser.add_argument(
        "--output",
        default="eval/golden_dataset.draft.yaml",
        help="output file path",
    )
    args = parser.parse_args()

    settings = get_settings()
    doc_texts = load_corpus_texts()

    # selected doc_ids to draft for, either all or a comma-separated list
    selected = (
        list(doc_texts)
        if args.docs == "all"
        else [d.strip() for d in args.docs.split(",")]
    )

    all_pairs: list[RawPair] = []
    for doc_id in selected:
        if doc_id not in doc_texts:
            log.info(f"  Skipping unknown doc_id: {doc_id}")
            continue
        log.info(f"Drafting {args.per_doc} pairs for {doc_id}...")
        all_pairs.extend(
            _draft_for_doc(
                doc_id,
                doc_texts[doc_id],
                args.per_doc,
                settings.draft_generate_llm_model,
            )
        )

    valid, needs_review = partition_validated(all_pairs, doc_texts)
    log.info(f"\nValid: {len(valid)}  Needs review: {len(needs_review)}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(to_yaml(valid, needs_review), encoding="utf-8")
    log.info(f"Wrote draft to {out_path} — review it before use.")


if __name__ == "__main__":
    main()
