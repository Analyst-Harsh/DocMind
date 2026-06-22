# scripts/validate_ragas_golden_set.py
"""
Runs each RAGAS golden set question through the current pipeline and prints
the system answer alongside the reference answer for manual comparison.
Do NOT skip this step — RAGAS scores on a broken golden set are meaningless.
"""

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.eval.golden_dataset import load_corpus_texts
from app.eval.ragas_dataset import load_ragas_dataset

GOLDEN_SET_PATH = Path("eval/ragas_dataset.yaml")
API_URL = "http://localhost:8000/query"


def main():
    items = load_ragas_dataset(GOLDEN_SET_PATH, load_corpus_texts())

    for i, item in enumerate(items, start=1):
        response = httpx.post(
            API_URL,
            json={"question": item.question, "top_k": 5, "hybrid": True},
            timeout=60.0,
        )
        result = response.json()

        print(f"\n{'=' * 60}")
        print(f"[{i}] ({item.category}) {item.question}")
        print(f"\nReference: {item.reference_answer}")
        print(f"\nSystem:    {result['answer']}")
        print(f"Sources:   {[s['doc_id'] for s in result['sources']]}")
        print(f"Expected:  {item.source_docs}")


if __name__ == "__main__":
    main()
