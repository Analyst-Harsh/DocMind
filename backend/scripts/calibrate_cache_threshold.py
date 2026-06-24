"""
Calibrate the semantic cache similarity threshold against real
paraphrase / non-paraphrase pairs pulled from the golden set.

Usage:
  python -m scripts.calibrate_cache_threshold
  python -m scripts.calibrate_cache_threshold --pairs-file eval/cache_threshold_pairs.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from rich.console import Console
from rich.table import Table

from app.ingestion.embedder import embed_query


def cosine_similarity(a: list[float], b: list[float]) -> float:
    vec_a, vec_b = np.array(a), np.array(b)
    return float(
        np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate the semantic cache similarity threshold."
    )
    parser.add_argument(
        "--pairs-file", default="eval/cache_threshold_pairs.yaml"
    )
    parser.add_argument(
        "--output", default="eval/results/cache_threshold_calibration.json"
    )
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.pairs_file).read_text(encoding="utf-8"))
    console = Console()

    table = Table(title="Paraphrase pairs (should hit)")
    table.add_column("original")
    table.add_column("paraphrase")
    table.add_column("similarity")

    paraphrase_similarities: list[float] = []
    for pair in raw["paraphrase_pairs"]:
        emb_a = embed_query(pair["original"])
        emb_b = embed_query(pair["paraphrase"])
        sim = cosine_similarity(emb_a, emb_b)
        paraphrase_similarities.append(sim)
        table.add_row(
            pair["original"][:50], pair["paraphrase"][:50], f"{sim:.4f}"
        )
    console.print(table)

    table2 = Table(title="Non-paraphrase pairs (should miss)")
    table2.add_column("a")
    table2.add_column("b")
    table2.add_column("similarity")

    non_paraphrase_similarities: list[float] = []
    for pair in raw["non_paraphrase_pairs"]:
        emb_a = embed_query(pair["a"])
        emb_b = embed_query(pair["b"])
        sim = cosine_similarity(emb_a, emb_b)
        non_paraphrase_similarities.append(sim)
        table2.add_row(pair["a"][:50], pair["b"][:50], f"{sim:.4f}")
    console.print(table2)

    min_paraphrase = min(paraphrase_similarities)
    max_non_paraphrase = max(non_paraphrase_similarities)
    console.print(f"\nMin paraphrase similarity: {min_paraphrase:.4f}")
    console.print(f"Max non-paraphrase similarity: {max_non_paraphrase:.4f}")

    if min_paraphrase > max_non_paraphrase:
        recommended = (min_paraphrase + max_non_paraphrase) / 2
        console.print(
            f"[green]Clean separation. Recommended threshold: {recommended:.4f}[/green]"
        )
    else:
        recommended = max_non_paraphrase + 0.001
        console.print(
            f"[yellow]No clean separation (overlap region: "
            f"{max_non_paraphrase:.4f}-{min_paraphrase:.4f}). "
            f"Recommended threshold (favors precision): {recommended:.4f}[/yellow]"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "paraphrase_similarities": paraphrase_similarities,
                "non_paraphrase_similarities": non_paraphrase_similarities,
                "min_paraphrase_similarity": min_paraphrase,
                "max_non_paraphrase_similarity": max_non_paraphrase,
                "recommended_threshold": recommended,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    console.print(f"\nWrote calibration results to {output_path}")


if __name__ == "__main__":
    main()
