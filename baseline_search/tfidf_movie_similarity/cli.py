from __future__ import annotations

import json
from pathlib import Path

from search import rank_with_overlap_counts


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    movies = json.loads((base_dir / "movies.json").read_text(encoding="utf-8"))
    print("TF-IDF Movie Similarity Browser")
    query = input("Describe a movie you want: ").strip()
    for row in rank_with_overlap_counts(movies, query, top_k=3):
        terms = ", ".join(row["overlap_terms"]) if row["overlap_terms"] else "(none)"
        print(f"- {row['title']} (score={row['score']:.4f}, overlap={row['overlap_count']})")
        print(f"  terms: {terms}")


if __name__ == "__main__":
    main()
