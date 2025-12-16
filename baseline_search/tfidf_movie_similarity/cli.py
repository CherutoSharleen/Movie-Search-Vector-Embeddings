from __future__ import annotations

import json
from pathlib import Path

from search import rank_movies


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    movies = json.loads((base_dir / "movies.json").read_text(encoding="utf-8"))
    print("TF-IDF Movie Similarity Browser")
    query = input("Describe a movie you want: ").strip()
    for title, score in rank_movies(movies, query, top_k=3):
        print(f"- {title} (score={score:.4f})")


if __name__ == "__main__":
    main()
