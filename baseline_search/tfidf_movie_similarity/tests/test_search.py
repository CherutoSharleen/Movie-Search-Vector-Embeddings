import json
from pathlib import Path
import unittest

from search import explain_top_match, rank_movies


class TfidfSearchTests(unittest.TestCase):
    def test_robot_query_ranks_robot_movie_first(self) -> None:
        base = Path(__file__).resolve().parents[1]
        movies = json.loads((base / "movies.json").read_text(encoding="utf-8"))
        ranked = rank_movies(movies, "robots in a floating city", top_k=3)
        self.assertEqual(ranked[0][0], "Skyward Robots")

    def test_returns_top_k(self) -> None:
        base = Path(__file__).resolve().parents[1]
        movies = json.loads((base / "movies.json").read_text(encoding="utf-8"))
        ranked = rank_movies(movies, "mystery signals", top_k=2)
        self.assertEqual(len(ranked), 2)

    def test_explain_top_match(self) -> None:
        base = Path(__file__).resolve().parents[1]
        movies = json.loads((base / "movies.json").read_text(encoding="utf-8"))
        explanation = explain_top_match(movies, "floating robots")
        self.assertEqual(explanation["title"], "Skyward Robots")
        self.assertIn("robots", explanation["overlap_terms"])


if __name__ == "__main__":
    unittest.main()
