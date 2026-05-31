import json
from pathlib import Path
import unittest

from search import explain_top_match, rank_movies, rank_with_overlap_counts, score_spread


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

    def test_rank_with_overlap_counts(self) -> None:
        base = Path(__file__).resolve().parents[1]
        movies = json.loads((base / "movies.json").read_text(encoding="utf-8"))
        ranked = rank_with_overlap_counts(movies, "floating robots", top_k=1)
        self.assertEqual(ranked[0]["title"], "Skyward Robots")
        self.assertGreaterEqual(ranked[0]["overlap_count"], 1)

    def test_score_spread_non_negative(self) -> None:
        base = Path(__file__).resolve().parents[1]
        movies = json.loads((base / "movies.json").read_text(encoding="utf-8"))
        spread = score_spread(movies, "floating robots", top_k=3)
        self.assertGreaterEqual(spread, 0.0)


if __name__ == "__main__":
    unittest.main()
