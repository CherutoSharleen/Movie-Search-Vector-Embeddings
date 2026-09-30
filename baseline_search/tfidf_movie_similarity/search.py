from __future__ import annotations

import math
import re
from collections import Counter


TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def build_tfidf_vectors(docs: list[str]) -> tuple[list[dict[str, float]], dict[str, float]]:
    tokenized = [tokenize(doc) for doc in docs]
    doc_freq: Counter[str] = Counter()
    for tokens in tokenized:
        for token in set(tokens):
            doc_freq[token] += 1

    total_docs = len(docs)
    idf = {token: math.log((1 + total_docs) / (1 + freq)) + 1 for token, freq in doc_freq.items()}

    vectors: list[dict[str, float]] = []
    for tokens in tokenized:
        counts = Counter(tokens)
        length = len(tokens) or 1
        tfidf = {token: (count / length) * idf[token] for token, count in counts.items()}
        vectors.append(tfidf)
    return vectors, idf


def cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    overlap = set(a.keys()) & set(b.keys())
    dot = sum(a[t] * b[t] for t in overlap)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def query_vector(query: str, idf: dict[str, float]) -> dict[str, float]:
    tokens = tokenize(query)
    if not tokens:
        return {}
    counts = Counter(tokens)
    length = len(tokens)
    return {token: (count / length) * idf.get(token, 0.0) for token, count in counts.items()}


def rank_movies(movies: list[dict[str, str]], query: str, top_k: int = 3) -> list[tuple[str, float]]:
    docs = [f"{movie['title']} {movie['description']}" for movie in movies]
    vectors, idf = build_tfidf_vectors(docs)
    qv = query_vector(query, idf)
    scored = [(movie["title"], cosine_similarity(qv, vector)) for movie, vector in zip(movies, vectors)]
    scored.sort(key=lambda row: row[1], reverse=True)
    return scored[:top_k]


def explain_top_match(movies: list[dict[str, str]], query: str) -> dict[str, object]:
    ranked = rank_movies(movies, query, top_k=1)
    if not ranked:
        return {"title": None, "overlap_terms": []}
    top_title, score = ranked[0]
    query_terms = set(tokenize(query))
    top_doc = next(movie for movie in movies if movie["title"] == top_title)
    doc_terms = set(tokenize(f"{top_doc['title']} {top_doc['description']}"))
    return {"title": top_title, "score": score, "overlap_terms": sorted(query_terms & doc_terms)}


def rank_with_overlap_counts(
    movies: list[dict[str, str]],
    query: str,
    top_k: int = 3,
) -> list[dict[str, object]]:
    ranked = rank_movies(movies, query, top_k=top_k)
    query_terms = set(tokenize(query))
    output: list[dict[str, object]] = []
    for title, score in ranked:
        movie = next(item for item in movies if item["title"] == title)
        terms = set(tokenize(f"{movie['title']} {movie['description']}"))
        overlap = sorted(query_terms & terms)
        output.append({"title": title, "score": score, "overlap_count": len(overlap), "overlap_terms": overlap})
    return output


def score_spread(movies: list[dict[str, str]], query: str, top_k: int = 3) -> float:
    ranked = rank_movies(movies, query, top_k=top_k)
    if len(ranked) < 2:
        return 0.0
    return ranked[0][1] - ranked[-1][1]


def query_terms(query: str) -> list[str]:
    return sorted(set(tokenize(query)))


def has_query_terms(query: str) -> bool:
    return len(query_terms(query)) > 0
