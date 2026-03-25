# TF-IDF Movie Similarity Browser

This mini application teaches information retrieval basics by ranking fictional movies against a free-text query using TF-IDF vectors and cosine similarity.

## What it does

- Loads a small synthetic movie catalog.
- Converts movie text into TF-IDF vectors.
- Ranks movies by cosine similarity to a user query.

## Concepts

- Tokenization and term frequencies
- Inverse document frequency
- Vector-space retrieval
- Similarity scoring

## Install

No external dependencies are required.

```bash
python --version
```

## Run

```bash
cd baseline_search/tfidf_movie_similarity
python cli.py
```

## Test

```bash
cd baseline_search/tfidf_movie_similarity
python -m unittest discover -s tests -p "test_*.py"
```

## Example

Input:

```text
robots in a floating city
```

Output shape:

```text
- Skyward Robots (score=...)
- ...
```

## Main components

- `search.py`: TF-IDF construction and ranking logic
- `cli.py`: terminal browser entry point
- `movies.json`: fictional movie dataset
- `tests/test_search.py`: basic correctness checks

## Data note

All movies and descriptions are fictional/synthetic.

## Limitations and optional exercises

- No stemming or synonym handling.
- Small corpus can produce unstable rankings.
- Extend with phrase boosts or snippet highlighting.
