# Movie-Search-Vector-Embeddings

This repository contains small, local educational search projects for learning retrieval fundamentals without paid APIs or cloud services.

## Current runnable project

`baseline_search/tfidf_movie_similarity`

This mini application demonstrates:

- TF-IDF feature construction from movie text
- cosine-similarity ranking
- local CLI-based retrieval over a fictional movie dataset

## Installation

No external dependencies are required for the TF-IDF mini project.

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

## Representative input and output

Input:

```text
robots in a floating city
```

Output shape:

```text
- Skyward Robots (score=...)
- ...
```

## Data note

All movie entries in the runnable mini project are fictional and synthetic.

## Limitations and learning exercises

- The corpus is intentionally small.
- Rankings are lexical and do not use neural embeddings.
- You can extend this by adding stemming, synonym expansion, or snippet generation.

## Educational mini-project in this repository

- **Path:** `baseline_search/tfidf_movie_similarity`
- **Focus:** TF-IDF movie retrieval with overlap explanations and query normalization helpers.
- **Run:** `cd baseline_search/tfidf_movie_similarity && python cli.py`
- **Test:** `cd baseline_search/tfidf_movie_similarity && python -m unittest discover -s tests -p "test_*.py"`
