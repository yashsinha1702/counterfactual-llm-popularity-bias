"""Embed all ML-1M movies with a simple but effective content representation:
TF-IDF over titles + one-hot genres + z-scored release year, concatenated and
L2-normalized. ~560-dim output. No external model dependencies.

Used by retrieval_utils.py to pick content-similar items from a user's
training history for retrieval-augmented abduction / diverse / CFv2 prompts.
"""
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from config import OUTPUT_DIR
from data import load_movies


def build():
    movies = load_movies()
    titles = movies["title_clean"].fillna("").astype(str).tolist()
    years = movies["year"].fillna(1990).astype(float).values
    genres_lists = movies["genres_list"].tolist()

    # TF-IDF on titles (1-2 grams)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=2000, stop_words="english",
                            lowercase=True, min_df=1)
    title_vec = tfidf.fit_transform(titles).toarray().astype(np.float32)

    # Genre one-hot (weight higher — genres are strong content signal)
    all_genres = sorted(set(g for gl in genres_lists for g in (gl or [])))
    g_idx = {g: i for i, g in enumerate(all_genres)}
    genre_vec = np.zeros((len(movies), len(all_genres)), dtype=np.float32)
    for i, gl in enumerate(genres_lists):
        for g in (gl or []):
            genre_vec[i, g_idx[g]] = 1.0
    genre_vec *= 3.0  # up-weight genre signal relative to tfidf

    # Year z-score (small weight — content similarity >> release year)
    mu_y = float(years.mean()); sd_y = float(years.std()) or 1.0
    year_vec = ((years - mu_y) / sd_y).reshape(-1, 1).astype(np.float32) * 0.5

    X = np.hstack([title_vec, genre_vec, year_vec])
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms
    movie_ids = movies.index.values.astype(np.int64)

    out_dir = Path(OUTPUT_DIR) / "embeddings"
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / "movies.npy", X)
    np.save(out_dir / "movie_ids.npy", movie_ids)
    print(f"Saved {X.shape} embeddings for {len(movies)} movies to {out_dir}")
    print(f"  tfidf dim={title_vec.shape[1]}  genres={len(all_genres)}  + year")
    return X, movie_ids


if __name__ == "__main__":
    build()
