"""ML-100K loaders that produce the same DataFrame schema as data.py
(movie_id, user_id, rating, timestamp, plus formatted profile / movie /
genres_list / year / title_clean / gender_name / age_group / occupation_name)
so every predictor and prompt function already in the codebase works
unchanged on MovieLens 100K.
"""
from pathlib import Path
import numpy as np
import pandas as pd

import os
ML100K_DIR = Path(os.environ.get("ML100K_DIR", Path(__file__).resolve().parent.parent / "data" / "ml-100k"))

# Per ML-100K README
OCCUPATIONS_100K = {
    "administrator", "artist", "doctor", "educator", "engineer",
    "entertainment", "executive", "healthcare", "homemaker", "lawyer",
    "librarian", "marketing", "none", "other", "programmer", "retired",
    "salesman", "scientist", "student", "technician", "writer",
}

GENRES_100K = [
    "unknown", "Action", "Adventure", "Animation", "Children's",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]


def _age_group(age):
    if age < 18: return "Under 18"
    if age < 25: return "18-24"
    if age < 35: return "25-34"
    if age < 45: return "35-44"
    if age < 50: return "45-49"
    if age < 56: return "50-55"
    return "56+"


def load_movies():
    cols = ["movie_id", "title", "release_date", "video_release_date", "url"] + GENRES_100K
    df = pd.read_csv(ML100K_DIR / "u.item", sep="|", header=None, names=cols,
                     encoding="latin-1")
    df["year"] = df["title"].str.extract(r"\((\d{4})\)$").astype("Int64")
    df["title_clean"] = df["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)

    def to_list(row):
        out = [g for g in GENRES_100K if g != "unknown" and row[g] == 1]
        return out or ["unknown"]

    df["genres_list"] = df.apply(to_list, axis=1)
    df["genres"] = df["genres_list"].apply(lambda xs: "|".join(xs))
    keep = ["movie_id", "title", "genres", "year", "title_clean", "genres_list"]
    return df[keep].set_index("movie_id", drop=False)


def load_users():
    df = pd.read_csv(ML100K_DIR / "u.user", sep="|", header=None,
                     names=["user_id", "age", "gender", "occupation", "zip"],
                     encoding="latin-1")
    df["age_group"] = df["age"].apply(_age_group)
    df["occupation_name"] = df["occupation"]
    df["gender_name"] = df["gender"].map({"M": "male", "F": "female"})
    return df.set_index("user_id", drop=False)


def load_ratings_full():
    return pd.read_csv(ML100K_DIR / "u.data", sep="\t", header=None,
                       names=["user_id", "movie_id", "rating", "timestamp"])


def load_fold(fold="u1"):
    """Return (train_df, test_df) for one of the built-in folds u1..u5."""
    cols = ["user_id", "movie_id", "rating", "timestamp"]
    train = pd.read_csv(ML100K_DIR / f"{fold}.base", sep="\t", header=None, names=cols)
    test = pd.read_csv(ML100K_DIR / f"{fold}.test", sep="\t", header=None, names=cols)
    return train, test


def item_popularity(train):
    counts = train.groupby("movie_id").size()
    counts.name = "pop_count"
    return counts


def user_history(train, user_id, max_items=20, strategy="recent_and_top"):
    u = train[train.user_id == user_id]
    if u.empty:
        return u
    if strategy == "recent":
        return u.sort_values("timestamp", ascending=False).head(max_items)
    if strategy == "top":
        return u.sort_values(["rating", "timestamp"], ascending=[False, False]).head(max_items)
    half = max_items // 2
    top = u.sort_values(["rating", "timestamp"], ascending=[False, False]).head(half)
    recent = u.sort_values("timestamp", ascending=False).head(max_items - half)
    combined = pd.concat([top, recent]).drop_duplicates("movie_id")
    return combined.head(max_items)


def user_stats(train, user_id):
    u = train[train.user_id == user_id]
    if u.empty:
        return {"count": 0, "mean": 3.5, "std": 1.0,
                "pct": {k: 20.0 for k in range(1, 6)}}
    counts = u["rating"].value_counts().to_dict()
    n = len(u)
    pct = {k: round(100.0 * counts.get(k, 0) / n, 1) for k in range(1, 6)}
    return {
        "count": int(n),
        "mean": float(u["rating"].mean()),
        "std": float(u["rating"].std(ddof=0)),
        "pct": pct,
    }
