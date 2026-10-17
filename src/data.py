import numpy as np
import pandas as pd

from config import (
    DATA_DIR, TEST_RATIO, HISTORY_MAX,
    TIPS_WINDOW_DAYS, TIPS_CLIP_MAX,
    AGE_GROUPS, OCCUPATIONS,
)


def load_movies():
    df = pd.read_csv(
        DATA_DIR / "movies.dat", sep="::", engine="python",
        names=["movie_id", "title", "genres"], encoding="latin-1",
    )
    df["year"] = df["title"].str.extract(r"\((\d{4})\)$").astype("Int64")
    df["title_clean"] = df["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
    df["genres_list"] = df["genres"].str.split("|")
    return df.set_index("movie_id", drop=False)


def load_users():
    df = pd.read_csv(
        DATA_DIR / "users.dat", sep="::", engine="python",
        names=["user_id", "gender", "age", "occupation", "zip"], encoding="latin-1",
    )
    df["age_group"] = df["age"].map(AGE_GROUPS)
    df["occupation_name"] = df["occupation"].map(OCCUPATIONS)
    df["gender_name"] = df["gender"].map({"M": "male", "F": "female"})
    return df.set_index("user_id", drop=False)


def load_ratings():
    return pd.read_csv(
        DATA_DIR / "ratings.dat", sep="::", engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"], encoding="latin-1",
    )


def temporal_split(ratings, test_ratio=TEST_RATIO):
    ratings = ratings.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    sizes = ratings.groupby("user_id").size()
    cuts = (sizes * (1 - test_ratio)).astype(int)
    ranks = ratings.groupby("user_id").cumcount()
    user_cut = ratings["user_id"].map(cuts)
    is_train = ranks < user_cut
    return ratings[is_train].copy(), ratings[~is_train].copy()


def item_popularity(train):
    counts = train.groupby("movie_id").size()
    counts.name = "pop_count"
    return counts


def popularity_quantile(pop):
    """Global rank-based popularity score in [0, 1] per movie_id."""
    ranks = pop.rank(method="average", pct=True)
    return ranks


def build_tips_table(train, window_days=TIPS_WINDOW_DAYS):
    """Time-bucketed propensity p(i, t) = share of ratings in bucket for item i.

    Returns (prop_table indexed by (movie_id, bucket), window_secs, bucket_max).
    """
    window_secs = int(window_days * 86400)
    buckets = (train["timestamp"] // window_secs).astype(int)
    tmp = train.assign(_bucket=buckets)
    joint = tmp.groupby(["movie_id", "_bucket"]).size().rename("n_item")
    per_bucket = tmp.groupby("_bucket").size().rename("n_bucket")
    prop = (joint / per_bucket).rename("propensity").reset_index()
    prop_idx = prop.set_index(["movie_id", "_bucket"])["propensity"]
    bucket_max = prop.groupby("_bucket")["propensity"].max().to_dict()
    return prop_idx, window_secs, bucket_max


def tips_popularity_score(movie_id, timestamp, prop_table, window_secs,
                          bucket_max, pop_quantile_series):
    """Time-aware popularity score in [0, 1].

    Within the time bucket of `timestamp`, the target item's propensity is
    rescaled by the max propensity in that bucket so scores are comparable
    across sparsely- and densely-rated time periods. If the item was not
    rated in the bucket, fall back to the global popularity quantile.
    """
    bucket = int(int(timestamp) // window_secs)
    mid = int(movie_id)
    try:
        p = float(prop_table.loc[(mid, bucket)])
        bmax = bucket_max.get(bucket, p)
        if bmax <= 0:
            raise KeyError
        return max(0.0, min(1.0, p / bmax))
    except KeyError:
        q = pop_quantile_series.get(mid, 0.0)
        return float(max(0.0, min(1.0, q)))


def user_history(train, user_id, max_items=HISTORY_MAX, strategy="recent_and_top"):
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


def stratified_subsample(test_df, pop, n, n_buckets, rng=None):
    rng = rng or np.random.default_rng()
    pop_vals = pop.reindex(test_df["movie_id"]).fillna(0).values
    log_pop = np.log1p(pop_vals)
    try:
        buckets = pd.qcut(log_pop, q=n_buckets, labels=False, duplicates="drop")
    except ValueError:
        buckets = np.zeros(len(log_pop), dtype=int)
    buckets = pd.Series(buckets, index=test_df.index).fillna(0).astype(int)
    per_bucket = max(1, n // max(1, buckets.nunique()))
    picks = []
    for b in sorted(buckets.unique()):
        pool = test_df.index[buckets == b].tolist()
        take = min(per_bucket, len(pool))
        if take:
            picks.extend(rng.choice(pool, size=take, replace=False).tolist())
    return test_df.loc[picks].copy()


def format_user_profile(user_row):
    return (
        f"Gender: {user_row['gender_name']}; "
        f"Age group: {user_row['age_group']}; "
        f"Occupation: {user_row['occupation_name']}"
    )


def user_stats(train, user_id):
    """Return the user's rating stats computed from the training split.

    Stats are passed into the LLM prompt so the model can calibrate to the
    user's actual rating scale instead of regressing toward "4" by default.
    """
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


def format_user_stats(stats):
    pct = stats["pct"]
    return (
        f"mean={stats['mean']:.2f}, std={stats['std']:.2f}, n_ratings={stats['count']}, "
        f"distribution: 1s:{pct[1]}%, 2s:{pct[2]}%, 3s:{pct[3]}%, 4s:{pct[4]}%, 5s:{pct[5]}%"
    )


def format_movie(movie_row):
    genres = ", ".join(movie_row["genres_list"]) if isinstance(movie_row["genres_list"], list) else ""
    year = int(movie_row["year"]) if pd.notna(movie_row["year"]) else "?"
    return f"\"{movie_row['title_clean']}\" ({year}) [{genres}]"


def format_history(history_df, movies_lookup):
    """movies_lookup: DataFrame indexed by movie_id, or dict[int -> row-dict]."""
    is_dict = isinstance(movies_lookup, dict)
    lines = []
    for _, r in history_df.iterrows():
        mid = int(r.movie_id)
        m = movies_lookup[mid] if is_dict else movies_lookup.loc[mid]
        lines.append(f"  - {format_movie(m)} — user rated {int(r.rating)}/5")
    return "\n".join(lines) if lines else "  (no history)"
