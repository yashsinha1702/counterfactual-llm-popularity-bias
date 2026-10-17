"""Retrieval-augmented history: for a test (user, target) pair, select the
K items the user has rated that are MOST content-similar to the target,
using the embeddings produced by embed_movies.py. Falls back to
top-rated-recent history if target not in index or user has no rated
items in the index."""
from pathlib import Path
import numpy as np

from config import OUTPUT_DIR

_EMBEDDINGS = None
_ID_TO_IDX = None


def _load():
    global _EMBEDDINGS, _ID_TO_IDX
    if _EMBEDDINGS is None:
        emb_dir = Path(OUTPUT_DIR) / "embeddings"
        _EMBEDDINGS = np.load(emb_dir / "movies.npy")
        ids = np.load(emb_dir / "movie_ids.npy")
        _ID_TO_IDX = {int(m): int(i) for i, m in enumerate(ids)}
    return _EMBEDDINGS, _ID_TO_IDX


def user_history_retrieved(train, user_id, target_movie_id, max_items=15):
    """Return top-K user-rated items most content-similar to target.

    Mixes: half by similarity, the rest by (rating, timestamp) to still
    surface strong preference signals from the user's history.
    """
    emb, id_to_idx = _load()
    u = train[train.user_id == user_id]
    if u.empty:
        return u

    target_idx = id_to_idx.get(int(target_movie_id))

    u_mids = u["movie_id"].astype(int).values
    valid_mask = np.array([m in id_to_idx for m in u_mids])
    if not valid_mask.any() or target_idx is None:
        # Fallback: half top, half recent
        top = u.sort_values(["rating", "timestamp"], ascending=[False, False]).head(max_items // 2)
        recent = u.sort_values("timestamp", ascending=False).head(max_items - len(top))
        return _dedupe(top, recent, max_items)

    u_valid = u[valid_mask]
    u_mids_valid = u_valid["movie_id"].astype(int).values
    u_emb_idxs = np.array([id_to_idx[m] for m in u_mids_valid], dtype=np.int64)

    target_vec = emb[target_idx]
    sims = emb[u_emb_idxs] @ target_vec  # cosine (embeddings are L2 normalized)

    # Take half by similarity, quarter by highest user rating, quarter by recency
    n_sim = max_items // 2
    sim_order = np.argsort(-sims)[:n_sim]
    sim_picks = u_valid.iloc[sim_order]

    n_top = (max_items - n_sim) // 2
    top_picks = u_valid.sort_values(["rating", "timestamp"], ascending=[False, False]).head(n_top)

    n_recent = max_items - n_sim - n_top
    recent_picks = u_valid.sort_values("timestamp", ascending=False).head(n_recent)

    return _dedupe(sim_picks, top_picks, recent_picks, max_items=max_items)


def retrieve_similar_user_items(user_id, target_movie_id, user_history_df, k=10):
    """Return up to k items from a pre-filtered user-history DataFrame most
    content-similar to target_movie_id. Attaches a `sim` column with cosine
    similarity to the target. Returns an empty DataFrame on failure."""
    import pandas as pd
    if user_history_df is None or len(user_history_df) == 0:
        return user_history_df
    emb, id_to_idx = _load()
    target_idx = id_to_idx.get(int(target_movie_id))
    if target_idx is None:
        return user_history_df.sort_values("timestamp", ascending=False).head(k).assign(sim=np.nan)
    mids = user_history_df["movie_id"].astype(int).values
    valid_mask = np.array([m in id_to_idx for m in mids])
    if not valid_mask.any():
        return user_history_df.sort_values("timestamp", ascending=False).head(k).assign(sim=np.nan)
    valid_df = user_history_df[valid_mask].copy()
    emb_idxs = np.array([id_to_idx[int(m)] for m in valid_df["movie_id"].values], dtype=np.int64)
    target_vec = emb[target_idx]
    sims = emb[emb_idxs] @ target_vec  # cosine (L2 normalized)
    valid_df = valid_df.assign(sim=sims)
    return valid_df.sort_values("sim", ascending=False).head(k)


def _dedupe(*frames, max_items=15):
    import pandas as pd
    combined = pd.concat(frames).drop_duplicates(subset="movie_id")
    return combined.head(max_items)
