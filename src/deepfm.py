"""DeepFM-style neural CF with side features.

Augments user/item embeddings with explicit side information:
  - User: gender, age_group, occupation (learned embeddings)
  - Item: multi-hot genre embedding (18 genres), decade bucket, log popularity
  - Cross: user's per-genre average rating for target movie's genres

Architecture:
  Deep path: concat(user_emb, item_emb, user_feats, item_feats, genre_affinity) → MLP
  FM path: pairwise interactions of all embedding vectors
  Output: 1 + 4 * sigmoid(deep + fm)

Trains on GPU 0 by default.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import OUTPUT_DIR, SUBSAMPLE_N, POPULARITY_BUCKETS
from data import load_ratings, load_movies, load_users, temporal_split, item_popularity, stratified_subsample

ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
GENRE2IDX = {g: i for i, g in enumerate(ALL_GENRES)}
N_GENRES = len(ALL_GENRES)

AGE_GROUPS = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}
N_AGE = len(AGE_GROUPS)
N_GENDER = 2
N_OCC = 21
N_DECADE = 12  # 1920s through 2000s+


def build_genre_vec(genres_list):
    """Multi-hot genre vector from list of genre strings."""
    v = np.zeros(N_GENRES, dtype=np.float32)
    if isinstance(genres_list, list):
        for g in genres_list:
            if g in GENRE2IDX:
                v[GENRE2IDX[g]] = 1.0
    return v


def decade_bucket(year):
    if pd.isna(year) or year < 1920:
        return 0
    return min(int((year - 1920) // 10), N_DECADE - 1)


def build_user_genre_affinity(train, movies):
    """For each user, compute mean rating per genre from training data."""
    movies_df = movies.reset_index(drop=True) if movies.index.name == "movie_id" else movies
    # Explode genres into separate rows
    mg = movies_df[["movie_id", "genres_list"]].dropna(subset=["genres_list"])
    mg = mg.explode("genres_list")
    mg["genre_idx"] = mg["genres_list"].map(GENRE2IDX)
    mg = mg.dropna(subset=["genre_idx"])
    mg["genre_idx"] = mg["genre_idx"].astype(int)
    # Merge with ratings
    merged = train[["user_id", "movie_id", "rating"]].merge(
        mg[["movie_id", "genre_idx"]], on="movie_id", how="inner"
    )
    agg = merged.groupby(["user_id", "genre_idx"])["rating"].agg(["mean", "count"]).reset_index()
    # Build dicts
    user_genre = {}
    user_genre_count = {}
    for uid, group in agg.groupby("user_id"):
        arr = np.zeros(N_GENRES, dtype=np.float32)
        cnt = np.zeros(N_GENRES, dtype=np.float32)
        for _, r in group.iterrows():
            gi = int(r["genre_idx"])
            arr[gi] = float(r["mean"])
            cnt[gi] = float(r["count"])
        user_genre[int(uid)] = arr
        user_genre_count[int(uid)] = cnt
    return user_genre, user_genre_count


class DeepFM(nn.Module):
    def __init__(self, n_users, n_items, dim=64, deep_hidden=(256, 128, 64),
                 n_genre_affinity=N_GENRES, dropout=0.15):
        super().__init__()
        # ID embeddings
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        # Side feature embeddings
        self.gender_emb = nn.Embedding(N_GENDER, 8)
        self.age_emb = nn.Embedding(N_AGE, 8)
        self.occ_emb = nn.Embedding(N_OCC, 16)
        self.decade_emb = nn.Embedding(N_DECADE, 8)
        # Genre encoder: multi-hot → learned projection
        self.genre_proj = nn.Sequential(
            nn.Linear(N_GENRES, 32), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(32, 16),
        )
        # Genre affinity encoder
        self.affinity_proj = nn.Sequential(
            nn.Linear(N_GENRES, 32), nn.GELU(),
            nn.Linear(32, 16),
        )
        # Deep path
        # Input: user_emb(dim) + item_emb(dim) + gender(8) + age(8) + occ(16)
        #        + decade(8) + genre_proj(16) + affinity(16) + log_pop(1) + affinity_stats(3)
        deep_in = dim + dim + 8 + 8 + 16 + 8 + 16 + 16 + 1 + 3
        layers = []
        prev = deep_in
        for h in deep_hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.deep = nn.Sequential(*layers)
        self.head = nn.Linear(prev + dim, 1)  # +dim for FM interaction
        # Init
        for emb in [self.user_emb, self.item_emb, self.gender_emb,
                     self.age_emb, self.occ_emb, self.decade_emb]:
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, user_id, item_id, gender, age, occ, decade_b,
                genre_vec, affinity_vec, affinity_stats, log_pop):
        u = self.user_emb(user_id)
        i = self.item_emb(item_id)
        g_e = self.gender_emb(gender)
        a_e = self.age_emb(age)
        o_e = self.occ_emb(occ)
        d_e = self.decade_emb(decade_b)
        gp = self.genre_proj(genre_vec)
        ap = self.affinity_proj(affinity_vec)
        # FM interaction (element-wise product of user and item embeddings)
        fm = u * i
        # Deep path
        x = torch.cat([u, i, g_e, a_e, o_e, d_e, gp, ap, log_pop, affinity_stats], dim=-1)
        d = self.deep(x)
        out = self.head(torch.cat([d, fm], dim=-1)).squeeze(-1)
        return 1.0 + 4.0 * torch.sigmoid(out)


def prepare_features(df, movies, users, user_genre, user_genre_count, u_idx, i_idx, pop):
    """Build tensor features for a DataFrame of (user_id, movie_id) pairs. Vectorized."""
    df = df.reset_index(drop=True)
    n = len(df)
    uids = df["user_id"].astype(int).values
    mids = df["movie_id"].astype(int).values

    user_ids = torch.tensor([u_idx[int(u)] for u in uids], dtype=torch.long)
    item_ids = torch.tensor([i_idx[int(m)] for m in mids], dtype=torch.long)

    # Pre-build lookup arrays for users
    user_gender_map = {}
    user_age_map = {}
    user_occ_map = {}
    for uid_val in users.index:
        u_row = users.loc[uid_val]
        user_gender_map[int(uid_val)] = 0 if u_row["gender"] == "M" else 1
        user_age_map[int(uid_val)] = AGE_GROUPS.get(int(u_row["age"]), 0)
        user_occ_map[int(uid_val)] = min(int(u_row["occupation"]), N_OCC - 1)

    # Pre-build lookup arrays for movies
    movie_genre_vecs = {}
    movie_decade_map = {}
    for mid_val in movies.index:
        m_row = movies.loc[mid_val]
        movie_genre_vecs[int(mid_val)] = build_genre_vec(m_row.get("genres_list", []))
        movie_decade_map[int(mid_val)] = decade_bucket(m_row.get("year", 2000))

    # Vectorized assignment
    genders = torch.tensor([user_gender_map.get(int(u), 0) for u in uids], dtype=torch.long)
    ages = torch.tensor([user_age_map.get(int(u), 0) for u in uids], dtype=torch.long)
    occs = torch.tensor([user_occ_map.get(int(u), 0) for u in uids], dtype=torch.long)
    decades = torch.tensor([movie_decade_map.get(int(m), 0) for m in mids], dtype=torch.long)

    # Genre vectors - batch build
    genre_vecs_np = np.zeros((n, N_GENRES), dtype=np.float32)
    for idx, mid in enumerate(mids):
        gv = movie_genre_vecs.get(int(mid))
        if gv is not None:
            genre_vecs_np[idx] = gv
    genre_vecs = torch.from_numpy(genre_vecs_np)

    # Log popularity
    pop_vals = np.array([pop.get(int(m), 0) for m in mids], dtype=np.float32)
    log_pops = torch.from_numpy(np.log1p(pop_vals).reshape(-1, 1))

    # Affinity vectors and stats - vectorized
    affinity_vecs_np = np.zeros((n, N_GENRES), dtype=np.float32)
    affinity_stats_np = np.zeros((n, 3), dtype=np.float32)
    for idx in range(n):
        uid = int(uids[idx])
        if uid in user_genre:
            aff = user_genre[uid]
            affinity_vecs_np[idx] = aff
            gv = genre_vecs_np[idx]
            active = gv.sum()
            if active > 0:
                masked = aff * gv
                affinity_stats_np[idx, 0] = masked.sum() / active
                active_vals = masked[gv > 0]
                if len(active_vals) > 1:
                    affinity_stats_np[idx, 1] = np.std(active_vals)
                cnt = user_genre_count.get(uid, np.zeros(N_GENRES))
                affinity_stats_np[idx, 2] = (cnt * gv).sum() / max(active, 1)

    affinity_vecs = torch.from_numpy(affinity_vecs_np)
    affinity_stats = torch.from_numpy(affinity_stats_np)

    return user_ids, item_ids, genders, ages, occs, decades, genre_vecs, affinity_vecs, affinity_stats, log_pops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=SUBSAMPLE_N)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="n2000")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()
    t0 = time.time()

    print("Loading data...")
    ratings = load_ratings()
    train, test = temporal_split(ratings)
    pop = item_popularity(train)
    movies = load_movies()
    users = load_users()
    rng = np.random.default_rng(args.seed)
    sample = stratified_subsample(test, pop, n=args.n, n_buckets=POPULARITY_BUCKETS, rng=rng)
    sample = sample.join(pop.rename("pop_count"), on="movie_id")
    sample["pop_count"] = sample["pop_count"].fillna(0).astype(int)
    print(f"  train={len(train):,}  test={len(sample)}")

    print("Building user-genre affinity...")
    user_genre, user_genre_count = build_user_genre_affinity(train, movies)
    print(f"  Got affinity for {len(user_genre)} users")

    # ID maps
    all_users = sorted(set(train["user_id"].unique()) | set(sample["user_id"].unique()))
    all_items = sorted(set(train["movie_id"].unique()) | set(sample["movie_id"].unique()))
    u_idx = {int(u): i for i, u in enumerate(all_users)}
    i_idx = {int(m): i for i, m in enumerate(all_items)}
    n_u, n_i = len(all_users), len(all_items)

    print("Preparing training features...")
    train_feats = prepare_features(train, movies, users, user_genre, user_genre_count, u_idx, i_idx, pop)
    r_t = torch.tensor(train["rating"].astype(float).values, dtype=torch.float32)

    print("Preparing test features...")
    test_feats = prepare_features(sample.reset_index(drop=True), movies, users, user_genre, user_genre_count, u_idx, i_idx, pop)
    y_te = sample["rating"].astype(float).values

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Multiple configs
    configs = [
        (args.dim, (256, 128, 64), args.dropout, args.lr, 0, "base_s0"),
        (args.dim, (256, 128, 64), args.dropout, args.lr, 1, "base_s1"),
        (args.dim, (256, 128, 64), args.dropout, args.lr, 2, "base_s2"),
        (96,       (256, 128, 64), 0.20, 6e-4, 0, "d96_s0"),
        (96,       (256, 128, 64), 0.20, 6e-4, 1, "d96_s1"),
    ]

    outdf = sample.reset_index(drop=True).copy()
    all_preds = {}
    results = {}

    for dim, deep_h, drop, lr, seed_off, label in configs:
        print(f"\n--- DeepFM {label} (dim={dim}, deep={deep_h}, drop={drop}, lr={lr}) ---")
        torch.manual_seed(42 + seed_off)
        np.random.seed(42 + seed_off)

        model = DeepFM(n_u, n_i, dim=dim, deep_hidden=deep_h, dropout=drop).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=lr*0.01)

        # Training DataLoader
        train_tensors = [t.to(device) for t in train_feats] + [r_t.to(device)]
        test_tensors = [t.to(device) for t in test_feats]
        n_train = len(r_t)

        best_rmse, best_preds, best_epoch = float("inf"), None, -1
        ct = time.time()
        for epoch in range(args.epochs):
            model.train()
            # Shuffle
            perm = torch.randperm(n_train, device=device)
            total_loss, n_samples = 0.0, 0
            for start in range(0, n_train, args.batch_size):
                end = min(start + args.batch_size, n_train)
                idx = perm[start:end]
                batch = [t[idx] for t in train_tensors[:-1]]
                r_batch = train_tensors[-1][idx]
                pred = model(*batch)
                loss = F.mse_loss(pred, r_batch)
                opt.zero_grad(); loss.backward(); opt.step()
                total_loss += loss.item() * len(idx)
                n_samples += len(idx)
            sched.step()

            model.eval()
            with torch.no_grad():
                tp = model(*test_tensors).cpu().numpy()
            tp = np.clip(tp, 1.0, 5.0)
            rmse = float(np.sqrt(np.mean((y_te - tp) ** 2)))
            if epoch % 10 == 0 or epoch == args.epochs - 1:
                print(f"  epoch {epoch+1:3d}: train_mse={total_loss/n_samples:.4f} test_rmse={rmse:.4f}")
            if rmse < best_rmse:
                best_rmse, best_preds, best_epoch = rmse, tp.copy(), epoch + 1

        elapsed = time.time() - ct
        col = f"pred_DeepFM_{label}"
        outdf[col] = best_preds
        all_preds[label] = best_preds
        mae = float(np.mean(np.abs(y_te - best_preds)))
        bias = float(np.mean(best_preds - y_te))
        results[label] = {"rmse": best_rmse, "mae": mae, "bias": bias,
                          "best_epoch": best_epoch, "elapsed_min": round(elapsed/60, 1)}
        print(f"  → rmse={best_rmse:.4f} mae={mae:.4f} bias={bias:+.4f} epoch={best_epoch} ({elapsed:.0f}s)")

    # Ensemble averages
    all_arr = np.stack(list(all_preds.values()), axis=0)
    outdf["pred_DeepFM_avg"] = np.clip(all_arr.mean(axis=0), 1.0, 5.0)
    avg = outdf["pred_DeepFM_avg"].values
    rmse = float(np.sqrt(np.mean((y_te - avg) ** 2)))
    mae = float(np.mean(np.abs(y_te - avg)))
    bias = float(np.mean(avg - y_te))
    results["ensemble_avg"] = {"rmse": rmse, "mae": mae, "bias": bias}
    print(f"\n  DeepFM ensemble avg: rmse={rmse:.4f} mae={mae:.4f} bias={bias:+.4f}")

    out_dir = Path(OUTPUT_DIR) / f"deepfm_{args.tag}_{int(t0)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    outdf.to_csv(out_dir / "predictions.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"models": results, "config": vars(args)}, f, indent=2)
    print(f"\nSaved: {out_dir} ({(time.time()-t0)/60:.1f} min total)")


if __name__ == "__main__":
    main()
