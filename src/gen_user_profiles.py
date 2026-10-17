"""Generate structured user taste profiles via LLM.

For each user: one prompt → JSON with preferred genres, disliked genres,
era preference, pacing preference, taste complexity. Used to produce a
user-side semantic embedding for the two-tower model, and as optional
features for the final stacker.
"""
import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

from config import OUTPUT_DIR
from data import (
    load_movies, load_users, load_ratings, temporal_split,
    format_user_profile, format_history, user_history, user_stats,
    format_user_stats,
)
from llm import chat_json, batch_map


SYSTEM = (
    "You are a movie-preference analyst. Given a user's demographic profile "
    "and their rating history, produce a structured JSON profile capturing "
    "their taste. Be specific and factually grounded in the history."
)


def build_prompt(user_row, history_df, movies_lookup, stats):
    profile = format_user_profile(user_row)
    hist = format_history(history_df, movies_lookup)
    stats_str = format_user_stats(stats)
    user_msg = f"""User profile: {profile}
Rating statistics: {stats_str}

Rating history (5 is best):
{hist}

Return JSON capturing this user's taste:
{{
  "loved_genres": ["..."],
  "disliked_genres": ["..."],
  "era_preference": "...",  // e.g., "80s-90s films"
  "pacing_preference": "fast|moderate|slow|mixed",
  "tone_preference": "light|serious|mixed|dark",
  "taste_summary": "one-paragraph 40-60 word summary of this user's taste"
}}"""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_msg},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out", type=str,
                    default=str(OUTPUT_DIR / "user_profiles.json"))
    args = ap.parse_args()

    t0 = time.time()
    movies = load_movies()
    users = load_users()
    ratings = load_ratings()
    train, _ = temporal_split(ratings)
    print(f"Generating profiles for {users.shape[0]} users...")

    movies_lookup = {int(mid): movies.loc[mid].to_dict() for mid in movies.index}

    uids = sorted(train["user_id"].unique())
    hist_cache = {int(u): user_history(train, int(u)) for u in uids}
    stats_cache = {int(u): user_stats(train, int(u)) for u in uids}

    def work(uid):
        u = users.loc[uid].to_dict()
        h = hist_cache.get(int(uid))
        s = stats_cache.get(int(uid))
        if h is None or s is None:
            return {"user_id": int(uid), "profile": {}}
        msgs = build_prompt(u, h, movies_lookup, s)
        try:
            obj = chat_json(msgs, extra_tag=("user_profile", "v1"))
            return {"user_id": int(uid), "profile": obj}
        except Exception as e:
            return {"user_id": int(uid), "profile": {}, "err": str(e)[:140]}

    with tqdm(total=len(uids), desc="users") as pbar:
        outs = batch_map(work, uids, concurrency=args.concurrency, progress=pbar)

    out_map = {o["user_id"]: o.get("profile", {}) for o in outs}
    Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    with open(args.out, "w") as f:
        json.dump(out_map, f, indent=1)
    print(f"\nSaved {len(out_map)} profiles to {args.out}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
