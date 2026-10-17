"""Generate rich text descriptions for all ML-1M movies via LLM.

For each movie: one prompt → structured description covering plot, themes,
mood, target audience. Outputs outputs/item_descriptions.json keyed by
movie_id, and a plain-text field used by embed_llm_items.py to produce
a semantic embedding matrix for downstream methods (LightGCN init,
NeuMF init, two-tower encoder).
"""
import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

from config import OUTPUT_DIR
from data import load_movies, format_movie
from llm import chat_json, batch_map


SYSTEM = (
    "You are a film reference writer. For each movie, produce a compact, "
    "factually accurate description focused on content features that matter "
    "for recommendation: genre/subgenre, themes, mood, pacing, audience, era, "
    "and distinguishing style. Keep it to 60-100 words. Do not reference the "
    "movie's popularity, awards, or box office. Neutral tone."
)


def build_prompt(movie_row):
    target = format_movie(movie_row)
    user_msg = f"""Movie: {target}

Return JSON with a concise 60-100 word description:
{{"description": "..."}}"""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_msg},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out", type=str,
                    default=str(OUTPUT_DIR / "item_descriptions.json"))
    args = ap.parse_args()

    t0 = time.time()
    movies = load_movies()
    print(f"Generating descriptions for {len(movies)} movies...")

    rows = [row.to_dict() for _, row in movies.iterrows()]

    def work(row):
        msgs = build_prompt(row)
        try:
            obj = chat_json(msgs, extra_tag=("item_description", "v1"))
            return {"movie_id": int(row["movie_id"]),
                    "description": str(obj.get("description", "")).strip()}
        except Exception as e:
            return {"movie_id": int(row["movie_id"]), "description": "",
                    "err": str(e)[:140]}

    with tqdm(total=len(rows), desc="items") as pbar:
        outs = batch_map(work, rows, concurrency=args.concurrency, progress=pbar)

    out_map = {o["movie_id"]: o["description"] for o in outs}
    Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    with open(args.out, "w") as f:
        json.dump(out_map, f, indent=1)
    print(f"\nSaved {len(out_map)} descriptions to {args.out}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
