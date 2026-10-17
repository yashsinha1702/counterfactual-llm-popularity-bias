from data import format_user_profile, format_movie, format_history, tips_popularity_score
from llm import chat_json
from config import TIPS_ALPHA

SYSTEM = (
    "You are a causal-inference analyst for a recommender system. Your job is to "
    "score candidate explanations of a user-movie match on two axes: (1) how "
    "plausible the explanation is given the user's history and the movie's "
    "features, and (2) how much the explanation depends on the movie's general "
    "popularity/mainstream-appeal rather than its intrinsic fit to this user. "
    "A causal-quality explanation has HIGH plausibility and LOW popularity dependence."
)


def build_prompt(user_row, history_df, target_movie, movies_df, hypotheses):
    profile = format_user_profile(user_row)
    hist = format_history(history_df, movies_df)
    target = format_movie(target_movie)
    listing = "\n".join(
        f"[{h['id']}] ({h['direction']}) {h['text']}" for h in hypotheses
    )
    user_msg = f"""User profile: {profile}

User's rating history:
{hist}

Target movie: {target}

Candidate explanations to score:
{listing}

For EACH candidate (by id), score:
- plausibility: 1-10, where 10 = strongly supported by concrete evidence in the \
user's history and the target's specific features.
- popularity_reliance: 0-10, where 10 = the explanation mainly leans on the movie \
being a famous/mainstream/classic/blockbuster (or any reasoning that would apply \
to any popular movie regardless of this user), and 0 = the explanation is entirely \
about intrinsic semantic fit to THIS user.

Return JSON:
{{"scores": [
  {{"id": 1, "plausibility": <int 1-10>, "popularity_reliance": <int 0-10>, "note": "short justification"}},
  ...
]}}"""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_msg},
    ]


def score_hypotheses(user_row, history_df, target_movie, movies_df, hypotheses):
    msgs = build_prompt(user_row, history_df, target_movie, movies_df, hypotheses)
    obj = chat_json(msgs, extra_tag=("intervention", len(hypotheses)))
    by_id = {}
    for s in obj.get("scores", []):
        try:
            by_id[int(s["id"])] = {
                "plausibility": float(s.get("plausibility", 5)),
                "popularity_reliance": float(s.get("popularity_reliance", 5)),
                "note": str(s.get("note", "")),
            }
        except (KeyError, TypeError, ValueError):
            continue
    merged = []
    for h in hypotheses:
        s = by_id.get(h["id"], {"plausibility": 5.0, "popularity_reliance": 5.0, "note": ""})
        merged.append({**h, **s})
    return merged


def select_debiased(scored_hypotheses, target_movie_id, timestamp,
                    prop_table, window_secs, bucket_max, pop_quantile_series,
                    alpha=TIPS_ALPHA):
    """Counterfactual selection: penalize popularity-reliance in proportion
    to the target item's time-aware popularity. Very popular items → strong
    penalty on explanations that lean on mainstream appeal; niche items →
    explanations are judged mostly on plausibility alone.
    """
    pop_score = tips_popularity_score(
        target_movie_id, timestamp, prop_table, window_secs,
        bucket_max, pop_quantile_series,
    )

    scored = []
    for h in scored_hypotheses:
        plaus = h["plausibility"] / 10.0
        pop_rel = h["popularity_reliance"] / 10.0
        penalty = 1.0 - pop_rel * (alpha + (1.0 - alpha) * pop_score)
        penalty = max(penalty, 0.05)
        score = plaus * penalty
        scored.append({**h, "tips_pop_score": pop_score, "causal_score": score})

    scored.sort(key=lambda x: x["causal_score"], reverse=True)
    return scored[0], scored
