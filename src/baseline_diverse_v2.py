"""Diverse baseline + user-rating anchoring. Same pros/cons prompt as
baseline_diverse, but augmented with the user's rating statistics from
training history so the model can calibrate to the user's actual scale
rather than regressing toward 4.
"""
from data import format_user_profile, format_movie, format_history, format_user_stats
from llm import chat_json
from prediction import _parse_rating

SYSTEM = (
    "You are a movie rating predictor. Before you predict, enumerate plausible "
    "reasons this specific user might rate the target movie HIGH and reasons "
    "they might rate it LOW, grounded in their rating history. You are given "
    "the user's rating statistics so you can calibrate to their actual rating "
    "scale. Weigh both sides, then commit to an integer 1-5 rating."
)


def build_prompt(user_row, history_df, target_movie, movies_df, stats):
    profile = format_user_profile(user_row)
    stats_str = format_user_stats(stats)
    hist = format_history(history_df, movies_df)
    target = format_movie(target_movie)
    user_msg = f"""User profile: {profile}
User's rating statistics (training history): {stats_str}

User's rating history (5 is best, 1 is worst):
{hist}

Target movie: {target}

Calibration guidance: If this user's mean is below 3.5, they are a harsh rater — \
a mediocre match should still be rated ≤ 3. If the mean is above 4, they are a \
generous rater — only a clear mismatch justifies ≤ 3. The committed rating should \
be consistent with the user's rating distribution.

Think before you answer:
1. List 2-3 concrete reasons this user might rate the target HIGH.
2. List 2-3 concrete reasons this user might rate the target LOW.
3. Calibrate against the user's historical distribution.
4. Commit to an integer rating.

Return JSON:
{{"reasons_high": ["..."], "reasons_low": ["..."], "rating": <int 1-5>, "brief": "final rationale"}}"""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_msg},
    ]


def predict_rating_diverse_v2(user_row, history_df, target_movie, movies_df, stats):
    msgs = build_prompt(user_row, history_df, target_movie, movies_df, stats)
    obj = chat_json(msgs, extra_tag=("diverse_v2",))
    return _parse_rating(obj), str(obj.get("brief", ""))
