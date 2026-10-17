"""A stronger baseline: single-call zero-shot prediction that asks the LLM to
list plausible reasons the user might rate the target movie high AND low
before committing to an integer rating. This tests whether the counterfactual
framework's improvement stems from the causal decomposition (abduction +
intervention + prediction) itself, or merely from prompting the model to
consider both sides.
"""
from data import format_user_profile, format_movie, format_history
from llm import chat_json
from prediction import _parse_rating

SYSTEM = (
    "You are a movie rating predictor. Before you predict, enumerate plausible "
    "reasons this specific user might rate the target movie HIGH and reasons "
    "they might rate it LOW, grounded in their rating history. Weigh both "
    "sides, then commit to an integer 1-5 rating."
)


def build_prompt(user_row, history_df, target_movie, movies_df):
    profile = format_user_profile(user_row)
    hist = format_history(history_df, movies_df)
    target = format_movie(target_movie)
    user_msg = f"""User profile: {profile}

User's rating history (5 is best, 1 is worst):
{hist}

Target movie: {target}

Think before you answer:
1. List 2-3 concrete reasons this user might rate the target HIGH.
2. List 2-3 concrete reasons this user might rate the target LOW.
3. Weigh the two sides and commit to an integer rating.

Return JSON:
{{"reasons_high": ["..."], "reasons_low": ["..."], "rating": <int 1-5>, "brief": "final rationale"}}"""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_msg},
    ]


def predict_rating_diverse_baseline(user_row, history_df, target_movie, movies_df):
    msgs = build_prompt(user_row, history_df, target_movie, movies_df)
    obj = chat_json(msgs, extra_tag=("diverse_baseline",))
    return _parse_rating(obj), str(obj.get("brief", ""))
