from data import format_user_profile, format_movie, format_history
from llm import chat_json
from prediction import _parse_rating

SYSTEM = (
    "You are a movie rating predictor. Given a user's profile, their rating history, "
    "and a target movie, predict the integer rating (1-5) the user will give the "
    "target movie."
)


def build_prompt(user_row, history_df, target_movie, movies_df):
    profile = format_user_profile(user_row)
    hist = format_history(history_df, movies_df)
    target = format_movie(target_movie)
    user_msg = f"""User profile: {profile}

User's rating history (5 is best, 1 is worst):
{hist}

Target movie: {target}

Rating anchors:
  1 = strongly dislike
  2 = dislike
  3 = neutral / mixed
  4 = like
  5 = strongly like

Return JSON:
{{"rating": <integer 1-5>, "reason": "short justification"}}"""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_msg},
    ]


def predict_rating_baseline(user_row, history_df, target_movie, movies_df):
    msgs = build_prompt(user_row, history_df, target_movie, movies_df)
    obj = chat_json(msgs, extra_tag=("baseline",))
    return _parse_rating(obj), str(obj.get("reason", ""))
