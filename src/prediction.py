import re

from data import format_user_profile, format_movie
from llm import chat_json

SYSTEM = (
    "You predict a user's explicit rating of a movie on a strict integer scale "
    "from 1 to 5. You are given a single causal explanation of how the user will "
    "evaluate this movie. You must treat that explanation as the sole causal "
    "antecedent of the rating and output a rating consistent with it. Do not "
    "re-derive reasons; use the provided explanation."
)


def build_prompt(user_row, target_movie, explanation_text):
    profile = format_user_profile(user_row)
    target = format_movie(target_movie)
    user_msg = f"""User profile: {profile}

Target movie: {target}

Causal explanation of how this user will evaluate this movie:
\"\"\"{explanation_text}\"\"\"

Task: Output the integer rating (1, 2, 3, 4, or 5) that this user will give the \
target movie, consistent with the explanation above.

Rating anchors:
  1 = strongly dislike
  2 = dislike
  3 = neutral / mixed
  4 = like
  5 = strongly like

Return JSON:
{{"rating": <integer 1-5>, "brief": "one-sentence rationale tied to the explanation"}}"""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_msg},
    ]


def _parse_rating(obj):
    r = obj.get("rating")
    if isinstance(r, (int, float)):
        r = float(r)
    elif isinstance(r, str):
        m = re.search(r"[1-5](?:\.\d+)?", r)
        r = float(m.group(0)) if m else 3.0
    else:
        r = 3.0
    return max(1.0, min(5.0, r))


def predict_rating(user_row, target_movie, explanation_text):
    msgs = build_prompt(user_row, target_movie, explanation_text)
    obj = chat_json(msgs, extra_tag=("prediction",))
    return _parse_rating(obj), str(obj.get("brief", ""))
