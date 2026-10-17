"""Causal-delta prediction: stage-3 predicts a rating by adjusting a
classical-CF reference (ItemKNN) based on the debiased causal explanation
chosen in the intervention stage.

This keeps the counterfactual-reasoning framework intact — abduction and
intervention still produce a debiased, popularity-corrected explanation —
but grounds the numerical output in the CF signal the LLM cannot access
through text. The LLM's role becomes: *given this debiased rationale,
should the CF reference be adjusted up, down, or accepted?*
"""
import re

from data import format_user_profile, format_movie
from llm import chat_json

SYSTEM = (
    "You calibrate a movie-rating prediction. You are given (a) a strong "
    "collaborative-filtering reference rating produced by a method that "
    "correlates the user's history with similar users and items, (b) a "
    "debiased causal explanation of how this specific user will evaluate "
    "this specific movie (with popularity-driven reasons already filtered "
    "out). Decide whether the CF reference is appropriately calibrated. "
    "Adjust it UP if the causal explanation suggests the user will like "
    "the movie more than typical; DOWN if less. Do not swing wildly. "
    "Output the final rating."
)


def build_prompt(user_row, target_movie, explanation_text, cf_reference,
                 user_mean_hint=None):
    profile = format_user_profile(user_row)
    target = format_movie(target_movie)
    hint = f"User's historical mean rating: {user_mean_hint:.2f}\n" if user_mean_hint is not None else ""
    user_msg = f"""User profile: {profile}
{hint}Target movie: {target}

Collaborative-filtering reference rating (1-5, real-valued): {cf_reference:.2f}

Debiased causal explanation of how this user will evaluate the movie:
\"\"\"{explanation_text}\"\"\"

Task:
1. Judge whether the causal explanation implies the user will rate the movie \
higher than the CF reference, lower, or about the same.
2. Choose a delta from {{-1.0, -0.5, 0.0, +0.5, +1.0}} to apply to the reference.
3. Output the final rating = clip(reference + delta, 1, 5) as a real number.

Return JSON:
{{"implied_direction": "higher|lower|same",
  "delta": <one of -1.0, -0.5, 0.0, +0.5, +1.0>,
  "final_rating": <real number 1-5>,
  "brief": "one-sentence rationale"}}"""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_msg},
    ]


def _parse_float(x, default=3.0):
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        m = re.search(r"-?\d+(?:\.\d+)?", x)
        return float(m.group(0)) if m else default
    return default


def predict_rating_delta(user_row, target_movie, explanation_text,
                         cf_reference, user_mean_hint=None, ensemble_id=0):
    """Return (final_rating, delta, direction, brief). ensemble_id varies
    the cache key so repeated calls at the same (user, item, explanation)
    produce fresh temperature-0.3 samples that can be averaged for variance
    reduction."""
    msgs = build_prompt(user_row, target_movie, explanation_text,
                        cf_reference, user_mean_hint=user_mean_hint)
    obj = chat_json(msgs, extra_tag=("prediction_delta", ensemble_id))
    final = _parse_float(obj.get("final_rating"), default=cf_reference)
    delta = _parse_float(obj.get("delta"), default=0.0)
    direction = str(obj.get("direction", obj.get("implied_direction", "same"))).lower()
    brief = str(obj.get("brief", ""))
    grid = (-1.0, -0.5, 0.0, 0.5, 1.0)
    delta = min(grid, key=lambda g: abs(g - delta))
    final_clean = max(1.0, min(5.0, cf_reference + delta))
    return final_clean, delta, direction, brief
