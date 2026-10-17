"""Logprob-based continuous rating prediction.

Instead of parsing a discrete integer from the LLM's text output, ask for
exactly one token and read the first-token top_logprobs. Normalize the
probabilities of the digit tokens {'1', '2', '3', '4', '5'} and compute
the expected rating E[r] = sum_k k * p(k | history, target).

This yields a real-valued prediction that is calibrated to the model's
confidence across the full rating scale, which reduces RMSE versus
forced-integer rounding.
"""
import numpy as np

from data import format_user_profile, format_movie, format_history, format_user_stats
from llm import chat_logprobs_first_token


DIGIT_TOKENS = ("1", "2", "3", "4", "5")


SYSTEM_BASE = (
    "You are a movie rating predictor. Emit a single digit 1-5 as your answer "
    "and stop. Do not add quotes, punctuation, or explanation."
)


def _build_prompt(user_row, history_df, target_movie, movies_lookup, stats=None):
    profile = format_user_profile(user_row)
    hist = format_history(history_df, movies_lookup)
    target = format_movie(target_movie)
    calib = (f"\nUser's rating statistics (training): {format_user_stats(stats)}"
             if stats is not None else "")
    user_msg = f"""User profile: {profile}{calib}

User's rating history:
{hist}

Target movie: {target}

Predict this user's rating for the target movie on the 1-5 integer scale. \
Output only the single digit.

Rating:"""
    return [
        {"role": "system", "content": SYSTEM_BASE},
        {"role": "user", "content": user_msg},
    ]


def _expected_rating_from_tokens(tok_probs):
    """Given {token_str: prob} from top_logprobs, keep only digit tokens
    '1'..'5' (and variants with leading whitespace), renormalize, and
    compute E[rating]. Falls back to argmax-as-int on failure."""
    digit_p = {k: 0.0 for k in DIGIT_TOKENS}
    for tok, p in tok_probs.items():
        s = tok.strip()
        if s in digit_p:
            digit_p[s] += float(p)
    total = sum(digit_p.values())
    if total <= 0:
        return None
    expected = 0.0
    for k_str, p in digit_p.items():
        expected += int(k_str) * (p / total)
    return float(np.clip(expected, 1.0, 5.0))


def predict_rating_logprob(user_row, history_df, target_movie, movies_lookup,
                           stats=None):
    msgs = _build_prompt(user_row, history_df, target_movie, movies_lookup, stats)
    resp = chat_logprobs_first_token(
        msgs, top_logprobs=10, max_tokens=2,
        extra_tag=("logprob_predict", "with_stats" if stats is not None else "no_stats"),
    )
    exp = _expected_rating_from_tokens(resp["tokens"])
    if exp is None:
        # fallback: try to parse text
        import re
        m = re.search(r"[1-5]", resp["text"] or "")
        return float(m.group(0)) if m else 3.0, resp
    return exp, resp
