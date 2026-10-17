from data import format_user_profile, format_movie, format_history
from llm import chat_json
from config import K_EXPLANATIONS

SYSTEM = (
    "You are a careful recommender system analyst. Given a user's profile and "
    "rating history, and a target movie, you generate diverse hypotheses about "
    "WHY this specific user might rate the target movie a particular way. "
    "Hypotheses must be grounded in concrete, distinguishing features of the "
    "target movie and concrete signals from the user's history. Produce a mix "
    "of reasons — some favorable, some unfavorable — so a downstream step can "
    "pick the most plausible one. Do not include a numeric rating in the hypotheses."
)


def build_prompt(user_row, history_df, target_movie, movies_df, k=K_EXPLANATIONS):
    profile = format_user_profile(user_row)
    hist = format_history(history_df, movies_df)
    target = format_movie(target_movie)
    user_msg = f"""User profile: {profile}

User's rating history (5 is best, 1 is worst):
{hist}

Target movie: {target}

Generate exactly {k} distinct candidate explanations (hypotheses) of how this user \
will evaluate the target movie. Each hypothesis is a short paragraph (2-4 sentences) \
that cites specific features of the target and specific signals from the user's \
history. Include at least one hypothesis that would imply a LOW rating and at least \
one that would imply a HIGH rating. Do NOT state a numerical rating in the hypothesis.

Return JSON with this exact shape:
{{"hypotheses": [
  {{"id": 1, "direction": "positive|negative|neutral", "text": "..."}},
  ... (total {k})
]}}"""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_msg},
    ]


def generate_candidates(user_row, history_df, target_movie, movies_df, k=K_EXPLANATIONS):
    msgs = build_prompt(user_row, history_df, target_movie, movies_df, k=k)
    obj = chat_json(msgs, extra_tag=("abduction", k))
    hyps = obj.get("hypotheses", [])
    cleaned = []
    for i, h in enumerate(hyps[:k]):
        cleaned.append({
            "id": int(h.get("id", i + 1)),
            "direction": str(h.get("direction", "neutral")).lower(),
            "text": str(h.get("text", "")).strip(),
        })
    while len(cleaned) < k:
        cleaned.append({"id": len(cleaned) + 1, "direction": "neutral", "text": ""})
    return cleaned
