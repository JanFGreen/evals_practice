"""Extra module D2: conversation metrics. Tests: uv run pytest tests/test_d2_multiturn.py"""


def conversation_metrics(transcript: list[dict], success_keywords: list[str]) -> dict:
    """{"turns", "goal_reached", "turns_to_goal", "refusals"}.

    The goal is reached at turn i when all keywords (after normalisation) have appeared in the assistant's
    answers from turns 1..i. For refusals use evalkit.checks.is_refusal.
    """
    raise NotImplementedError
