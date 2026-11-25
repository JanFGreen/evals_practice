"""Module 8: production monitoring. Tests: uv run pytest tests/test_m08_monitor.py"""


def psi(expected: dict[str, int], actual: dict[str, int], eps: float = 1e-4) -> float:
    """Population Stability Index over categories; replace shares < eps with eps (new categories!)."""
    raise NotImplementedError


def sample_for_review(traces: list[dict], budget: int, seed: int = 0, low_score: float = 1.5) -> list[dict]:
    """Up to budget // 2 flagged traces (first feedback == "down", then status != "ok", then the lowest
    retrieval_top_score < low_score), the rest of the budget at random (random.Random(seed)) from the remaining ones.
    Every selected trace gets a review_reason field: thumbs_down | error | low_retrieval | random."""
    raise NotImplementedError
