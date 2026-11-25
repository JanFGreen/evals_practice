"""Module 6: agent metrics. Tests: uv run pytest tests/test_m06_agent.py"""


def tool_calls_match(expected: list[dict], actual: list[dict]) -> dict:
    """{"recall", "precision", "missing", "unexpected"}. Order does not matter, each call is used once.

    Expected args are a subset: a value matches when, after normalisation, it is contained in the actual value
    (e.g. "B" matches "hall B").
    """
    raise NotImplementedError


def pass_at_k(n: int, c: int, k: int) -> float:
    """Probability that at least one of k attempts succeeds (n attempts, c successes)."""
    raise NotImplementedError


def pass_hat_k(n: int, c: int, k: int) -> float:
    """pass^k: probability that all k attempts succeed."""
    raise NotImplementedError
