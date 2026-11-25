"""Module 4: the LLM judge and its calibration. Tests: uv run pytest tests/test_m04_judge.py"""

from typing import Callable


def pairwise_with_swap(judge_fn: Callable[[str, str, str], str], question: str, a: str, b: str) -> dict:
    """Ask judge_fn(question, a, b) and judge_fn(question, b, a); return {"winner": "a"|"b"|"tie", "consistent", "raw"}."""
    raise NotImplementedError


def confusion(pred: list[str], gold: list[str], positive: str = "pass") -> dict[str, int]:
    """{"tp", "fp", "tn", "fn"}; pred = judge, gold = human."""
    raise NotImplementedError


def rates(cm: dict[str, int]) -> dict[str, float | None]:
    """{"tpr", "tnr", "precision", "accuracy"}; None when the denominator is zero."""
    raise NotImplementedError


def cohen_kappa(pred: list[str], gold: list[str]) -> float:
    raise NotImplementedError


def corrected_pass_rate(observed: float, tpr: float, tnr: float) -> float:
    """Rogan-Gladen estimator, clipped to [0, 1]; ValueError when tpr + tnr <= 1."""
    raise NotImplementedError
