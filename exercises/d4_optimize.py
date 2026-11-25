"""Extra module D4: optimisation without overfitting. Tests: uv run pytest tests/test_d4_optimize.py"""

from typing import Callable


def stratified_split(items: list, key: Callable, test_frac: float = 0.5, seed: int = 0) -> tuple[list, list]:
    """(train, test): in every key(item) group, shuffle (random.Random(seed)) and put round(len * test_frac) into test."""
    raise NotImplementedError


def find_leaked_cases(text: str, cases: list, threshold: float = 0.8) -> list[str]:
    """Ids of cases whose question appears in some line of the text (word subset or Jaccard >= threshold)."""
    raise NotImplementedError
