"""Extra module D1: eval statistics. Tests: uv run pytest tests/test_d1_stats.py"""

from typing import Sequence


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson interval for a proportion; for n == 0 return (0.0, 1.0). Quantile: statistics.NormalDist().inv_cdf."""
    raise NotImplementedError


def paired_bootstrap_diff(
    a: Sequence[float], b: Sequence[float], n_boot: int = 2000, confidence: float = 0.95, seed: int = 0
) -> tuple[float, float, float]:
    """(mean(b) - mean(a), lower, upper CI bound) from resampling cases with replacement; pairs stay together."""
    raise NotImplementedError


def mcnemar_exact(regressions: int, fixes: int) -> float:
    """Two-sided p from an exact binomial test on discordant pairs (p=0.5); for 0 pairs return 1.0."""
    raise NotImplementedError
