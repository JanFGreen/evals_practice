"""Statistics for evals (extra module D1): intervals, paired comparisons, sample sizes. Stdlib only."""

import math
import random
from statistics import NormalDist, fmean
from typing import Callable, Sequence


def _z(confidence: float) -> float:
    return NormalDist().inv_cdf(1 - (1 - confidence) / 2)


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z = _z(confidence)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[Sequence[float]], float] = fmean,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(values)
    boots = sorted(stat([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    lo = boots[int((1 - confidence) / 2 * n_boot)]
    hi = boots[min(n_boot - 1, int((1 + confidence) / 2 * n_boot))]
    return (stat(values), lo, hi)


def paired_bootstrap_diff(
    a: Sequence[float], b: Sequence[float], n_boot: int = 2000, confidence: float = 0.95, seed: int = 0
) -> tuple[float, float, float]:
    """mean(b) - mean(a) with a CI from resampling *cases* (pairs stay together)."""
    if len(a) != len(b) or not a:
        raise ValueError("a and b must be non-empty lists of the same length (the same cases)")
    diffs = [y - x for x, y in zip(a, b)]
    return bootstrap_ci(diffs, fmean, n_boot, confidence, seed)


def mcnemar_exact(regressions: int, fixes: int) -> float:
    """Two-sided exact McNemar p-value. Only discordant pairs matter: pass->fail and fail->pass."""
    n = regressions + fixes
    if n == 0:
        return 1.0
    k = min(regressions, fixes)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def noise_floor(n_cases: int, reps: int = 1) -> float:
    """Rule of thumb: half-width of a 95% CI on a pass-rate difference is about 1/sqrt(n*reps)."""
    return 1 / math.sqrt(n_cases * reps)


def sample_size_for_halfwidth(halfwidth: float, p: float = 0.5, confidence: float = 0.95) -> int:
    z = _z(confidence)
    return math.ceil(z * z * p * (1 - p) / (halfwidth * halfwidth))


def per_case_means(rows: list[dict], key: str = "passed") -> dict[str, float]:
    sums: dict[str, list[float]] = {}
    for r in rows:
        if r.get(key) is None:
            continue
        sums.setdefault(r["case_id"], []).append(float(r[key]))
    return {cid: fmean(v) for cid, v in sums.items()}
