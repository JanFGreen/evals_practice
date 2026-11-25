"""Agent metrics (module 6): tool-call matching, pass@k and pass^k."""

import math

from evalkit.checks import normalize


def _args_match(expected: dict, actual: dict) -> bool:
    return all(normalize(str(v)) in normalize(str(actual.get(k, ""))) for k, v in expected.items())


def tool_calls_match(expected: list[dict], actual: list[dict]) -> dict:
    """expected: [{"tool": name, "args": {subset}}]; order-insensitive, each actual call used once."""
    used: set[int] = set()
    missing = []
    for exp in expected:
        idx = next(
            (i for i, act in enumerate(actual)
             if i not in used and act["tool"] == exp["tool"] and _args_match(exp.get("args", {}), act.get("args", {}))),
            None,
        )
        if idx is None:
            missing.append(exp)
        else:
            used.add(idx)
    matched = len(used)
    return {
        "recall": matched / len(expected) if expected else 1.0,
        "precision": matched / len(actual) if actual else 1.0,
        "missing": missing,
        "unexpected": len(actual) - matched,
    }


def pass_at_k(n: int, c: int, k: int) -> float:
    """P(at least one of k samples succeeds), unbiased estimator from n trials with c successes."""
    if n - c < k:
        return 1.0
    return 1 - math.comb(n - c, k) / math.comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """pass^k: P(all k samples succeed). The reliability number users actually feel."""
    if c < k:
        return 0.0
    return math.comb(c, k) / math.comb(n, k)
