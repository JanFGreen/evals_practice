"""Production monitoring helpers (module 8): drift and review sampling."""

import math
import random
from collections import Counter, defaultdict
from statistics import fmean, quantiles


def psi(expected: dict[str, int], actual: dict[str, int], eps: float = 1e-4) -> float:
    """Population Stability Index over categories. ~<0.1 stable, 0.1-0.25 watch, >0.25 shifted."""
    cats = set(expected) | set(actual)
    e_total = sum(expected.values()) or 1
    a_total = sum(actual.values()) or 1
    total = 0.0
    for c in cats:
        e = max(expected.get(c, 0) / e_total, eps)
        a = max(actual.get(c, 0) / a_total, eps)
        total += (a - e) * math.log(a / e)
    return total


def _flag_reason(t: dict, low_score: float) -> str | None:
    if t.get("feedback") == "down":
        return "thumbs_down"
    if t.get("status", "ok") != "ok":
        return "error"
    if t.get("retrieval_top_score", math.inf) < low_score:
        return "low_retrieval"
    return None


def sample_for_review(traces: list[dict], budget: int, seed: int = 0, low_score: float = 1.5) -> list[dict]:
    """Half the budget for flagged traces (thumbs-down first, then weakest retrieval), half random."""
    flagged = [(t, r) for t in traces if (r := _flag_reason(t, low_score))]
    order = {"thumbs_down": 0, "error": 1, "low_retrieval": 2}
    flagged.sort(key=lambda tr: (order[tr[1]], tr[0].get("retrieval_top_score", 0)))
    picked = [{**t, "review_reason": r} for t, r in flagged[: budget // 2]]
    picked_ids = {id(t) for t, _ in flagged[: budget // 2]}

    rest = [t for t in traces if id(t) not in picked_ids]
    rng = random.Random(seed)
    rng.shuffle(rest)
    picked += [{**t, "review_reason": "random"} for t in rest[: budget - len(picked)]]
    return picked


def daily_metrics(traces: list[dict]) -> list[dict]:
    by_day: dict[str, list[dict]] = defaultdict(list)
    for t in traces:
        by_day[t["day"]].append(t)
    rows = []
    for day in sorted(by_day):
        ts = by_day[day]
        rated = [t for t in ts if t.get("feedback") in ("up", "down")]
        lat = [t["latency_ms"] for t in ts]
        rows.append({
            "day": day,
            "n": len(ts),
            "refusal_rate": fmean(t["refused"] for t in ts),
            "thumbs_down_rate": fmean(t["feedback"] == "down" for t in rated) if rated else None,
            "mean_top_score": fmean(t["retrieval_top_score"] for t in ts),
            "p95_latency_ms": quantiles(lat, n=20)[-1] if len(lat) >= 2 else lat[0],
            "topics": dict(Counter(t["topic"] for t in ts)),
        })
    return rows
