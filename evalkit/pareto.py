"""Model / config selection (extra module D7): the quality-cost-latency Pareto front."""


def dominates(a: dict, b: dict, maximize: tuple[str, ...], minimize: tuple[str, ...]) -> bool:
    no_worse = all(a[m] >= b[m] for m in maximize) and all(a[m] <= b[m] for m in minimize)
    better = any(a[m] > b[m] for m in maximize) or any(a[m] < b[m] for m in minimize)
    return no_worse and better


def pareto_front(
    points: list[dict], maximize: tuple[str, ...] = ("quality",), minimize: tuple[str, ...] = ("cost", "latency")
) -> list[dict]:
    return [p for p in points if not any(dominates(q, p, maximize, minimize) for q in points if q is not p)]
