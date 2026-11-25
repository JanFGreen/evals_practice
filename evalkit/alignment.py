"""Judge vs human agreement (module 4)."""

from collections import Counter


def confusion(pred: list[str], gold: list[str], positive: str = "pass") -> dict[str, int]:
    if len(pred) != len(gold):
        raise ValueError("pred and gold must have the same length")
    cm = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for p, g in zip(pred, gold):
        if p == positive and g == positive:
            cm["tp"] += 1
        elif p == positive:
            cm["fp"] += 1
        elif g == positive:
            cm["fn"] += 1
        else:
            cm["tn"] += 1
    return cm


def rates(cm: dict[str, int]) -> dict[str, float | None]:
    def div(a, b):
        return a / b if b else None

    total = sum(cm.values())
    return {
        "tpr": div(cm["tp"], cm["tp"] + cm["fn"]),  # human pass -> judge pass
        "tnr": div(cm["tn"], cm["tn"] + cm["fp"]),  # human fail -> judge fail (catching failures)
        "precision": div(cm["tp"], cm["tp"] + cm["fp"]),
        "accuracy": div(cm["tp"] + cm["tn"], total),
    }


def cohen_kappa(pred: list[str], gold: list[str]) -> float:
    n = len(gold)
    if n == 0:
        raise ValueError("empty lists")
    po = sum(p == g for p, g in zip(pred, gold)) / n
    cp, cg = Counter(pred), Counter(gold)
    pe = sum(cp[label] * cg[label] for label in set(cp) | set(cg)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def corrected_pass_rate(observed: float, tpr: float, tnr: float) -> float:
    """Rogan-Gladen: estimate of the true pass rate given the judge's measured error rates."""
    denom = tpr + tnr - 1
    if denom <= 0:
        raise ValueError("the judge is no better than chance (tpr + tnr <= 1)")
    return min(1.0, max(0.0, (observed + tnr - 1) / denom))
