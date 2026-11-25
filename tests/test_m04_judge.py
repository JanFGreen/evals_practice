import pytest

from tests.helpers import load

m = load("exercises.m04_judge", "evalkit.judge", "evalkit.alignment")


def test_swap_keeps_a_consistent_verdict():
    def judge(q, a, b):
        return "A" if "23" in a else "B"

    assert m.pairwise_with_swap(judge, "q", "23 kg", "25 kg") == {"winner": "a", "consistent": True, "raw": ["A", "B"]}


def test_swap_exposes_position_bias():
    result = m.pairwise_with_swap(lambda q, a, b: "A", "q", "x", "y")
    assert result["winner"] == "tie"
    assert result["consistent"] is False


def test_swap_tie_both_ways_is_consistent():
    result = m.pairwise_with_swap(lambda q, a, b: "tie", "q", "x", "y")
    assert result["winner"] == "tie"
    assert result["consistent"] is True


def test_confusion_and_rates():
    pred = ["pass", "pass", "fail", "fail", "pass"]
    gold = ["pass", "fail", "fail", "pass", "pass"]
    cm = m.confusion(pred, gold)
    assert cm == {"tp": 2, "fp": 1, "tn": 1, "fn": 1}
    r = m.rates(cm)
    assert r["tpr"] == pytest.approx(2 / 3)
    assert r["tnr"] == pytest.approx(0.5)
    assert r["accuracy"] == pytest.approx(0.6)


def test_rates_with_no_negatives():
    assert m.rates({"tp": 3, "fp": 0, "tn": 0, "fn": 0})["tnr"] is None


def test_cohen_kappa():
    assert m.cohen_kappa(["pass", "fail"] * 5, ["pass", "fail"] * 5) == pytest.approx(1.0)
    assert m.cohen_kappa(["pass", "pass", "fail", "fail"], ["pass", "fail", "pass", "fail"]) == pytest.approx(0.0)


def test_corrected_pass_rate():
    # true rate 0.8 seen through a judge with TPR 0.9 / TNR 0.8 shows up as 0.76
    assert m.corrected_pass_rate(0.76, 0.9, 0.8) == pytest.approx(0.8)
    with pytest.raises(ValueError):
        m.corrected_pass_rate(0.5, 0.5, 0.5)
