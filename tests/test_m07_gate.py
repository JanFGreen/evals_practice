from tests.helpers import load

m = load("exercises.m07_gate", "evalkit.gate")

POLICY = {
    "hard": [{"grader": "security:", "max_failures": 0}],
    "min_pass_rate": 0.7,
    "regression": {"max_drop": 0.05},
    "warn": {"cost_increase": 0.25, "latency_p95_increase": 0.3},
}
SAME = ([1] * 16 + [0] * 4, [1] * 16 + [0] * 4)


def summary(pass_rate, security_fails=0, cost=0.01, p95=1000):
    return {
        "pass_rate": pass_rate, "n_scored": 20, "cost_usd_per_row": cost, "latency_p95_ms": p95,
        "by_grader": {
            "security:canary": {"n": 20, "pass_rate": 1 - security_fails / 20, "fails": security_fails, "mean_score": None},
            "check:refusal": {"n": 20, "pass_rate": pass_rate, "fails": 0, "mean_score": None},
        },
    }


def test_pass():
    assert m.decide(summary(0.8), summary(0.8), SAME, POLICY).status == "pass"


def test_security_failure_is_a_hard_fail():
    r = m.decide(summary(0.8), summary(0.9, security_fails=1), SAME, POLICY)
    assert r.status == "fail"
    assert any("security" in f for f in r.failures)


def test_below_minimum_pass_rate():
    assert m.decide(summary(0.8), summary(0.6), SAME, POLICY).status == "fail"


def test_significant_regression_fails():
    r = m.decide(summary(1.0), summary(0.75), ([1] * 20, [1] * 12 + [0] * 8), POLICY)
    assert r.status == "fail"
    assert any("regres" in f for f in r.failures)


def test_drop_within_noise_only_warns():
    a = [1, 0] * 10
    b = a[:]
    b[0] = b[2] = 0
    r = m.decide(summary(0.8), summary(0.75), (a, b), POLICY)
    assert r.status == "warn"
    assert r.failures == []


def test_cost_increase_warns():
    r = m.decide(summary(0.8), summary(0.8, cost=0.02), SAME, POLICY)
    assert r.status == "warn"
    assert any("cost" in w for w in r.warnings)
