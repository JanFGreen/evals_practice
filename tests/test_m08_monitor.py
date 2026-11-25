from collections import Counter

import pytest

from tests.helpers import load

m = load("exercises.m08_monitor", "evalkit.monitor")


def test_psi_is_zero_for_same_distribution():
    assert m.psi({"a": 10, "b": 10}, {"a": 20, "b": 20}) == pytest.approx(0.0)


def test_psi_detects_shift_and_new_categories():
    assert m.psi({"a": 90, "b": 10}, {"a": 50, "b": 50}) > 0.25
    assert m.psi({"a": 10}, {"a": 5, "nowy": 5}) > 0.25


def traces(n_down=3):
    return [{"id": i, "feedback": "down" if i < n_down else None,
             "retrieval_top_score": 0.5 if i in (n_down, n_down + 1) else 4.0, "status": "ok"} for i in range(30)]


def test_sample_for_review_mixes_flagged_and_random():
    picked = m.sample_for_review(traces(), budget=10, seed=0)
    ids = [p["id"] for p in picked]
    assert len(ids) == len(set(ids)) == 10
    assert {0, 1, 2, 3, 4} <= set(ids)
    assert Counter(p["review_reason"] for p in picked) == {"thumbs_down": 3, "low_retrieval": 2, "random": 5}


def test_flagged_share_is_capped_at_half_budget():
    picked = m.sample_for_review(traces(n_down=8), budget=6, seed=0)
    reasons = Counter(p["review_reason"] for p in picked)
    assert reasons["thumbs_down"] == 3
    assert reasons["random"] == 3


def test_sample_for_review_is_seeded():
    assert m.sample_for_review(traces(), 10, seed=1) == m.sample_for_review(traces(), 10, seed=1)
