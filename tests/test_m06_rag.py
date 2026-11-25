import math

import pytest

from tests.helpers import load

m = load("exercises.m06_rag", "evalkit.rag_metrics")


def test_recall_and_precision_dedupe_first():
    retrieved, relevant = ["a", "b", "a", "c"], ["c", "d"]
    assert m.recall_at_k(retrieved, relevant, 3) == 0.5
    assert m.recall_at_k(retrieved, relevant, 2) == 0.0
    assert m.recall_at_k(retrieved, [], 3) is None
    assert m.precision_at_k(retrieved, relevant, 3) == pytest.approx(1 / 3)
    assert m.precision_at_k([], relevant, 3) == 0.0


def test_mrr():
    assert m.mrr(["x", "y", "z"], ["z"]) == pytest.approx(1 / 3)
    assert m.mrr(["x", "x", "z"], ["z"]) == pytest.approx(0.5)
    assert m.mrr(["x"], ["z"]) == 0.0


def test_ndcg():
    assert m.ndcg_at_k(["a", "b"], ["a"], 2) == pytest.approx(1.0)
    assert m.ndcg_at_k(["b", "a"], ["a"], 2) == pytest.approx(1 / math.log2(3))
    assert m.ndcg_at_k(["b", "c"], ["a"], 2) == 0.0


def test_attribute_failure():
    assert m.attribute_failure(True, True) == "ok"
    assert m.attribute_failure(True, False) == "generation"
    assert m.attribute_failure(False, False) == "retrieval"
    assert m.attribute_failure(False, True) == "correct_without_context"
