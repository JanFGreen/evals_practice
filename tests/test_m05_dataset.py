import pytest

from tests.helpers import load

m = load("exercises.m05_dataset", "evalkit.dataset")

DIMS = {"persona": ["a", "b", "c"], "topic": ["x", "y", "z", "w"], "difficulty": ["easy", "hard"]}


def test_jaccard():
    assert m.jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert m.jaccard(set(), set()) == 1.0


def test_near_duplicates():
    items = [("1", "How many kg can I lift alone?"), ("2", "how many kg can i lift alone"), ("3", "What is the hotel limit?")]
    assert [(a, b) for a, b, _ in m.near_duplicates(items, 0.8)] == [("1", "2")]


def test_sample_tuples_covers_every_value():
    out = m.sample_tuples(DIMS, 5, seed=1)
    assert len(out) == 5
    assert len({tuple(sorted(o.items())) for o in out}) == 5
    for key, values in DIMS.items():
        assert {o[key] for o in out} == set(values)


def test_sample_tuples_is_deterministic_and_complete_when_n_is_large():
    assert m.sample_tuples(DIMS, 5, seed=7) == m.sample_tuples(DIMS, 5, seed=7)
    assert len(m.sample_tuples(DIMS, 100, seed=0)) == 24
