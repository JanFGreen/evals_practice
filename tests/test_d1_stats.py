import pytest

from tests.helpers import load

m = load("exercises.d1_stats", "evalkit.stats")


def test_wilson_known_values():
    lo, hi = m.wilson_interval(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3)
    assert hi == pytest.approx(0.9433, abs=1e-3)


def test_wilson_edges():
    assert m.wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = m.wilson_interval(0, 20)
    assert lo == pytest.approx(0.0, abs=1e-12)
    assert 0 < hi < 0.2


def test_paired_bootstrap_diff():
    a = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]
    b = [1, 1, 1, 0, 1, 1, 1, 0, 1, 1]
    diff, lo, hi = m.paired_bootstrap_diff(a, b, seed=0)
    assert diff == pytest.approx(0.3)
    assert lo <= diff <= hi
    assert hi - lo < 0.8


def test_paired_bootstrap_identical_runs_and_bad_input():
    assert m.paired_bootstrap_diff([1, 0, 1], [1, 0, 1]) == (0, 0, 0)
    with pytest.raises(ValueError):
        m.paired_bootstrap_diff([1], [1, 0])


def test_mcnemar_exact():
    assert m.mcnemar_exact(0, 0) == 1.0
    assert m.mcnemar_exact(0, 6) == pytest.approx(0.03125)
    assert m.mcnemar_exact(3, 3) == pytest.approx(1.0)
