import pytest

from tests.helpers import load

m = load("exercises.m02_report", "evalkit.report")


def row(case_id, passed, status="ok", tags=("fact",)):
    return {"case_id": case_id, "rep": 0, "tags": list(tags), "status": status, "passed": passed,
            "grades": [], "output": {}}


def test_errors_and_truncated_are_counted_but_not_scored():
    rows = [row("a", True), row("b", False), row("c", None, status="truncated"), row("d", True, tags=("fact", "edge"))]
    s = m.summarize_rows(rows, [{"case_id": "e", "kind": "timeout"}])
    assert (s["n_rows"], s["n_scored"], s["n_errors"]) == (4, 3, 1)
    assert s["pass_rate"] == pytest.approx(2 / 3)


def test_breakdown_by_tag():
    rows = [row("a", True), row("b", False), row("d", True, tags=("fact", "edge"))]
    s = m.summarize_rows(rows, [])
    assert s["by_tag"]["edge"] == {"n": 1, "pass_rate": 1.0}
    assert s["by_tag"]["fact"]["n"] == 3
    assert s["by_tag"]["fact"]["pass_rate"] == pytest.approx(2 / 3)


def test_empty_run():
    s = m.summarize_rows([], [])
    assert s["n_scored"] == 0
    assert s["pass_rate"] is None
