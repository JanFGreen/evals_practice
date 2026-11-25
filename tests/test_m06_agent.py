import pytest

from tests.helpers import load

m = load("exercises.m06_agent", "evalkit.agent_metrics")

EXPECTED = [{"tool": "file_incident_report", "args": {"kind": "accident", "location": "B"}}]


def test_extra_calls_lower_precision_not_recall():
    actual = [{"tool": "search_docs", "args": {"query": "x"}},
              {"tool": "file_incident_report", "args": {"kind": "accident", "location": "hall B", "anonymous": False}}]
    r = m.tool_calls_match(EXPECTED, actual)
    assert (r["recall"], r["precision"], r["unexpected"], r["missing"]) == (1.0, 0.5, 1, [])


def test_wrong_arguments_do_not_match():
    actual = [{"tool": "file_incident_report", "args": {"kind": "dangerous_occurrence", "location": "B"}}]
    r = m.tool_calls_match(EXPECTED, actual)
    assert r["recall"] == 0.0
    assert r["missing"] == EXPECTED


def test_each_actual_call_is_used_once():
    r = m.tool_calls_match([{"tool": "search_docs"}, {"tool": "search_docs"}], [{"tool": "search_docs", "args": {}}])
    assert r["recall"] == 0.5


def test_no_calls_expected_and_none_made():
    r = m.tool_calls_match([], [])
    assert (r["recall"], r["precision"]) == (1.0, 1.0)


def test_pass_at_k():
    assert m.pass_at_k(5, 5, 1) == 1.0
    assert m.pass_at_k(5, 0, 3) == 0.0
    assert m.pass_at_k(4, 2, 1) == pytest.approx(0.5)
    assert m.pass_at_k(4, 2, 2) == pytest.approx(5 / 6)


def test_pass_hat_k():
    assert m.pass_hat_k(4, 2, 2) == pytest.approx(1 / 6)
    assert m.pass_hat_k(4, 4, 4) == 1.0
    assert m.pass_hat_k(4, 3, 4) == 0.0
