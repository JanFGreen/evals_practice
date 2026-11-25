"""Module 2: run summary. Tests: uv run pytest tests/test_m02_report.py"""


def summarize_rows(rows: list[dict], errors: list[dict] = ()) -> dict:
    """Return n_rows, n_scored, n_errors, pass_rate, by_tag ({tag: {"n", "pass_rate"}}).

    Only rows with status == "ok" and passed not None are scored; errors and truncated answers
    are counted separately and never count towards the pass rate as failures.
    """
    raise NotImplementedError
