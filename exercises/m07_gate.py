"""Module 7: regression gate. Tests: uv run pytest tests/test_m07_gate.py"""

from evalkit.gate import GateResult
from evalkit.stats import paired_bootstrap_diff  # noqa: F401 - you will need it


def decide(baseline: dict, candidate: dict, paired: tuple[list[float], list[float]], policy: dict) -> GateResult:
    """Return GateResult("pass"|"warn"|"fail", failures, warnings) according to the policy.

    - hard: sum of "fails" of graders whose name starts with rule["grader"] > max_failures -> fail
    - min_pass_rate: candidate["pass_rate"] below the threshold -> fail
    - regression: drop larger than max_drop and upper CI bound < 0 -> fail; drop alone -> warn
    - warn: increase of cost_usd_per_row / latency_p95_ms above the threshold -> warn
    """
    raise NotImplementedError
