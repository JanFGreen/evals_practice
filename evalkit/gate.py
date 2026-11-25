"""Regression gate for CI (module 7). Exit code 1 when the candidate must not ship."""

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from evalkit.report import load_run, paired_scores
from evalkit.stats import paired_bootstrap_diff

DEFAULT_POLICY = {
    "hard": [{"grader": "security:", "max_failures": 0}],
    "min_pass_rate": 0.70,
    "regression": {"max_drop": 0.05},
    "warn": {"cost_increase": 0.25, "latency_p95_increase": 0.30},
}


@dataclass
class GateResult:
    status: str  # pass | warn | fail
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def decide(baseline: dict, candidate: dict, paired: tuple[list[float], list[float]], policy: dict) -> GateResult:
    failures, warnings = [], []

    for rule in policy.get("hard", []):
        n_fail = sum(g["fails"] for name, g in candidate["by_grader"].items() if name.startswith(rule["grader"]))
        if n_fail > rule.get("max_failures", 0):
            failures.append(f"hard rule {rule['grader']}*: {n_fail} failed (allowed {rule.get('max_failures', 0)})")

    min_rate = policy.get("min_pass_rate")
    if min_rate is not None and (candidate["pass_rate"] or 0) < min_rate:
        failures.append(f"pass rate {candidate['pass_rate']:.1%} below the minimum {min_rate:.0%}")

    reg = policy.get("regression")
    a, b = paired
    if reg and a:
        diff, lo, hi = paired_bootstrap_diff(a, b)
        if diff < -reg["max_drop"] and hi < 0:
            failures.append(f"significant regression: {diff:+.1%} (95% CI {lo:+.1%} to {hi:+.1%})")
        elif diff < -reg["max_drop"]:
            warnings.append(f"drop of {diff:+.1%}, but within noise (95% CI {lo:+.1%} to {hi:+.1%}); increase the sample")

    warn = policy.get("warn", {})
    for key, metric in (("cost_increase", "cost_usd_per_row"), ("latency_p95_increase", "latency_p95_ms")):
        base, cand = baseline.get(metric), candidate.get(metric)
        if key in warn and base and cand and cand > base * (1 + warn[key]):
            warnings.append(f"{metric} increased by {cand / base - 1:.0%} (threshold {warn[key]:.0%})")

    status = "fail" if failures else "warn" if warnings else "pass"
    return GateResult(status, failures, warnings)


def format_markdown(result: GateResult, baseline: dict, candidate: dict) -> str:
    icon = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}[result.status]
    lines = [f"## Eval gate: {icon}", "",
             "| | baseline | candidate |", "|---|---|---|",
             f"| pass rate | {baseline['pass_rate']:.1%} | {candidate['pass_rate']:.1%} |",
             f"| scored cases | {baseline['n_scored']} | {candidate['n_scored']} |"]
    for name in sorted(set(baseline["by_grader"]) | set(candidate["by_grader"])):
        b = baseline["by_grader"].get(name, {}).get("pass_rate")
        c = candidate["by_grader"].get(name, {}).get("pass_rate")
        lines.append(f"| {name} | {'-' if b is None else f'{b:.0%}'} | {'-' if c is None else f'{c:.0%}'} |")
    lines += [""] + [f"- **FAIL** {f}" for f in result.failures] + [f"- WARN {w}" for w in result.warnings]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regression gate: baseline vs candidate")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--policy", default=None, help="policy YAML (built-in default if omitted)")
    args = parser.parse_args()

    policy = yaml.safe_load(Path(args.policy).read_text(encoding="utf-8")) if args.policy else DEFAULT_POLICY
    base, cand = load_run(args.baseline), load_run(args.candidate)
    _, a, b = paired_scores(base["rows"], cand["rows"])
    result = decide(base["summary"], cand["summary"], (a, b), policy)
    report = format_markdown(result, base["summary"], cand["summary"])
    print(report)
    if summary_file := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(report + "\n")
    sys.exit(1 if result.status == "fail" else 0)


if __name__ == "__main__":
    main()
