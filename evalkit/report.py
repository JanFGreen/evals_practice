import json
from collections import Counter
from pathlib import Path
from statistics import fmean, quantiles

from evalkit.stats import mcnemar_exact, paired_bootstrap_diff, per_case_means, wilson_interval


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def load_run(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    return {
        "rows": read_jsonl(run_dir / "results.jsonl"),
        "errors": read_jsonl(run_dir / "errors.jsonl"),
        "summary": json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {},
    }


def _pct(values: list[float], q: int) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return quantiles(values, n=100)[q - 1]


def summarize_rows(rows: list[dict], errors: list[dict] = ()) -> dict:
    """Errors and truncated/refused outputs are counted, but never averaged in as failures."""
    scored = [r for r in rows if r["status"] == "ok" and r["passed"] is not None]
    n_pass = sum(bool(r["passed"]) for r in scored)

    by_tag: dict[str, list[bool]] = {}
    for r in scored:
        for tag in r.get("tags", []):
            by_tag.setdefault(tag, []).append(bool(r["passed"]))

    graders: dict[str, dict] = {}
    for r in rows:
        for g in r.get("grades", []):
            e = graders.setdefault(g["name"], {"n": 0, "passed": 0, "scores": []})
            if g["passed"] is not None:
                e["n"] += 1
                e["passed"] += bool(g["passed"])
            if g.get("score") is not None:
                e["scores"].append(g["score"])

    outputs = [r["output"] for r in rows if r.get("output")]
    costs = [o["cost_usd"] for o in outputs if o.get("cost_usd") is not None]
    latencies = [o["latency_ms"] for o in outputs if o.get("latency_ms") and not o.get("cached")]
    lo, hi = wilson_interval(n_pass, len(scored))
    return {
        "n_rows": len(rows),
        "n_scored": len(scored),
        "n_errors": len(errors),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "pass_rate": n_pass / len(scored) if scored else None,
        "pass_rate_ci95": [round(lo, 4), round(hi, 4)],
        "by_tag": {t: {"n": len(v), "pass_rate": fmean(v)} for t, v in sorted(by_tag.items())},
        "by_grader": {
            name: {
                "n": e["n"],
                "pass_rate": e["passed"] / e["n"] if e["n"] else None,
                "fails": e["n"] - e["passed"],
                "mean_score": fmean(e["scores"]) if e["scores"] else None,
            }
            for name, e in sorted(graders.items())
        },
        "cost_usd_total": round(sum(costs), 6) if costs else None,
        "cost_usd_per_row": round(sum(costs) / len(costs), 6) if costs else None,
        "latency_p50_ms": _pct(latencies, 50),
        "latency_p95_ms": _pct(latencies, 95),
    }


def paired_scores(rows_a: list[dict], rows_b: list[dict]) -> tuple[list[str], list[float], list[float]]:
    a, b = per_case_means(rows_a), per_case_means(rows_b)
    common = sorted(set(a) & set(b))
    return common, [a[c] for c in common], [b[c] for c in common]


def compare_runs(run_a: str | Path, run_b: str | Path) -> dict:
    ra, rb = load_run(run_a), load_run(run_b)
    common, a, b = paired_scores(ra["rows"], rb["rows"])
    if not common:
        raise ValueError("the runs have no scored cases in common")
    diff, lo, hi = paired_bootstrap_diff(a, b)
    regressions = [c for c, x, y in zip(common, a, b) if x >= 0.5 > y]
    fixes = [c for c, x, y in zip(common, a, b) if y >= 0.5 > x]
    return {
        "n_common": len(common),
        "pass_rate_a": fmean(a),
        "pass_rate_b": fmean(b),
        "diff": diff,
        "diff_ci95": [lo, hi],
        "regressions": regressions,
        "fixes": fixes,
        "mcnemar_p": mcnemar_exact(len(regressions), len(fixes)),
        "graders_a": ra["summary"].get("by_grader", {}),
        "graders_b": rb["summary"].get("by_grader", {}),
    }


def format_summary(summary: dict) -> str:
    lines = []
    pr = summary.get("pass_rate")
    ci = summary.get("pass_rate_ci95", [0, 0])
    head = f"pass rate: {pr:.1%}  (95% CI {ci[0]:.1%}–{ci[1]:.1%})" if pr is not None else "pass rate: no scores"
    lines.append(head)
    lines.append(f"scored: {summary['n_scored']}/{summary['n_rows']}  infrastructure errors: {summary['n_errors']}  "
                 f"statuses: {summary['status_counts']}")
    if summary.get("cost_usd_total") is not None:
        lines.append(f"cost: ${summary['cost_usd_total']:.4f}  (${summary['cost_usd_per_row']:.4f}/case)")
    if summary.get("latency_p95_ms"):
        lines.append(f"latency: p50 {summary['latency_p50_ms']:.0f} ms, p95 {summary['latency_p95_ms']:.0f} ms")
    lines.append("\nGrader                          n    pass   mean score")
    for name, g in summary["by_grader"].items():
        p = f"{g['pass_rate']:.0%}" if g["pass_rate"] is not None else "  - "
        s = f"{g['mean_score']:.2f}" if g["mean_score"] is not None else ""
        lines.append(f"{name:30s} {g['n']:3d}  {p:>5}   {s}")
    lines.append("\nTag                  n    pass")
    for tag, t in summary["by_tag"].items():
        lines.append(f"{tag:18s} {t['n']:3d}   {t['pass_rate']:.0%}")
    return "\n".join(lines)


def format_comparison(cmp: dict, label_a: str = "A", label_b: str = "B") -> str:
    lo, hi = cmp["diff_ci95"]
    verdict = "significant difference" if lo > 0 or hi < 0 else "difference within noise"
    return "\n".join([
        f"common cases: {cmp['n_common']}",
        f"{label_a}: {cmp['pass_rate_a']:.1%}   {label_b}: {cmp['pass_rate_b']:.1%}",
        f"difference ({label_b} - {label_a}): {cmp['diff']:+.1%}  (95% CI {lo:+.1%} to {hi:+.1%}) → {verdict}",
        f"McNemar p = {cmp['mcnemar_p']:.3f}",
        f"regressions (pass→fail): {cmp['regressions'] or '-'}",
        f"fixed (fail→pass): {cmp['fixes'] or '-'}",
    ])


def render_review_markdown(rows: list[dict], cases_by_id: dict) -> str:
    out = ["# Answer review (error analysis)\n",
           "For each case, write one sentence about what went wrong (open coding). "
           "Do not invent categories up front.\n"]
    for r in rows:
        case = cases_by_id.get(r["case_id"])
        o = r.get("output") or {}
        verdict = {True: "PASS", False: "FAIL", None: "—"}[r.get("passed")]
        out.append(f"## {r['case_id']} · {', '.join(r.get('tags', []))} · {verdict}\n")
        out.append(f"**Question:** {r.get('question') or (case.question if case else '')}\n")
        answer = (o.get("answer") or "").strip().replace("\n", "\n> ")
        out.append(f"**Answer:**\n> {answer}\n")
        out.append(f"**Documents in context:** {', '.join(o.get('retrieved_doc_ids', [])) or 'none'}\n")
        if case and case.reference_answer:
            out.append(f"**Reference:** {case.reference_answer}\n")
        failed = [g for g in r.get("grades", []) if g["passed"] is False]
        if failed:
            out.append("**Failed checks:** " + "; ".join(f"`{g['name']}` {g['detail']}" for g in failed) + "\n")
        out.append("**Note:** \n\n---\n")
    return "\n".join(out)
