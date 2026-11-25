"""The eval loop (module 2): cases -> target -> graders -> results.jsonl / errors.jsonl / summary.json."""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable

from evalkit import config
from evalkit.checks import run_checks
from evalkit.dataset import Case
from evalkit.grades import Grade
from evalkit.llm import LLMError
from evalkit.report import format_summary, read_jsonl, summarize_rows, write_jsonl

Target = Callable[[Case, int], dict]
Grader = Callable[[Case, dict], list[Grade]]


def app_target(prompt_version: str = "v2", k: int = 4, model: str | None = None, enforce_tenant: bool = True) -> Target:
    from ragapp.app import answer

    def run(case: Case, rep: int) -> dict:
        return answer(case.question, case.workspace, prompt_version=prompt_version, k=k, model=model,
                      enforce_tenant=enforce_tenant, rep=rep).to_dict()

    run.meta = {"prompt_version": prompt_version, "k": k, "model": model or config.app_model()}
    return run


def outputs_target(path: str | Path) -> Target:
    """Grade outputs recorded earlier instead of calling the app (offline exercises, re-grading)."""
    by_case = {row["case_id"]: row for row in read_jsonl(path)}

    def run(case: Case, rep: int) -> dict:
        if case.id not in by_case:
            raise KeyError(f"no saved answer for {case.id}")
        return {"status": "ok", **by_case[case.id]}

    run.meta = {"outputs": str(path)}
    return run


def checks_grader(require_citations: bool = True) -> Grader:
    def grade(case: Case, output: dict) -> list[Grade]:
        return [Grade(f"check:{r.name}", r.passed, detail=r.detail, kind="check")
                for r in run_checks(case, output, require_citations=require_citations)]

    grade.__name__ = "checks"
    return grade


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=config.ROOT, check=True).stdout.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _run_one(case: Case, rep: int, target: Target, graders: list[Grader]) -> tuple[dict | None, dict | None]:
    try:
        output = target(case, rep)
    except LLMError as e:
        return None, {"case_id": case.id, "rep": rep, "kind": e.kind, "error": str(e)}
    except Exception as e:  # noqa: BLE001 - harness bugs must land in errors.jsonl, not as model failures
        return None, {"case_id": case.id, "rep": rep, "kind": "harness", "error": repr(e)}

    output = {**output, "rep": rep}
    status = output.get("status", "ok")
    grades: list[Grade] = []
    if status == "ok":
        for grader in graders:
            try:
                grades.extend(grader(case, output))
            except LLMError as e:
                grades.append(Grade(f"{getattr(grader, '__name__', 'grader')}:error", None,
                                    detail=f"grader failed: {e}", kind="error"))
    gradable = [g for g in grades if g.passed is not None]
    passed = all(g.passed for g in gradable) if status == "ok" and gradable else None
    row = {
        "case_id": case.id,
        "rep": rep,
        "tags": case.tags,
        "question": case.question,
        "status": status,
        "passed": passed,
        "grades": [asdict(g) for g in grades],
        "output": output,
    }
    return row, None


def run_eval(
    cases: list[Case],
    target: Target,
    graders: list[Grader],
    *,
    reps: int = 1,
    label: str = "run",
    out_dir: str | Path | None = None,
    max_workers: int = 4,
    quiet: bool = False,
) -> Path:
    run_id = f"{label}-{datetime.now():%Y%m%d-%H%M%S}"
    out = Path(out_dir) if out_dir else config.RUNS_DIR / run_id
    out.mkdir(parents=True, exist_ok=True)

    tasks = [(c, r) for c in cases for r in range(reps)]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(lambda t: _run_one(t[0], t[1], target, graders), tasks))

    rows = sorted((r for r, _ in results if r), key=lambda r: (r["case_id"], r["rep"]))
    errors = [e for _, e in results if e]
    write_jsonl(out / "results.jsonl", rows)
    write_jsonl(out / "errors.jsonl", errors)

    summary = summarize_rows(rows, errors)
    summary["meta"] = {
        "run_id": run_id,
        "label": label,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_cases": len(cases),
        "reps": reps,
        "graders": [getattr(g, "__name__", "grader") for g in graders],
        "llm_mode": config.llm_mode(),
        "git_commit": _git_commit(),
        **getattr(target, "meta", {}),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if not quiet:
        print(f"\n== {run_id} → {out}")
        print(format_summary(summary))
        if errors:
            kinds = sorted({e["kind"] for e in errors})
            print(f"\n{len(errors)} cases without a result (errors.jsonl), kinds: {kinds}")
    return out
