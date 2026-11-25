"""promptfoo python assertion: the evalkit LLM judge (calibrated in module 4) instead of a generic rubric."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit.dataset import Case  # noqa: E402
from evalkit.judge import judge_correctness  # noqa: E402


def get_assert(output: str, context: dict) -> dict:
    v = context["vars"]
    case = Case(id=v.get("case_id", "promptfoo"), question=v["question"], reference_answer=v.get("reference"),
                expect_refusal=bool(v.get("expect_refusal")), must_include_any=v.get("must_include_any", []),
                must_include_all=v.get("must_include_all", []), must_not_include=v.get("must_not_include", []))
    verdict = judge_correctness(case, output)
    return {"pass": verdict.passed, "score": 1.0 if verdict.passed else 0.0,
            "reason": f"{verdict.failure_mode}: {verdict.reasoning}"}
