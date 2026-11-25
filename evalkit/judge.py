"""LLM-as-judge (module 4): correctness vs reference, faithfulness to context, pairwise with swap."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from evalkit import config
from evalkit.checks import contains_all, contains_any, not_contains, refusal_matches
from evalkit.grades import Grade
from evalkit.llm import complete_json

PROMPTS = Path(__file__).parent / "prompts"
SYSTEM = "You are a meticulous and impartial evaluator. You answer only in the required JSON format."
FAILURE_MODES = ["none", "wrong_fact", "outdated_policy", "missing_info", "hallucination", "should_refuse",
                 "unnecessary_refusal", "followed_injection", "other"]

CORRECTNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "verdict": {"type": "string", "enum": ["pass", "fail"]},
        "failure_mode": {"type": "string", "enum": FAILURE_MODES},
    },
    "required": ["reasoning", "verdict", "failure_mode"],
    "additionalProperties": False,
}
FAITHFULNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "supported": {"type": "string", "enum": ["yes", "partial", "no"]},
                    "evidence": {"type": "string"},
                },
                "required": ["claim", "supported", "evidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}
PAIRWISE_SCHEMA = {
    "type": "object",
    "properties": {"reasoning": {"type": "string"}, "winner": {"type": "string", "enum": ["A", "B", "tie"]}},
    "required": ["reasoning", "winner"],
    "additionalProperties": False,
}


def render(template: str, **values: str) -> str:
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)
    return template


@dataclass
class Verdict:
    verdict: str
    failure_mode: str
    reasoning: str
    model: str
    usage: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"


def _reference(case) -> str:
    if case.reference_answer:
        return case.reference_answer
    return "The assistant should say that it does not know." if case.expect_refusal else "(no reference)"


def _fake_correctness(case, answer: str) -> dict:
    checks = [refusal_matches(answer, case.expect_refusal)]
    if case.must_include_any:
        checks.append(contains_any(answer, case.must_include_any))
    if case.must_include_all:
        checks.append(contains_all(answer, case.must_include_all))
    if case.must_not_include:
        checks.append(not_contains(answer, case.must_not_include))
    ok = all(c.passed for c in checks)
    return {"reasoning": "[fake] heuristic based on deterministic checks",
            "verdict": "pass" if ok else "fail", "failure_mode": "none" if ok else "other"}


def judge_correctness(case, answer: str, *, prompt_path: str | Path | None = None, model: str | None = None,
                      rep: int = 0) -> Verdict:
    template = Path(prompt_path or PROMPTS / "judge_correctness.md").read_text(encoding="utf-8")
    prompt = render(template, question=case.question, reference=_reference(case), answer=answer)
    data, res = complete_json(
        model=model or config.judge_model(), system=SYSTEM, messages=[{"role": "user", "content": prompt}],
        max_tokens=2048, effort=config.judge_effort(), schema=CORRECTNESS_SCHEMA, rep=rep,
        fake=lambda: json.dumps(_fake_correctness(case, answer), ensure_ascii=False),
    )
    return Verdict(data["verdict"], data["failure_mode"], data["reasoning"], res.model, res.usage)


def correctness_grader(prompt_path: str | Path | None = None, model: str | None = None):
    def grade(case, output: dict) -> list[Grade]:
        v = judge_correctness(case, output.get("answer", ""), prompt_path=prompt_path, model=model,
                              rep=output.get("rep", 0))
        return [Grade("judge:correctness", v.passed, detail=f"{v.failure_mode}: {v.reasoning}", kind="judge")]

    grade.__name__ = "judge_correctness"
    return grade


def context_for(retrieved: list[dict]) -> str:
    from ragapp.retriever import load_chunks

    chunks = {c.id: c for c in load_chunks()}
    return "\n\n".join(f'<document id="{c.doc_id}">\n{c.text}\n</document>'
                       for r in retrieved if (c := chunks.get(r["id"])))


def _fake_faithfulness(answer: str, context: str) -> dict:
    from ragapp.retriever import tokenize

    ctx = set(tokenize(context))
    claims = []
    for sentence in re.split(r"(?<=[.!?])\s+", re.sub(r"\[[a-z0-9\-]+\]", "", answer)):
        toks = set(tokenize(sentence))
        if len(toks) < 3 or sentence.lower().startswith("i don't know"):
            continue
        overlap = len(toks & ctx) / len(toks)
        label = "yes" if overlap >= 0.6 else "partial" if overlap >= 0.3 else "no"
        claims.append({"claim": sentence.strip(), "supported": label, "evidence": f"[fake] overlap {overlap:.0%}"})
    return {"claims": claims}


def judge_faithfulness(answer: str, retrieved: list[dict], *, model: str | None = None, rep: int = 0) -> dict:
    context = context_for(retrieved)
    prompt = render((PROMPTS / "judge_faithfulness.md").read_text(encoding="utf-8"), context=context, answer=answer)
    data, res = complete_json(
        model=model or config.judge_model(), system=SYSTEM, messages=[{"role": "user", "content": prompt}],
        max_tokens=4096, effort=config.judge_effort(), schema=FAITHFULNESS_SCHEMA, rep=rep,
        fake=lambda: json.dumps(_fake_faithfulness(answer, context), ensure_ascii=False),
    )
    claims = data["claims"]
    if not claims:
        return {"score": 1.0, "passed": True, "claims": [], "model": res.model}
    weight = {"yes": 1.0, "partial": 0.5, "no": 0.0}
    score = sum(weight[c["supported"]] for c in claims) / len(claims)
    return {"score": score, "passed": all(c["supported"] != "no" for c in claims), "claims": claims, "model": res.model}


def faithfulness_grader(model: str | None = None):
    def grade(case, output: dict) -> list[Grade]:
        r = judge_faithfulness(output.get("answer", ""), output.get("retrieved", []), model=model,
                               rep=output.get("rep", 0))
        unsupported = [c["claim"] for c in r["claims"] if c["supported"] == "no"]
        return [Grade("judge:faithfulness", r["passed"], score=r["score"],
                      detail=f"unsupported: {unsupported}" if unsupported else "", kind="judge")]

    grade.__name__ = "judge_faithfulness"
    return grade


def _fake_pairwise(answer_a: str, answer_b: str) -> dict:
    # A deliberately biased fake judge (prefers position A and longer text), to make swap tests meaningful offline.
    winner = "A" if len(answer_a) >= 0.8 * len(answer_b) else "B"
    return {"reasoning": "[fake] biased judge", "winner": winner}


def judge_pairwise(question: str, answer_a: str, answer_b: str, *, model: str | None = None, rep: int = 0) -> str:
    prompt = render((PROMPTS / "judge_pairwise.md").read_text(encoding="utf-8"),
                    question=question, answer_a=answer_a, answer_b=answer_b)
    data, _ = complete_json(
        model=model or config.judge_model(), system=SYSTEM, messages=[{"role": "user", "content": prompt}],
        max_tokens=2048, effort=config.judge_effort(), schema=PAIRWISE_SCHEMA, rep=rep,
        fake=lambda: json.dumps(_fake_pairwise(answer_a, answer_b)),
    )
    return data["winner"]


def pairwise_with_swap(judge_fn: Callable[[str, str, str], str], question: str, a: str, b: str) -> dict:
    """Ask twice with positions swapped; trust only verdicts that survive the swap."""
    first = judge_fn(question, a, b)
    second = judge_fn(question, b, a)
    w1 = {"A": "a", "B": "b", "tie": "tie"}[first]
    w2 = {"A": "b", "B": "a", "tie": "tie"}[second]
    if w1 == w2:
        return {"winner": w1, "consistent": True, "raw": [first, second]}
    return {"winner": "tie", "consistent": False, "raw": [first, second]}
