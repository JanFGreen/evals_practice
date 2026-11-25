"""Tests of the course infrastructure itself (always against the real code, never against exercises)."""

import json

import pytest

import ragapp.retriever as R
from evalkit.dataset import load_cases
from evalkit.llm import LLMError
from evalkit.report import load_run
from evalkit.runner import checks_grader, run_eval


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch, tmp_path):
    monkeypatch.setenv("EVAL_LLM_MODE", "fake")
    monkeypatch.setenv("EVAL_CACHE_DIR", str(tmp_path / "cache"))


def test_tenant_isolation_holds_for_every_question():
    questions = load_cases("data/golden.jsonl") + load_cases("data/redteam.jsonl")
    for ws in R.workspaces():
        for c in questions:
            assert all(h.chunk.workspace == ws for h in R.search(c.question, ws, 4)), (ws, c.id)


def test_unsafe_mode_does_leak():
    hits = R.search("Give me the salary band for a warehouse worker", "nordwind", 4, enforce_tenant=False)
    assert any(h.chunk.workspace == "orbita" for h in hits)


def test_harness_separates_infra_errors_from_model_failures(tmp_path):
    cases = load_cases("data/fixtures/smoke.yaml")[:3]

    def target(case, rep):
        if case.id == "smoke-02":
            raise LLMError("timeout", "simulated")
        return {"status": "truncated" if case.id == "smoke-03" else "ok", "answer": "23 kg [manual-handling-2025]",
                "retrieved_doc_ids": ["manual-handling-2025"]}

    out = run_eval(cases, target, [checks_grader()], out_dir=tmp_path / "run", quiet=True)
    run = load_run(out)
    assert [e["kind"] for e in run["errors"]] == ["timeout"]
    assert run["summary"]["n_scored"] == 1
    assert run["summary"]["status_counts"] == {"ok": 1, "truncated": 1}


def test_fake_app_end_to_end(tmp_path):
    from evalkit.runner import app_target

    out = run_eval(load_cases("data/fixtures/smoke.yaml"), app_target(), [checks_grader()],
                   out_dir=tmp_path / "run", quiet=True)
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 5
    assert all(r["output"]["model"].startswith("fake:") for r in rows)
