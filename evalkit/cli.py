"""Course CLI: uv run python -m evalkit.cli <command> --help"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from statistics import fmean

import yaml

from evalkit import config
from evalkit.dataset import Case, audit_cases, load_cases, sample_tuples, stratified_split
from evalkit.report import (compare_runs, format_comparison, load_run, read_jsonl, render_review_markdown,
                            write_jsonl)
from evalkit.runner import app_target, checks_grader, outputs_target, run_eval

GOLDEN = config.DATA_DIR / "golden.jsonl"
SAMPLE = config.DATA_DIR / "runs" / "v1_sample"


def _graders(names: str, k: int = 4):
    from evalkit.judge import correctness_grader, faithfulness_grader
    from evalkit.rag_metrics import retrieval_grader
    from evalkit.security import security_grader

    table = {
        "checks": lambda: checks_grader(),
        "checks-nocite": lambda: checks_grader(require_citations=False),
        "judge": lambda: correctness_grader(),
        "faith": lambda: faithfulness_grader(),
        "retrieval": lambda: retrieval_grader(k),
        "security": lambda: security_grader,
    }
    unknown = [n for n in names.split(",") if n not in table]
    if unknown:
        sys.exit(f"unknown graders: {unknown}; available: {', '.join(table)}")
    return [table[n]() for n in names.split(",")]


def _select(cases: list[Case], args) -> list[Case]:
    if getattr(args, "tags", None):
        wanted = set(args.tags.split(","))
        cases = [c for c in cases if wanted & set(c.tags)]
    if getattr(args, "ids", None):
        wanted = set(args.ids.split(","))
        cases = [c for c in cases if c.id in wanted]
    if getattr(args, "limit", None):
        cases = cases[: args.limit]
    return cases


def cmd_ask(args):
    from ragapp.app import answer

    r = answer(args.question, args.workspace, prompt_version=args.prompt, k=args.k)
    print(json.dumps(r.to_dict(), ensure_ascii=False, indent=2) if args.json else r.answer)


def cmd_run(args):
    cases = _select(load_cases(args.cases), args)
    target = app_target(args.prompt, args.k, args.model, enforce_tenant=not args.unsafe_tenant)
    run_eval(cases, target, _graders(args.graders, args.k), reps=args.reps,
             label=args.label or f"{args.prompt}", out_dir=args.out, max_workers=args.workers)


def cmd_grade(args):
    cases = _select(load_cases(args.cases), args)
    recorded = {r["case_id"] for r in read_jsonl(args.outputs)}
    cases = [c for c in cases if c.id in recorded]
    run_eval(cases, outputs_target(args.outputs), _graders(args.graders), label=args.label, max_workers=args.workers)


def cmd_render(args):
    run = load_run(args.run)
    cases = {c.id: c for c in load_cases(args.cases)}
    out = Path(args.out or f"notes/review-{Path(args.run).name}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_review_markdown(run["rows"], cases), encoding="utf-8")
    print(f"saved {out} ({len(run['rows'])} cases)")


def cmd_compare(args):
    print(format_comparison(compare_runs(args.run_a, args.run_b), Path(args.run_a).name, Path(args.run_b).name))


def oracle_answer(case: Case) -> str:
    if case.expect_refusal:
        return "I don't know — it is not in the documents. Contact the HR department (hr@nordwind.example)."
    return f"{case.reference_answer} " + " ".join(f"[{d}]" for d in case.relevant_doc_ids)


def cmd_sanity(args):
    cases = _select(load_cases(args.cases), args)
    graders = _graders(args.graders)
    for name, make in (("oracle", oracle_answer), ("null", lambda c: "")):
        def target(case, rep, make=make):
            return {"status": "ok", "answer": make(case), "retrieved_doc_ids": case.relevant_doc_ids, "retrieved": []}
        target.meta = {"sanity": name}
        print(f"\n######## {name.upper()} — {'should be ~100%' if name == 'oracle' else 'should be ~0%'}")
        run_eval(cases, target, graders, label=f"sanity-{name}")


def cmd_align(args):
    from evalkit.alignment import cohen_kappa, confusion, corrected_pass_rate, rates
    from evalkit.judge import judge_correctness

    cases = {c.id: c for c in load_cases(args.cases)}
    outputs = {r["case_id"]: r for r in read_jsonl(args.outputs)}
    labels = [lab for lab in read_jsonl(args.labels) if lab["case_id"] in outputs]
    dev, test = stratified_split(labels, key=lambda lab: lab["label"], test_frac=0.5, seed=42)
    chosen = {"dev": dev, "test": test, "all": labels}[args.split]

    pred, gold, disagreements = [], [], []
    for lab in sorted(chosen, key=lambda x: x["case_id"]):
        v = judge_correctness(cases[lab["case_id"]], outputs[lab["case_id"]]["answer"], prompt_path=args.prompt)
        pred.append(v.verdict)
        gold.append(lab["label"])
        if v.verdict != lab["label"]:
            disagreements.append((lab, v))

    cm = confusion(pred, gold)
    r = rates(cm)
    print(f"split={args.split}  n={len(gold)}  prompt={args.prompt or 'default'}")
    print(f"confusion matrix: {cm}")
    print("TPR (human pass → judge pass): " + (f"{r['tpr']:.0%}" if r["tpr"] is not None else "-"))
    print("TNR (human fail → judge fail): " + (f"{r['tnr']:.0%}" if r["tnr"] is not None else "-"))
    print(f"accuracy: {r['accuracy']:.0%}   Cohen kappa: {cohen_kappa(pred, gold):.2f}")
    observed = pred.count("pass") / len(pred)
    if r["tpr"] and r["tnr"] and r["tpr"] + r["tnr"] > 1:
        print(f"pass rate per judge: {observed:.0%} → corrected for judge errors: "
              f"{corrected_pass_rate(observed, r['tpr'], r['tnr']):.0%} (per humans: {gold.count('pass') / len(gold):.0%})")
    for lab, v in disagreements:
        print(f"\n✗ {lab['case_id']}: human={lab['label']} ({lab.get('failure_mode', '')}), judge={v.verdict}"
              f" ({v.failure_mode})\n  judge: {v.reasoning}\n  human: {lab.get('note', '')}")


def cmd_bias(args):
    from evalkit.judge import judge_pairwise, pairwise_with_swap

    cases = [c for c in load_cases(args.cases) if c.reference_answer and not c.expect_refusal][: args.limit]
    padding = (" Remember that safety at work comes first, and following procedures protects you"
               " and your colleagues. If you have any doubts at all, check with your manager.")
    first_pos = verbose_wins = concise_wins = ties = inconsistent = 0
    for c in cases:
        concise = c.reference_answer
        # position bias: identical answers, any non-tie is position preference
        if judge_pairwise(c.question, concise, concise) == "A":
            first_pos += 1
        res = pairwise_with_swap(judge_pairwise, c.question, concise, concise + padding)
        inconsistent += not res["consistent"]
        verbose_wins += res["winner"] == "b"
        concise_wins += res["winner"] == "a"
        ties += res["winner"] == "tie"
    n = len(cases)
    print(f"n={n}")
    print(f"identical answers, position A chosen: {first_pos}/{n} (unbiased ≈ 0, because it should be a tie)")
    print(f"concise vs the same + padding (with swap): concise {concise_wins}, verbose {verbose_wins},"
          f" tie {ties}; inconsistent after swap: {inconsistent}")


def cmd_audit(args):
    report = audit_cases(load_cases(args.cases))
    print(yaml.safe_dump(report, allow_unicode=True, sort_keys=False))


def cmd_synth(args):
    from evalkit.llm import complete_json

    dims = yaml.safe_load(Path(args.dimensions).read_text(encoding="utf-8"))
    schema = {"type": "object", "properties": {"question": {"type": "string"}, "expect_refusal": {"type": "boolean"}},
              "required": ["question", "expect_refusal"], "additionalProperties": False}
    out = []
    for i, t in enumerate(sample_tuples(dims, args.n, seed=args.seed), 1):
        prompt = (f"Write one realistic question that an employee of a logistics company could ask the company's "
                  f"policy assistant.\nPersona: {t['persona']}\nTopic: {t['topic']}\n"
                  f"Question type: {t['difficulty']}\nWrite it the way this person would really write it. Do not add "
                  f"an answer. Set expect_refusal=true only for a question the policies probably do not answer.")
        fake = json.dumps({"question": f"[fake] {t['persona']} pyta: {t['topic']} ({t['difficulty']})?",
                           "expect_refusal": "spoza" in t["difficulty"]}, ensure_ascii=False)
        data, _ = complete_json(model=config.judge_model(), system="You generate test data.",
                                messages=[{"role": "user", "content": prompt}], max_tokens=1024, effort="low",
                                schema=schema, rep=i, fake=lambda fake=fake: fake)
        out.append({"id": f"s{i:03d}", "question": data["question"], "expect_refusal": data["expect_refusal"],
                    "tags": [t["topic"], t["difficulty"]], "source": "synthetic", "notes": f"persona: {t['persona']}"})
    write_jsonl(args.out, out)
    print(f"saved {len(out)} candidates to {args.out}. Review them, add reference_answer and relevant_doc_ids, "
          f"and only then move the chosen ones to golden.jsonl.")


def cmd_retrieval(args):
    import ragapp.retriever as R
    from evalkit.rag_metrics import mrr, ndcg_at_k, recall_at_k

    cases = [c for c in load_cases(args.cases) if c.relevant_doc_ids]

    def evaluate(k: int) -> list[dict]:
        rows = []
        for c in cases:
            docs = list(dict.fromkeys(h.chunk.doc_id for h in R.search(c.question, c.workspace, k)))
            rows.append({"id": c.id, "recall": recall_at_k(docs, c.relevant_doc_ids, k),
                         "mrr": mrr(docs, c.relevant_doc_ids), "ndcg": ndcg_at_k(docs, c.relevant_doc_ids, k),
                         "docs": docs, "relevant": c.relevant_doc_ids, "tags": c.tags})
        return rows

    if args.sweep:
        print("stem  k   recall@k   MRR    nDCG@k")
        for stem in (3, 4, 5, 6, 7):
            R.configure(stem)
            for k in (2, 4, 6):
                rows = evaluate(k)
                print(f"{stem:4d} {k:2d}   {fmean(r['recall'] for r in rows):.2f}     "
                      f"{fmean(r['mrr'] for r in rows):.2f}   {fmean(r['ndcg'] for r in rows):.2f}")
        return
    if args.stem_len:
        R.configure(args.stem_len)
    rows = evaluate(args.k)
    for r in rows:
        mark = "  " if r["recall"] == 1 else "✗ "
        print(f"{mark}{r['id']} recall={r['recall']:.2f} mrr={r['mrr']:.2f}  {r['docs']}  (expected {r['relevant']})")
    print(f"\nmean: recall@{args.k}={fmean(r['recall'] for r in rows):.2f}  MRR={fmean(r['mrr'] for r in rows):.2f}"
          f"  nDCG@{args.k}={fmean(r['ndcg'] for r in rows):.2f}  (n={len(rows)}, stem={R.STEM_LEN})")


def cmd_agent(args):
    from evalkit.agent_metrics import pass_at_k, pass_hat_k, tool_calls_match
    from evalkit.checks import contains_any
    from ragapp.agent import run_agent

    tasks = read_jsonl(args.tasks)
    if args.ids:
        tasks = [t for t in tasks if t["id"] in args.ids.split(",")]
    log, table = [], []
    for t in tasks:
        successes = 0
        for rep in range(args.reps):
            r = run_agent(t["task"], rep=rep)
            actual = [{"tool": s["tool"], "args": s["args"]} for s in r.trajectory]
            calls = tool_calls_match(t.get("expected_tools", []), actual)
            forbidden = [s["tool"] for s in r.trajectory if s["tool"] in t.get("forbidden_tools", [])]
            expected_tickets = t.get("expected_tickets", [])
            tickets_ok = len(r.tickets) == len(expected_tickets) and all(
                all(str(v).lower() in str(tk.get(key, "")).lower() for key, v in exp.items())
                for exp, tk in zip(expected_tickets, r.tickets))
            answer_ok = contains_any(r.final_text, t["answer_must_include_any"]).passed if t.get(
                "answer_must_include_any") else True
            ok = r.status == "ok" and tickets_ok and not forbidden and answer_ok
            successes += ok
            log.append({"task_id": t["id"], "rep": rep, "success": ok, "tickets_ok": tickets_ok,
                        "forbidden_calls": forbidden, "answer_ok": answer_ok, "tool_recall": calls["recall"],
                        "steps": r.steps, "result": r.to_dict()})
        n = args.reps
        table.append((t["id"], successes, n, pass_at_k(n, successes, 1), pass_hat_k(n, successes, n)))
    out = config.RUNS_DIR / f"agent-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, log)
    print(f"task     successes  pass@1  pass^{args.reps}")
    for tid, c, n, p1, pk in table:
        print(f"{tid:7s}  {c}/{n}      {p1:.2f}    {pk:.2f}")
    print(f"\nmean pass@1={fmean(x[3] for x in table):.2f}, pass^{args.reps}={fmean(x[4] for x in table):.2f}"
          f"\ntrajectories: {out}")


def cmd_leakcheck(args):
    import ragapp.retriever as R

    questions = [c for c in load_cases(GOLDEN)] + [c for c in load_cases(config.DATA_DIR / "redteam.jsonl")]
    leaks = []
    for ws in R.workspaces():
        for c in questions:
            for h in R.search(c.question, ws, 4, enforce_tenant=not args.unsafe):
                if h.chunk.workspace != ws:
                    leaks.append((ws, c.id, h.chunk.id))
    mode = "WITHOUT tenant filter (simulated bug)" if args.unsafe else "with tenant filter"
    print(f"{mode}: {len(leaks)} chunks from another workspace")
    for ws, cid, chunk in leaks[:15]:
        print(f"  workspace={ws} question={cid} → {chunk}")
    sys.exit(1 if leaks and not args.unsafe else 0)


def cmd_chat(args):
    from evalkit.checks import contains_any
    from evalkit.simulator import conversation_metrics, judge_conversation, simulate
    from ragapp.chat import ChatSession

    data = yaml.safe_load(Path(args.file).read_text(encoding="utf-8"))
    print(f"== scripted conversations (question rewriting: {'on' if not args.no_rewrite else 'off'})")
    ok_count = 0
    for conv in data["scripted"]:
        session = ChatSession(rewrite=not args.no_rewrite)
        for msg in conv["turns"]:
            turn = session.send(msg)
        ok = contains_any(turn.result.answer, conv["final_must_include_any"]).passed
        ok_count += ok
        print(f"{'✓' if ok else '✗'} {conv['id']}: last question after rewriting: «{turn.standalone_question}»")
    print(f"passed: {ok_count}/{len(data['scripted'])}")

    if args.skip_sim:
        return
    print(f"\n== user simulator (repetitions: {args.reps})")
    for sc in data["scenarios"]:
        for rep in range(args.reps):
            transcript = simulate(sc, lambda: ChatSession(rewrite=not args.no_rewrite), max_turns=args.max_turns, rep=rep)
            m = conversation_metrics(transcript, sc.get("success_keywords", []))
            j = judge_conversation(sc, transcript, rep)
            print(f"{sc['id']} #{rep}: turns={m['turns']} goal(keywords)={m['goal_reached']} "
                  f"goal(judge)={j['goal_achieved']} contradiction={j['contradiction']}")
            if args.verbose:
                for t in transcript:
                    print(f"   U: {t['user']}\n   A: {t['assistant'][:160]}")


def cmd_redteam(args):
    cases = _select(load_cases(args.cases), args)
    target = app_target(args.prompt, 4, None, enforce_tenant=not args.unsafe_tenant)
    run_eval(cases, target, _graders("security,checks-nocite"), label=f"redteam-{args.prompt}")


def cmd_optimize(args):
    from evalkit.report import summarize_rows

    cases = load_cases(args.cases)
    train, test = stratified_split(cases, key=lambda c: c.primary_tag, test_frac=0.4, seed=args.seed)
    print(f"train={len(train)} test={len(test)} (random split, stratified by first tag, seed={args.seed})\n")
    from evalkit.dataset import find_leaked_cases
    from ragapp.app import load_prompt

    for version in args.candidates.split(","):
        leaked = find_leaked_cases(load_prompt(version), cases)
        if leaked:
            print(f"WARNING: prompt {version} contains questions from the eval set: {leaked} (label leakage)")
    print("\nprompt            train    test    difference")
    for version in args.candidates.split(","):
        rates = []
        for split_name, split in (("train", train), ("test", test)):
            out = run_eval(split, app_target(version), _graders(args.graders), label=f"opt-{version}-{split_name}",
                           quiet=True)
            rates.append(summarize_rows(load_run(out)["rows"])["pass_rate"] or 0)
        print(f"{version:16s} {rates[0]:6.1%}  {rates[1]:6.1%}   {rates[0] - rates[1]:+.1%}")


def cmd_models(args):
    from evalkit.pareto import pareto_front
    import os

    cases = _select(load_cases(args.cases), args)
    points = []
    for model in args.models.split(","):
        for effort in args.efforts.split(","):
            os.environ["APP_EFFORT"] = effort
            out = run_eval(cases, app_target(args.prompt, model=model), _graders(args.graders),
                           label=f"models-{model}-{effort}", quiet=True)
            s = load_run(out)["summary"]
            points.append({"config": f"{model} / {effort}", "quality": s["pass_rate"] or 0,
                           "cost": s["cost_usd_per_row"] or 0.0, "latency": s["latency_p95_ms"] or 0.0})
    front = pareto_front(points)
    print("configuration                        quality  $/case     p95 ms   front")
    for p in sorted(points, key=lambda p: -p["quality"]):
        print(f"{p['config']:35s} {p['quality']:6.1%}  {p['cost']:9.5f}  {p['latency']:7.0f}   "
              f"{'★' if p in front else ''}")


def cmd_monitor(args):
    from evalkit.monitor import daily_metrics, psi, sample_for_review

    traces = read_jsonl(args.traces)
    days = sorted({t["day"] for t in traces})
    print("day          n   refusals thumb↓  mean score p95 ms")
    for d in daily_metrics(traces):
        td = f"{d['thumbs_down_rate']:.0%}" if d["thumbs_down_rate"] is not None else "-"
        print(f"{d['day']}  {d['n']:3d}   {d['refusal_rate']:5.0%}  {td:>6}   {d['mean_top_score']:6.2f}  "
              f"{d['p95_latency_ms']:6.0f}")
    half = len(days) // 2
    from collections import Counter
    before = Counter(t["topic"] for t in traces if t["day"] in days[:half])
    after = Counter(t["topic"] for t in traces if t["day"] in days[half:])
    value = psi(before, after)
    level = "stable" if value < 0.1 else "watch" if value < 0.25 else "significant shift"
    print(f"\ntopic PSI (1st half vs 2nd half): {value:.3f} → {level}")
    print("largest share changes: " + ", ".join(
        f"{t}: {before[t] / sum(before.values()):.0%}→{after[t] / sum(after.values()):.0%}"
        for t in sorted(set(before) | set(after),
                        key=lambda t: -abs(after[t] / sum(after.values()) - before[t] / sum(before.values())))[:3]))
    queue = sample_for_review(traces, args.budget, seed=0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, queue)
    from collections import Counter as C
    print(f"\nreview queue: {len(queue)} traces → {out}  {dict(C(q['review_reason'] for q in queue))}")


def cmd_sim(args):
    from evalkit.stats import noise_floor

    rng = random.Random(args.seed)
    z = 1.96

    def trial(n: int, reps: int, effect: float) -> tuple[float, bool]:
        diffs = []
        for _ in range(n):
            p_a = min(0.98, max(0.02, rng.betavariate(2.3, 1.0)))  # per-case difficulty, mean ≈ 0.70
            p_b = min(1.0, max(0.0, p_a + effect))
            a = sum(rng.random() < p_a for _ in range(reps)) / reps
            b = sum(rng.random() < p_b for _ in range(reps)) / reps
            diffs.append(b - a)
        m = fmean(diffs)
        sd = (sum((d - m) ** 2 for d in diffs) / (n - 1)) ** 0.5
        return m, m - z * sd / n**0.5 > 0

    print(f"True improvement of B over A: {args.effect:+.0%}. {args.trials} simulated comparisons per row.\n")
    print("    cases  reps  noise floor 'B looks worse'     improvement found  false alarm (effect 0)")
    for n in (20, 50, 100, 200, 500):
        for reps in (1, 3):
            res = [trial(n, reps, args.effect) for _ in range(args.trials)]
            null = [trial(n, reps, 0.0) for _ in range(args.trials)]
            print(f"{n:9d}  {reps:4d}   ±{noise_floor(n, reps):6.1%}      {fmean(m < 0 for m, _ in res):7.0%}"
                  f"             {fmean(s for _, s in res):6.0%}          {fmean(s for _, s in null):6.0%}")


def cmd_online(args):
    from evalkit.online import evaluate_online
    from evalkit.tracing import setup_tracing
    from ragapp.app import answer

    provider = setup_tracing(exporter=args.exporter, jsonl_path=args.jsonl)
    rng = random.Random(args.seed)
    cases = load_cases(args.cases)[: args.limit]
    judged = []
    for c in cases:
        result = answer(c.question, c.workspace)
        r = evaluate_online(result, sample_rate=args.sample_rate, rng=rng)
        if r:
            judged.append(r["score"])
    provider.shutdown()
    print(f"\nserved {len(cases)} questions, judged online {len(judged)}"
          + (f", mean faithfulness {fmean(judged):.2f}" if judged else ""))
    if args.exporter == "jsonl":
        print(f"spans saved to {args.jsonl}")


def main() -> None:
    p = argparse.ArgumentParser(prog="evalkit", description="AI evals practice toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_, description=help_)
        sp.set_defaults(fn=fn)
        return sp

    def common(sp, cases=True):
        if cases:
            sp.add_argument("--cases", default=str(GOLDEN))
        sp.add_argument("--tags", help="only cases with these tags (comma-separated)")
        sp.add_argument("--ids", help="only these ids (comma-separated)")
        sp.add_argument("--limit", type=int)

    sp = add("ask", cmd_ask, "Ask the app one question")
    sp.add_argument("question")
    sp.add_argument("--workspace", default="nordwind")
    sp.add_argument("--prompt", default="v2")
    sp.add_argument("--k", type=int, default=4)
    sp.add_argument("--json", action="store_true")

    sp = add("run", cmd_run, "Run an eval of the app on a case set")
    common(sp)
    sp.add_argument("--prompt", default="v2")
    sp.add_argument("--k", type=int, default=4)
    sp.add_argument("--model")
    sp.add_argument("--reps", type=int, default=1)
    sp.add_argument("--graders", default="checks,retrieval,security")
    sp.add_argument("--label")
    sp.add_argument("--out", help="output directory (default runs/<label>-<time>)")
    sp.add_argument("--workers", type=int, default=4)
    sp.add_argument("--unsafe-tenant", action="store_true", help="simulate a missing tenant filter in the retriever")

    sp = add("grade", cmd_grade, "Grade previously saved answers (without calling the app)")
    common(sp)
    sp.add_argument("--outputs", default=str(SAMPLE / "outputs.jsonl"))
    sp.add_argument("--graders", default="checks")
    sp.add_argument("--label", default="graded")
    sp.add_argument("--workers", type=int, default=4)

    sp = add("render", cmd_render, "Turn a run into markdown for manual review (error analysis)")
    sp.add_argument("run")
    sp.add_argument("--cases", default=str(GOLDEN))
    sp.add_argument("--out")

    sp = add("compare", cmd_compare, "Compare two runs (paired, with a confidence interval)")
    sp.add_argument("run_a")
    sp.add_argument("run_b")

    sp = add("sanity", cmd_sanity, "Oracle and null baselines through the graders")
    common(sp)
    sp.add_argument("--graders", default="checks")

    sp = add("align", cmd_align, "Agreement of the LLM judge with human labels")
    sp.add_argument("--cases", default=str(GOLDEN))
    sp.add_argument("--outputs", default=str(SAMPLE / "outputs.jsonl"))
    sp.add_argument("--labels", default=str(SAMPLE / "human_labels.jsonl"))
    sp.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    sp.add_argument("--prompt", help="path to your own judge prompt")

    sp = add("bias", cmd_bias, "Experiment: position and length bias of the judge")
    sp.add_argument("--cases", default=str(GOLDEN))
    sp.add_argument("--limit", type=int, default=10)

    sp = add("audit", cmd_audit, "Audit the case set (duplicates, balance, gaps)")
    sp.add_argument("cases", nargs="?", default=str(GOLDEN))

    sp = add("synth", cmd_synth, "Generate case candidates from dimensions (persona × topic × difficulty)")
    sp.add_argument("--dimensions", default=str(config.DATA_DIR / "synth_dimensions.yaml"))
    sp.add_argument("--n", type=int, default=12)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--out", default=str(config.DATA_DIR / "synthetic_candidates.jsonl"))

    sp = add("retrieval", cmd_retrieval, "Retriever metrics (offline, no LLM)")
    sp.add_argument("--cases", default=str(GOLDEN))
    sp.add_argument("--k", type=int, default=4)
    sp.add_argument("--stem-len", type=int)
    sp.add_argument("--sweep", action="store_true", help="sweep stem × k")

    sp = add("agent", cmd_agent, "Eval of the tool-using agent: final state, trajectory, pass^k")
    sp.add_argument("--tasks", default=str(config.DATA_DIR / "agent_tasks.jsonl"))
    sp.add_argument("--reps", type=int, default=3)
    sp.add_argument("--ids")

    sp = add("leakcheck", cmd_leakcheck, "Tenant isolation test at retriever level (offline)")
    sp.add_argument("--unsafe", action="store_true")

    sp = add("redteam", cmd_redteam, "Adversarial cases with the security grader")
    common(sp, cases=False)
    sp.add_argument("--cases", default=str(config.DATA_DIR / "redteam.jsonl"))
    sp.add_argument("--prompt", default="v2")
    sp.add_argument("--unsafe-tenant", action="store_true")

    sp = add("chat", cmd_chat, "Multi-turn evals: scripted conversations and a user simulator")
    sp.add_argument("--file", default=str(config.DATA_DIR / "conversations.yaml"))
    sp.add_argument("--no-rewrite", action="store_true")
    sp.add_argument("--skip-sim", action="store_true")
    sp.add_argument("--reps", type=int, default=2)
    sp.add_argument("--max-turns", type=int, default=5)
    sp.add_argument("--verbose", action="store_true")

    sp = add("optimize", cmd_optimize, "Compare prompt versions on train/test")
    sp.add_argument("--cases", default=str(GOLDEN))
    sp.add_argument("--candidates", default="v1,v2,v3,v3_leaky")
    sp.add_argument("--graders", default="checks")
    sp.add_argument("--seed", type=int, default=7)

    sp = add("models", cmd_models, "Quality vs cost vs latency for models and effort levels")
    common(sp)
    sp.add_argument("--models", default=config.app_model())
    sp.add_argument("--efforts", default="low,medium")
    sp.add_argument("--prompt", default="v2")
    sp.add_argument("--graders", default="checks")

    sp = add("monitor", cmd_monitor, "Analyse production traces: daily metrics, drift, review queue")
    sp.add_argument("--traces", default=str(config.DATA_DIR / "prod_traces.jsonl"))
    sp.add_argument("--budget", type=int, default=20)
    sp.add_argument("--out", default="notes/review_queue.jsonl")

    sp = add("sim", cmd_sim, "Simulation: how many cases you need to detect an improvement")
    sp.add_argument("--effect", type=float, default=0.05)
    sp.add_argument("--trials", type=int, default=300)
    sp.add_argument("--seed", type=int, default=0)

    sp = add("online", cmd_online, "Online evals demo with OpenTelemetry")
    sp.add_argument("--cases", default=str(GOLDEN))
    sp.add_argument("--limit", type=int, default=10)
    sp.add_argument("--sample-rate", type=float, default=0.5)
    sp.add_argument("--exporter", choices=["console", "jsonl", "otlp"], default="jsonl")
    sp.add_argument("--jsonl", default="runs/traces.jsonl")
    sp.add_argument("--seed", type=int, default=0)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
