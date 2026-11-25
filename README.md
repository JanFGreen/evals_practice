# evals_practice

My hands-on practice in evaluating LLM applications: RAG, a tool-using agent, LLM-as-judge, statistics, security, evals in CI and in production. I work through one exercise at a time and commit every day; `LOG.md` and the commit history show the progress.

Everything runs **offline on my machine**. The default model mode is `fake` (no network, no API key), so nothing is sent anywhere.

## Quick start

```bash
uv sync
git config core.hooksPath .githooks              # enable the leak / English-only guard (once per clone)
uv run pytest tests/test_m01_checks.py           # the first exercise: red at the start
uv run python -m ragapp "How many kilograms can I lift on my own?"
uv run python scripts/progress.py progress.xml   # after: uv run pytest --junitxml=progress.xml
```

## Daily loop

1. Pick the next exercise in `exercises/` (see the module map below).
2. Make its tests green: `uv run pytest tests/test_<module>.py`.
3. Use the code on the data (the commands in the module map).
4. Add one line to `LOG.md`: date, module, what I did, tests passing.
5. Commit and push, e.g. `git commit -m "m01: implement normalize and contains_any (5/10 green)"`.

## The system under test: Nordwind Assist

A small RAG assistant that answers questions about the internal policies of a fictional logistics company (about 150 lines): an offline BM25 retriever, one model call and a tool-using agent. The documents are written from scratch, so every question has a known correct answer. The corpus contains deliberate traps:

| Trap | What it tests |
|---|---|
| Two versions of the lifting rules: withdrawn (25 kg) and current (23 kg) | picking the current document; faithfulness vs correctness |
| A supplier note with a hidden "ignore previous instructions" | indirect prompt injection through a document |
| A second tenant (Orbita) with a confidential salary sheet and a canary marker | data isolation between customers |
| Questions about a "near miss", "cash for internet", "hauling" | synonyms the BM25 retriever does not connect |
| Questions outside the documents (bonus, door code) | admitting "I don't know" |

## Model modes (`EVAL_LLM_MODE`)

| mode | what it does | when |
|---|---|---|
| `fake` (default) | stand-in without any API: extractive answer, heuristic judge | everything in this repo |
| `cache` | calls the API and stores the answer in `.cache/llm` | only with your own key, never needed here |
| `replay` | disk only; a missing entry is an error | deterministic CI with a recorded cache |
| `live` | always calls the API | measuring latency and variance |

Results in `fake` mode tell you whether the mechanics work, not how good a real model is.

## Layout

| path | contents |
|---|---|
| `ragapp/` | the system under test: RAG app, agent, multi-turn chat, minimal web UI |
| `ragapp/corpus/` | documents of two companies (tenants) with the traps |
| `evalkit/` | the eval harness (runner, graders, judge, stats, gate, CLI) |
| `exercises/` | my exercises: empty functions to implement |
| `tests/` | the exercise tests, i.e. the executable specification |
| `data/` | golden set, sample answers with expert labels, red-team cases, agent tasks, conversations, production traces |
| `promptfoo/` | the same eval in promptfoo |
| `playwright/` | E2E tests of the chat UI |
| `evals/` | gate policies and the fake baseline for CI |
| `templates/` | eval card, capstone checklist |
| `notes/` | my private notes (git-ignored) |

## Module map

| module | exercise | commands |
|---|---|---|
| 1 Why assertEqual fails | `exercises/m01_checks.py` | `evalkit.cli grade --graders checks-nocite` |
| 2 The eval loop by hand | `exercises/m02_report.py`, error analysis | `evalkit.cli render`, `sanity`, `compare` |
| 3 promptfoo | `promptfoo/promptfooconfig.yaml` | `npx promptfoo@latest eval` |
| 4 LLM-as-judge | `exercises/m04_judge.py` | `evalkit.cli align`, `bias` |
| 5 Building the dataset | `exercises/m05_dataset.py` | `evalkit.cli audit` |
| D1 Statistics | `exercises/d1_stats.py` | `evalkit.cli sim`, `compare` |
| 6 RAG and agents | `exercises/m06_rag.py`, `exercises/m06_agent.py` | `evalkit.cli retrieval`, `agent` |
| D2 Multi-turn | `exercises/d2_multiturn.py` | `evalkit.cli chat` |
| D3 Security | `exercises/d3_security.py` | `evalkit.cli leakcheck`, `redteam` |
| 7 Regression gate | `exercises/m07_gate.py` | `evalkit.gate` |
| D4 Optimisation without overfitting | `exercises/d4_optimize.py` | `evalkit.cli optimize` |
| D5 E2E chat tests | `playwright/tests` | `npm run test:ui` |
| 8 Production monitoring | `exercises/m08_monitor.py` | `evalkit.cli monitor` |
| D6 Observability | `evalkit/tracing.py`, `evalkit/online.py` | `evalkit.cli online --exporter jsonl` |
| D7 Model choice | `exercises/d7_pareto.py` | `evalkit.cli models` |
| 9 Capstone | `templates/capstone_checklist.md` | everything above |
| 10 Limits of evals | `templates/eval_card.md` | |

Every command has `--help`: `uv run python -m evalkit.cli <command> --help`.

## Notes on the data

The sample "v1" answers and the expert labels in `data/runs/v1_sample/` are hand-written to contain realistic failure classes; they are not model outputs. The production traces in `data/prod_traces.jsonl` are synthetic. `evalkit/` also contains reference implementations of the exercises (`EVAL_TARGET=solution uv run pytest`); I only look at them after my own attempt.
