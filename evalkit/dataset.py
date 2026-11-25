import itertools
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Callable

import yaml
from pydantic import BaseModel, ConfigDict


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    workspace: str = "nordwind"
    reference_answer: str | None = None
    relevant_doc_ids: list[str] = []
    must_include_any: list[str] = []
    must_include_all: list[str] = []
    must_not_include: list[str] = []
    expect_refusal: bool = False
    max_words: int | None = None
    tags: list[str] = []
    source: str = "handwritten"
    notes: str | None = None

    @property
    def primary_tag(self) -> str:
        return self.tags[0] if self.tags else "untagged"


def load_cases(path: str | Path) -> list[Case]:
    path = Path(path)
    if path.suffix in (".yaml", ".yml"):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        items = raw["cases"] if isinstance(raw, dict) else raw
    else:
        items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [Case(**item) for item in items]


def save_cases(cases: list[Case], path: str | Path) -> None:
    lines = [json.dumps(c.model_dump(exclude_defaults=True), ensure_ascii=False) for c in cases]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def near_duplicates(items: list[tuple[str, str]], threshold: float = 0.8) -> list[tuple[str, str, float]]:
    """Pairs (id_a, id_b, similarity) whose word-set Jaccard >= threshold."""
    words = [(i, _words(text)) for i, text in items]
    pairs = []
    for (ia, wa), (ib, wb) in itertools.combinations(words, 2):
        sim = jaccard(wa, wb)
        if sim >= threshold:
            pairs.append((ia, ib, round(sim, 3)))
    return pairs


def sample_tuples(dimensions: dict[str, list[str]], n: int, seed: int = 0) -> list[dict[str, str]]:
    """n combinations of dimension values; covers every single value at least once when n allows."""
    keys = list(dimensions)
    combos = [dict(zip(keys, values)) for values in itertools.product(*(dimensions[k] for k in keys))]
    rng = random.Random(seed)
    rng.shuffle(combos)
    if n >= len(combos):
        return combos

    uncovered = {(k, v) for k in keys for v in dimensions[k]}
    chosen: list[dict[str, str]] = []
    while uncovered and len(chosen) < n:
        best = max(combos, key=lambda c: len(set(c.items()) & uncovered))  # ties: first in shuffled order
        chosen.append(best)
        uncovered -= set(best.items())
    for combo in combos:
        if len(chosen) == n:
            break
        if combo not in chosen:
            chosen.append(combo)
    return chosen


def stratified_split(items: list, key: Callable, test_frac: float = 0.5, seed: int = 0) -> tuple[list, list]:
    """Random split stratified by key(item); never split by score (regression to the mean)."""
    groups: dict = {}
    for item in items:
        groups.setdefault(key(item), []).append(item)
    rng = random.Random(seed)
    train, test = [], []
    for k in sorted(groups, key=str):
        group = groups[k][:]
        rng.shuffle(group)
        n_test = round(len(group) * test_frac)
        test.extend(group[:n_test])
        train.extend(group[n_test:])
    return train, test


def find_leaked_cases(text: str, cases: list[Case], threshold: float = 0.8) -> list[str]:
    """Ids of cases whose question appears (verbatim or nearly) in a prompt or few-shot block."""
    lines = [_words(line) for line in text.splitlines() if line.strip()]
    leaked = []
    for c in cases:
        q = _words(c.question)
        if any(q <= line or jaccard(q, line) >= threshold for line in lines):
            leaked.append(c.id)
    return leaked


def audit_cases(cases: list[Case]) -> dict:
    ids = Counter(c.id for c in cases)
    questions = Counter(f"{c.workspace}: {c.question.strip().lower()}" for c in cases)
    return {
        "n": len(cases),
        "duplicate_ids": [i for i, n in ids.items() if n > 1],
        "duplicate_questions": [q for q, n in questions.items() if n > 1],
        "near_duplicates": near_duplicates([(c.id, f"{c.workspace} {c.question}") for c in cases], 0.75),
        "primary_tags": dict(Counter(c.primary_tag for c in cases).most_common()),
        "expect_refusal": sum(c.expect_refusal for c in cases),
        "answerable": sum(not c.expect_refusal for c in cases),
        "workspaces": dict(Counter(c.workspace for c in cases)),
        "sources": dict(Counter(c.source for c in cases)),
        "missing_reference": [c.id for c in cases if not c.reference_answer],
        "no_deterministic_check": [
            c.id for c in cases
            if not (c.must_include_any or c.must_include_all or c.must_not_include or c.expect_refusal)
        ],
        "question_words": {
            "min": min((len(c.question.split()) for c in cases), default=0),
            "max": max((len(c.question.split()) for c in cases), default=0),
        },
    }
