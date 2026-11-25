"""Retrieval metrics (module 6). Binary relevance at document level."""

import math

from evalkit.grades import Grade


def dedupe(ids: list[str]) -> list[str]:
    return list(dict.fromkeys(ids))


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float | None:
    if not relevant:
        return None
    top = set(dedupe(retrieved)[:k])
    return len(top & set(relevant)) / len(set(relevant))


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    top = dedupe(retrieved)[:k]
    if not top:
        return 0.0
    return sum(d in set(relevant) for d in top) / k


def hit_rate_at_k(retrieved: list[str], relevant: list[str], k: int) -> float | None:
    if not relevant:
        return None
    return float(bool(set(dedupe(retrieved)[:k]) & set(relevant)))


def mrr(retrieved: list[str], relevant: list[str]) -> float:
    for rank, doc in enumerate(dedupe(retrieved), 1):
        if doc in set(relevant):
            return 1 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float | None:
    if not relevant:
        return None
    rel = set(relevant)
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(dedupe(retrieved)[:k]) if d in rel)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg


def attribute_failure(retrieval_hit: bool, answer_correct: bool) -> str:
    """Where did it break? Retrieval vs generation, the 2x2 every RAG debug session starts from."""
    if retrieval_hit and answer_correct:
        return "ok"
    if retrieval_hit and not answer_correct:
        return "generation"
    if not retrieval_hit and not answer_correct:
        return "retrieval"
    return "correct_without_context"


def retrieval_grader(k: int = 4):
    """Diagnostic metrics only (passed=None): they explain failures, they don't gate."""

    def grade(case, output: dict) -> list[Grade]:
        if not case.relevant_doc_ids:
            return []
        retrieved = output.get("retrieved_doc_ids", [])
        return [
            Grade("retrieval:recall@k", None, score=recall_at_k(retrieved, case.relevant_doc_ids, k), kind="metric"),
            Grade("retrieval:mrr", None, score=mrr(retrieved, case.relevant_doc_ids), kind="metric"),
        ]

    grade.__name__ = "retrieval"
    return grade
