"""Module 6: retriever metrics. Tests: uv run pytest tests/test_m06_rag.py"""


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float | None:
    """Remove duplicates from retrieved (keep order), take the first k; None when relevant is empty."""
    raise NotImplementedError


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Hits in the first k (after deduplication) divided by k; 0.0 for empty retrieved."""
    raise NotImplementedError


def mrr(retrieved: list[str], relevant: list[str]) -> float:
    raise NotImplementedError


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float | None:
    """Binary relevance, gain 1/log2(position+1)."""
    raise NotImplementedError


def attribute_failure(retrieval_hit: bool, answer_correct: bool) -> str:
    """'ok' | 'generation' | 'retrieval' | 'correct_without_context'."""
    raise NotImplementedError
