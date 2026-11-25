"""Module 1: deterministic checks. Tests: uv run pytest tests/test_m01_checks.py"""

import re
import unicodedata

from evalkit.checks import CheckResult

CITATION_RE = re.compile(r"\[([a-z0-9][a-z0-9\-]*)\]")
REFUSAL_MARKERS = (
    "i don't know", "i do not know", "it is not in the documents", "not in the documents", "i couldn't find",
    "i could not find", "no information", "i don't have information", "i don't have access", "i do not have access",
    "the documents do not contain", "do not contain information", "i can't provide", "i cannot provide",
    "i can't share", "i cannot share",
)


def normalize(text: str) -> str:
    """NFKC + casefold, collapse whitespace to one space, glue a number to its unit: curly apostrophe -> ', '23 kg' -> '23kg'."""
    raise NotImplementedError


def contains_any(text: str, needles: list[str]) -> CheckResult:
    """Pass when, after normalisation, the text contains any of the phrases. '23' must not match '123' or '2023'."""
    raise NotImplementedError


def not_contains(text: str, needles: list[str]) -> CheckResult:
    """Pass when the text contains none of the phrases (same matching rules as contains_any)."""
    raise NotImplementedError


def max_words(text: str, limit: int) -> CheckResult:
    raise NotImplementedError


def extract_citations(text: str) -> list[str]:
    """Identifiers in square brackets, without duplicates, in order of appearance."""
    raise NotImplementedError


def citations_grounded(citations: list[str], retrieved_doc_ids: list[str]) -> CheckResult:
    """Every citation must point to a document from the context; list the ones outside the context in detail."""
    raise NotImplementedError


def is_refusal(text: str) -> bool:
    raise NotImplementedError
