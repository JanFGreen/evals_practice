"""Deterministic checks (module 1). Cheap, fast, run on every output before any LLM judge."""

import json
import re
import unicodedata
from dataclasses import dataclass

CITATION_RE = re.compile(r"\[([a-z0-9][a-z0-9\-]*)\]")
REFUSAL_MARKERS = (
    "i don't know", "i do not know", "it is not in the documents", "not in the documents", "i couldn't find",
    "i could not find", "no information", "i don't have information", "i don't have access", "i do not have access",
    "the documents do not contain", "do not contain information", "i can't provide", "i cannot provide",
    "i can't share", "i cannot share",
)
BOILERPLATE = (
    "as a language model", "as an ai", "as an artificial intelligence", "i hope this helps", "great question",
)


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def normalize(text: str) -> str:
    """Casefold, NFKC (NBSP -> space), curly apostrophe -> ', collapse whitespace, glue numbers to units: '23 kg' -> '23kg'."""
    text = unicodedata.normalize("NFKC", text).replace("\u2019", "'").casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"(\d) (?=[^\W\d_])", r"\1", text)


def _find(text_norm: str, needle: str) -> bool:
    n = normalize(needle)
    # A needle starting with a digit must not be glued to a preceding digit: "23" must not match "123".
    prefix = r"(?<!\d)" if n[:1].isdigit() else ""
    return re.search(prefix + re.escape(n), text_norm) is not None


def contains_any(text: str, needles: list[str]) -> CheckResult:
    t = normalize(text)
    found = [n for n in needles if _find(t, n)]
    return CheckResult("contains_any", bool(found), f"found: {found}" if found else f"none of: {needles}")


def contains_all(text: str, needles: list[str]) -> CheckResult:
    t = normalize(text)
    missing = [n for n in needles if not _find(t, n)]
    return CheckResult("contains_all", not missing, f"missing: {missing}" if missing else "")


def not_contains(text: str, needles: list[str]) -> CheckResult:
    t = normalize(text)
    found = [n for n in needles if _find(t, n)]
    return CheckResult("not_contains", not found, f"forbidden: {found}" if found else "")


def max_words(text: str, limit: int) -> CheckResult:
    n = len(text.split())
    return CheckResult("max_words", n <= limit, f"{n} words (limit {limit})")


def extract_citations(text: str) -> list[str]:
    return list(dict.fromkeys(CITATION_RE.findall(text)))


def has_citation(text: str) -> CheckResult:
    cites = extract_citations(text)
    return CheckResult("has_citation", bool(cites), ", ".join(cites) or "no citations")


def citations_grounded(citations: list[str], retrieved_doc_ids: list[str]) -> CheckResult:
    invented = [c for c in citations if c not in set(retrieved_doc_ids)]
    return CheckResult("citations_grounded", not invented,
                       f"cites documents outside the context: {invented}" if invented else "")


def is_refusal(text: str) -> bool:
    t = normalize(text)
    return any(normalize(m) in t for m in REFUSAL_MARKERS)


def refusal_matches(text: str, expect_refusal: bool) -> CheckResult:
    refused = is_refusal(text)
    ok = refused == expect_refusal
    if ok:
        detail = ""
    elif expect_refusal:
        detail = "should refuse / say it does not know, but answered"
    else:
        detail = "refused, although the answer is in the documents"
    return CheckResult("refusal", ok, detail)


def no_boilerplate(text: str) -> CheckResult:
    t = normalize(text)
    found = [b for b in BOILERPLATE if b in t]
    return CheckResult("no_boilerplate", not found, f"boilerplate: {found}" if found else "")


def valid_json(text: str, required_keys: tuple[str, ...] = ()) -> CheckResult:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return CheckResult("valid_json", False, f"invalid JSON: {e}")
    if not isinstance(data, dict):
        return CheckResult("valid_json", False, "JSON is not an object")
    missing = [k for k in required_keys if k not in data]
    return CheckResult("valid_json", not missing, f"missing keys: {missing}" if missing else "")


def run_checks(case, output: dict, *, require_citations: bool = True, default_max_words: int = 120) -> list[CheckResult]:
    answer = output.get("answer", "")
    results: list[CheckResult] = []
    if case.must_include_any:
        results.append(contains_any(answer, case.must_include_any))
    if case.must_include_all:
        results.append(contains_all(answer, case.must_include_all))
    if case.must_not_include:
        results.append(not_contains(answer, case.must_not_include))
    results.append(refusal_matches(answer, case.expect_refusal))
    results.append(max_words(answer, case.max_words or default_max_words))
    results.append(no_boilerplate(answer))
    if require_citations and not case.expect_refusal:
        results.append(has_citation(answer))
    results.append(citations_grounded(extract_citations(answer), output.get("retrieved_doc_ids", [])))
    return results
