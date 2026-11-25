"""Nordwind Assist: the RAG system under test. Entry point: answer()."""

import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from opentelemetry import trace

from evalkit import config
from evalkit.llm import LLMResult, complete
from evalkit.pricing import cost_usd
from ragapp.retriever import Hit, search, tokenize

PROMPTS_DIR = Path(__file__).parent / "prompts"
CITATION_RE = re.compile(r"\[([a-z0-9][a-z0-9\-]*)\]")
FAKE_MIN_SCORE = 1.0
tracer = trace.get_tracer("ragapp")


@dataclass
class AppResult:
    question: str
    workspace: str
    answer: str
    citations: list[str]
    retrieved: list[dict]
    prompt_version: str
    model: str
    status: str = "ok"
    stop_reason: str | None = None
    usage: dict = field(default_factory=dict)
    cost_usd: float | None = None
    latency_ms: float = 0.0
    cached: bool = False
    trace_id: str | None = None
    span_id: str | None = None

    @property
    def retrieved_doc_ids(self) -> list[str]:
        return list(dict.fromkeys(r["doc_id"] for r in self.retrieved))

    def to_dict(self) -> dict:
        data = asdict(self)
        data["retrieved_doc_ids"] = self.retrieved_doc_ids
        return data


def load_prompt(version: str) -> str:
    return (PROMPTS_DIR / f"answer_{version}.txt").read_text(encoding="utf-8")


def build_user_message(question: str, hits: list[Hit]) -> str:
    docs = "\n".join(
        f'<document id="{h.chunk.doc_id}" version="{h.chunk.version}" status="{h.chunk.status}" '
        f'valid_from="{h.chunk.valid_from}">\n{h.chunk.text}\n</document>'
        for h in hits
    )
    return f"<documents>\n{docs}\n</documents>\n\n<question>{question}</question>"


def extract_citations(text: str) -> list[str]:
    return list(dict.fromkeys(CITATION_RE.findall(text)))


def _fake_answer(question: str, hits: list[Hit], prompt_version: str) -> str:
    # Extractive baseline for offline runs: best-overlapping sentence of the top chunk.
    if not hits or hits[0].score < FAKE_MIN_SCORE:
        return "I don't know — it is not in the documents. Ask the HR department (hr@nordwind.example)."
    q = set(tokenize(question))
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n", hits[0].chunk.text)[1:] if s.strip()]
    best = max(sentences, key=lambda s: len(q & set(tokenize(s))), default=hits[0].chunk.text)
    return best.strip() if prompt_version == "v1" else f"{best.strip()} [{hits[0].chunk.doc_id}]"


def answer(
    question: str,
    workspace: str = "nordwind",
    *,
    prompt_version: str = "v2",
    k: int = 4,
    model: str | None = None,
    enforce_tenant: bool = True,
    rep: int = 0,
) -> AppResult:
    model = model or config.app_model()
    with tracer.start_as_current_span("ragapp.answer") as span:
        started = time.perf_counter()
        hits = search(question, workspace, k, enforce_tenant=enforce_tenant)
        llm: LLMResult = complete(
            model=model,
            system=load_prompt(prompt_version),
            messages=[{"role": "user", "content": build_user_message(question, hits)}],
            max_tokens=2048,
            effort=config.app_effort(),
            rep=rep,
            fake=lambda: _fake_answer(question, hits, prompt_version),
        )
        status = {"max_tokens": "truncated", "refusal": "refusal"}.get(llm.stop_reason or "", "ok")
        result = AppResult(
            question=question,
            workspace=workspace,
            answer=llm.text,
            citations=extract_citations(llm.text),
            retrieved=[{"id": h.chunk.id, "doc_id": h.chunk.doc_id, "workspace": h.chunk.workspace,
                        "score": h.score, "rank": h.rank} for h in hits],
            prompt_version=prompt_version,
            model=llm.model,
            status=status,
            stop_reason=llm.stop_reason,
            usage=llm.usage,
            cost_usd=cost_usd(llm.model, llm.usage),
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            cached=llm.cached,
        )
        _annotate_span(span, result, k)
        ctx = span.get_span_context()
        if ctx.is_valid:
            result.trace_id, result.span_id = format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")
        return result


def _annotate_span(span, r: AppResult, k: int) -> None:
    # One wide event per answer: every attribute you may want to slice by later.
    span.set_attributes({
        "ragapp.workspace": r.workspace,
        "ragapp.prompt_version": r.prompt_version,
        "ragapp.retrieval.k": k,
        "ragapp.retrieval.doc_ids": r.retrieved_doc_ids,
        "ragapp.retrieval.top_score": r.retrieved[0]["score"] if r.retrieved else 0.0,
        "ragapp.answer.citations": r.citations,
        "ragapp.answer.refused": r.answer.strip().lower().startswith("i don't know"),
        "ragapp.answer.words": len(r.answer.split()),
        "ragapp.status": r.status,
        "ragapp.cache_hit": r.cached,
        "gen_ai.request.model": r.model,
        "gen_ai.usage.input_tokens": r.usage.get("input_tokens", 0),
        "gen_ai.usage.output_tokens": r.usage.get("output_tokens", 0),
    })
    if r.cost_usd is not None:
        span.set_attribute("ragapp.cost_usd", r.cost_usd)
    if os.environ.get("RAGAPP_TRACE_CONTENT") == "1":
        span.set_attribute("ragapp.question", r.question)
        span.set_attribute("ragapp.answer.text", r.answer)
