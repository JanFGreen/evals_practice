"""Online evals (extra module D6): judge a sample of live answers and emit the score as its own span."""

import random

from opentelemetry import trace
from opentelemetry.trace import Link, SpanContext, TraceFlags

from evalkit import config
from evalkit.judge import judge_faithfulness

tracer = trace.get_tracer("evalkit.online")


def evaluate_online(result, *, sample_rate: float = 0.3, rng: random.Random | None = None) -> dict | None:
    """Runs after the answer was served. Linked (not parented) to the answer span: in production this is async."""
    rng = rng or random.Random()
    if rng.random() >= sample_rate:
        return None
    links = []
    if result.trace_id and result.span_id:
        ctx = SpanContext(int(result.trace_id, 16), int(result.span_id, 16), is_remote=True,
                          trace_flags=TraceFlags(TraceFlags.SAMPLED))
        links.append(Link(ctx))
    with tracer.start_as_current_span("eval.faithfulness", links=links) as span:
        r = judge_faithfulness(result.answer, result.retrieved)
        span.set_attributes({
            "eval.name": "faithfulness",
            "eval.score": r["score"],
            "eval.passed": r["passed"],
            "eval.judge_model": r["model"],
            "eval.unsupported_claims": sum(c["supported"] == "no" for c in r["claims"]),
            "eval.target.trace_id": result.trace_id or "",
            "ragapp.prompt_version": result.prompt_version,
            "ragapp.workspace": result.workspace,
            "gen_ai.request.model": config.judge_model(),
        })
    return r
