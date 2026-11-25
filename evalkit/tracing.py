"""OpenTelemetry setup (extra module D6). Requires: uv sync --group otel."""

import json
from pathlib import Path
from typing import Sequence


def setup_tracing(service_name: str = "ragapp", exporter: str = "console", jsonl_path: str | Path | None = None):
    """exporter: console | jsonl (local file, no backend needed) | otlp (reads OTEL_EXPORTER_OTLP_* env vars)."""
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if exporter == "console":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    elif exporter == "jsonl":
        provider.add_span_processor(SimpleSpanProcessor(JsonlSpanExporter(jsonl_path or "runs/traces.jsonl")))
    elif exporter == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    else:
        raise ValueError(f"unknown exporter: {exporter}")
    trace.set_tracer_provider(provider)
    return provider


def _make_jsonl_exporter():
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

    class _JsonlSpanExporter(SpanExporter):
        def __init__(self, path: str | Path):
            self.path = Path(path)
            self.path.parent.mkdir(parents=True, exist_ok=True)

        def export(self, spans: Sequence) -> "SpanExportResult":
            with self.path.open("a", encoding="utf-8") as f:
                for s in spans:
                    ctx = s.get_span_context()
                    f.write(json.dumps({
                        "name": s.name,
                        "trace_id": format(ctx.trace_id, "032x"),
                        "span_id": format(ctx.span_id, "016x"),
                        "parent_span_id": format(s.parent.span_id, "016x") if s.parent else None,
                        "start_ns": s.start_time,
                        "duration_ms": (s.end_time - s.start_time) / 1e6 if s.end_time else None,
                        "status": s.status.status_code.name,
                        "attributes": {k: (list(v) if isinstance(v, tuple) else v) for k, v in s.attributes.items()},
                        "links": [format(link.context.span_id, "016x") for link in s.links],
                    }, ensure_ascii=False) + "\n")
            return SpanExportResult.SUCCESS

    return _JsonlSpanExporter


def JsonlSpanExporter(path: str | Path):  # noqa: N802 - behaves like a class, built lazily to keep otel optional
    return _make_jsonl_exporter()(path)
