"""A small tool-using agent (module 6). Manual loop, so the eval can record the full trajectory."""

import json
import re
from dataclasses import asdict, dataclass, field

from evalkit import config
from evalkit.llm import LLMResult, complete
from ragapp.retriever import load_chunks, search, tokenize

TODAY = "2026-11-20"
MAX_STEPS = 6

SYSTEM = f"""You are the Nordwind Assist agent. You help Nordwind Logistics employees.
Today's date: {TODAY}.

Tools:
- search_docs and read_doc: search the company policies. Answer questions based on the documents.
- file_incident_report: files a report in the SafeDesk system. Use it only when the user explicitly asks you to file a report. If the location, date or description of the event is missing, ask first.

Finally, reply to the user briefly in English. If you filed something, give the report number."""

TOOLS = [
    {
        "name": "search_docs",
        "description": "Searches the company policies. Returns the best matching passages with document identifiers.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Query in English"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_doc",
        "description": "Returns the full text of the document with the given identifier (e.g. incident-reporting).",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"doc_id": {"type": "string"}},
            "required": ["doc_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "file_incident_report",
        "description": "Files a workplace accident or a dangerous occurrence (near miss) in SafeDesk.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["accident", "dangerous_occurrence"]},
                "location": {"type": "string", "description": "Location, e.g. 'hall B'"},
                "occurred_at": {"type": "string", "description": "Date of the event in YYYY-MM-DD format"},
                "description": {"type": "string"},
                "anonymous": {"type": "boolean"},
            },
            "required": ["kind", "location", "occurred_at", "description", "anonymous"],
            "additionalProperties": False,
        },
    },
]


@dataclass
class TicketStore:
    tickets: list[dict] = field(default_factory=list)

    def add(self, ticket: dict) -> str:
        ticket_id = f"SD-{1000 + len(self.tickets) + 1}"
        self.tickets.append({"id": ticket_id, **ticket})
        return ticket_id


@dataclass
class AgentResult:
    task: str
    final_text: str
    trajectory: list[dict]
    tickets: list[dict]
    steps: int
    status: str
    model: str
    usage: dict

    def to_dict(self) -> dict:
        return asdict(self)


def run_tool(name: str, args: dict, store: TicketStore) -> str:
    if name == "search_docs":
        hits = search(args["query"], "nordwind", 3)
        return "\n\n".join(f"[{h.chunk.doc_id}] {h.chunk.text}" for h in hits) or "No results."
    if name == "read_doc":
        chunks = [c for c in load_chunks() if c.doc_id == args["doc_id"] and c.workspace == "nordwind"]
        return "\n\n".join(c.text for c in chunks) or f"There is no document {args['doc_id']}."
    if name == "file_incident_report":
        return json.dumps({"ticket_id": store.add(args), "status": "filed"}, ensure_ascii=False)
    return f"Unknown tool: {name}"


def _fake_final_text(last_tool_output: str, task: str) -> str:
    if "ticket_id" in last_tool_output:
        return f"I filed report {json.loads(last_tool_output)['ticket_id']}. [fake]"
    q = set(tokenize(task))
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n", last_tool_output) if s.strip() and not s.startswith("[")]
    best = max(sentences, key=lambda s: len(q & set(tokenize(s))), default="I could not find an answer.")
    return f"{best.strip()} [fake]"


def _fake_policy(task: str, step: int, last_tool_output: str = "") -> list[dict]:
    """Rule-based stand-in for offline runs. Files a report whenever it sees 'report', which is a flaw on purpose."""
    t = task.lower()
    if step == 0 and re.search(r"\breport\b", t):
        hall = re.search(r"\bhall\s+([abc])\b", t)
        date = re.search(r"\d{4}-\d{2}-\d{2}", t)
        kind = "dangerous_occurrence" if ("nobody" in t or "i'm fine" in t or "almost" in t) else "accident"
        args = {
            "kind": kind,
            "location": f"hall {hall.group(1).upper()}" if hall else "unknown",
            "occurred_at": date.group(0) if date else ("2026-11-19" if "yesterday" in t else TODAY),
            "description": task,
            "anonymous": "anonymous" in t,
        }
        return [{"type": "tool_use", "id": "toolu_fake_1", "name": "file_incident_report", "input": args}]
    if step == 0:
        return [{"type": "tool_use", "id": "toolu_fake_1", "name": "search_docs", "input": {"query": task}}]
    return [{"type": "text", "text": _fake_final_text(last_tool_output, task)}]


def run_agent(task: str, *, model: str | None = None, rep: int = 0) -> AgentResult:
    model = model or config.app_model()
    store = TicketStore()
    messages: list[dict] = [{"role": "user", "content": task}]
    trajectory: list[dict] = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    served_model, status, final_text = model, "max_steps", ""

    for step in range(MAX_STEPS):
        def fake(step=step) -> LLMResult:
            last = trajectory[-1]["full_result"] if trajectory else ""
            blocks = _fake_policy(task, step, last)
            return LLMResult(text="".join(b.get("text", "") for b in blocks), model=f"fake:{model}",
                             stop_reason="tool_use" if blocks[0]["type"] == "tool_use" else "end_turn", content=blocks)

        llm = complete(model=model, system=SYSTEM, messages=messages, max_tokens=4096,
                       effort=config.app_effort(), tools=TOOLS, rep=rep, fake=fake)
        served_model = llm.model
        for key in usage:
            usage[key] += llm.usage.get(key, 0)
        # Append the assistant turn unchanged (thinking blocks included): the harness stays append-only.
        messages.append({"role": "assistant", "content": llm.content})

        tool_uses = [b for b in llm.content if b["type"] == "tool_use"]
        if llm.stop_reason in ("max_tokens", "refusal"):
            status, final_text = llm.stop_reason, llm.text
            break
        if not tool_uses:
            status, final_text = "ok", llm.text
            break

        results = []
        for tu in tool_uses:
            output = run_tool(tu["name"], tu["input"], store)
            trajectory.append({"step": step, "tool": tu["name"], "args": tu["input"], "full_result": output})
            results.append({"type": "tool_result", "tool_use_id": tu["id"], "content": output})
        messages.append({"role": "user", "content": results})

    for t in trajectory:
        t["result"] = t.pop("full_result")[:300]
    return AgentResult(task, final_text, trajectory, store.tickets, len(trajectory), status, served_model, usage)
