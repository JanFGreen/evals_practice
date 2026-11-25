"""Minimal chat UI for E2E tests (extra module D5). Stdlib only: uv run python -m ragapp.web"""

import json
import os
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from evalkit.llm import LLMError
from ragapp.app import answer

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Nordwind Assist</title>
<style>
 body{font:15px/1.5 system-ui,sans-serif;max-width:720px;margin:40px auto;padding:0 16px;color:#1d232a}
 form{display:flex;gap:8px} input[type=text]{flex:1;padding:8px} button{padding:8px 14px}
 #panel{margin-top:20px;padding:14px;border:1px solid #d5dbe1;border-radius:6px;min-height:60px}
 #panel[data-state=loading]::before{content:"Assistant is typing…";color:#68737d}
 .badge{display:inline-block;background:#fff3cd;padding:2px 8px;border-radius:4px;font-size:13px}
 #error{color:#a4161a} ul{padding-left:18px} li{font-size:13px;color:#46515c}
</style></head><body>
<h1>Nordwind Assist</h1>
<form id="ask">
  <select id="workspace" aria-label="Workspace"><option>nordwind</option><option>orbita</option></select>
  <input id="question" type="text" aria-label="Question" placeholder="Ask about company policies" required>
  <button id="send" type="submit">Ask</button>
</form>
<div id="panel" data-testid="answer-panel" data-state="idle" aria-live="polite">
  <span id="refusal" class="badge" data-testid="refusal-badge" hidden>Not in the documents</span>
  <p id="answer" data-testid="answer"></p>
  <ul id="sources" data-testid="sources"></ul>
  <p id="error" data-testid="error" role="alert" hidden></p>
</div>
<script>
const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
$("ask").addEventListener("submit", async (e) => {
  e.preventDefault();
  const panel = $("panel");
  panel.dataset.state = "loading";
  $("send").disabled = true;
  $("answer").textContent = ""; $("sources").replaceChildren();
  $("error").hidden = true; $("refusal").hidden = true;
  try {
    const res = await fetch("/api/ask", {method: "POST", headers: {"content-type": "application/json"},
      body: JSON.stringify({question: $("question").value, workspace: $("workspace").value})});
    if (!res.ok) throw new Error((await res.json()).error || res.statusText);
    const data = await res.json();
    panel.dataset.state = "streaming";
    for (const word of data.answer.split(" ")) {  // simulated token streaming
      $("answer").textContent += ($("answer").textContent ? " " : "") + word;
      await sleep(15);
    }
    for (const id of data.retrieved_doc_ids) {
      const li = document.createElement("li");
      li.textContent = id; li.dataset.testid = "source"; li.dataset.cited = data.citations.includes(id);
      $("sources").append(li);
    }
    $("refusal").hidden = !data.answer.trim().toLowerCase().startsWith("i don't know");
    panel.dataset.state = "done";
  } catch (err) {
    $("error").textContent = "Could not get an answer: " + err.message;
    $("error").hidden = false;
    panel.dataset.state = "error";
  } finally {
    $("send").disabled = false;
  }
});
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/health":
            self._send(200, b'{"ok": true}', "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):  # noqa: N802
        if self.path != "/api/ask":
            return self._send(404, b"not found", "text/plain")
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
            result = answer(payload["question"], payload.get("workspace", "nordwind"),
                            prompt_version=payload.get("prompt_version", "v2"))
            body = {**asdict(result), "retrieved_doc_ids": result.retrieved_doc_ids}
            self._send(200, json.dumps(body, ensure_ascii=False).encode("utf-8"), "application/json")
        except (LLMError, KeyError, json.JSONDecodeError) as e:
            self._send(502 if isinstance(e, LLMError) else 400,
                       json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8"), "application/json")

    def log_message(self, fmt, *args):
        if os.environ.get("RAGAPP_WEB_LOG"):
            super().log_message(fmt, *args)


def main() -> None:
    port = int(os.environ.get("PORT", "8765"))
    print(f"Nordwind Assist: http://127.0.0.1:{port}  (EVAL_LLM_MODE={os.environ.get('EVAL_LLM_MODE', 'cache')})")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
