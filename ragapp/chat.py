"""Multi-turn wrapper (extra module D2): rewrite follow-ups into standalone questions, then call the RAG app."""

from dataclasses import dataclass, field

from evalkit import config
from evalkit.llm import complete
from ragapp.app import AppResult, answer

REWRITE_SYSTEM = """You rewrite the user's last question into a standalone question that makes sense without the conversation history.
Use the context of earlier messages (e.g. "and in Warsaw?" after a question about hotels -> "What is the hotel limit in Warsaw?").
If the question is already standalone, return it unchanged. Return only the question, with no comment."""

FOLLOW_UP_STARTS = ("and ", "what about", "how about", "what if", "so ", "then ", "ok", "okay", "right")


@dataclass
class Turn:
    user: str
    standalone_question: str
    result: AppResult


@dataclass
class ChatSession:
    workspace: str = "nordwind"
    prompt_version: str = "v2"
    rewrite: bool = True
    turns: list[Turn] = field(default_factory=list)

    def _history_text(self) -> str:
        return "\n".join(f"User: {t.user}\nAssistant: {t.result.answer}" for t in self.turns[-3:])

    def _fake_rewrite(self, message: str) -> str:
        if self.turns and message.lower().startswith(FOLLOW_UP_STARTS):
            return f"{self.turns[-1].standalone_question} {message}"
        return message

    def standalone(self, message: str, rep: int = 0) -> str:
        if not self.rewrite or not self.turns:
            return message
        prompt = f"<history>\n{self._history_text()}\n</history>\n\n<last_question>{message}</last_question>"
        res = complete(model=config.app_model(), system=REWRITE_SYSTEM, messages=[{"role": "user", "content": prompt}],
                       max_tokens=512, effort="low", rep=rep, fake=lambda: self._fake_rewrite(message))
        return res.text.strip() or message

    def send(self, message: str, rep: int = 0) -> Turn:
        question = self.standalone(message, rep)
        result = answer(question, self.workspace, prompt_version=self.prompt_version, rep=rep)
        turn = Turn(message, question, result)
        self.turns.append(turn)
        return turn
