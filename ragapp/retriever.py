"""BM25 over Markdown policies, one chunk per `##` section. Deliberately simple and offline."""

import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

CORPUS_DIR = Path(__file__).parent / "corpus"
# Crude prefix stemming (truncate every token); blind to synonyms. 4 is the default; check it with `evalkit.cli retrieval --sweep`.
STEM_LEN = int(os.environ.get("RAGAPP_STEM_LEN", "4"))
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at", "by", "for", "with", "from",
    "is", "are", "was", "be", "been", "do", "does", "did", "can", "could", "should", "would", "will", "have",
    "has", "i", "me", "my", "we", "you", "your", "it", "its", "this", "that", "what", "which", "who", "when",
    "how", "much", "many", "there", "any", "some", "as", "so", "not", "no", "yes", "get", "am",
}


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    workspace: str
    title: str
    section: str
    text: str
    version: str
    status: str
    valid_from: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    rank: int


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\w+", text.lower())
    return [t[:STEM_LEN] for t in tokens if t.isdigit() or (len(t) > 1 and t not in STOPWORDS)]


def _parse_meta(line: str) -> dict[str, str]:
    meta = {}
    for part in line.split("|"):
        if ":" in part:
            key, value = part.split(":", 1)
            meta[key.strip().lower()] = value.strip()
    return meta


def parse_doc(path: Path, workspace: str) -> list[Chunk]:
    lines = path.read_text(encoding="utf-8").splitlines()
    title = lines[0].lstrip("#").strip()
    meta = _parse_meta(lines[1]) if len(lines) > 1 else {}
    doc_id = path.stem

    sections: list[tuple[str, list[str]]] = []
    for line in lines[2:]:
        if line.startswith("## "):
            sections.append((line[3:].strip(), []))
        elif sections and line.strip():
            sections[-1][1].append(line.strip())

    return [
        Chunk(
            id=f"{doc_id}#{i}",
            doc_id=doc_id,
            workspace=workspace,
            title=title,
            section=heading,
            text=f"{title} — {heading}\n" + "\n".join(body),
            version=meta.get("version", ""),
            status=meta.get("status", ""),
            valid_from=meta.get("valid from", ""),
        )
        for i, (heading, body) in enumerate(sections)
    ]


@lru_cache
def load_chunks(corpus_dir: Path = CORPUS_DIR) -> tuple[Chunk, ...]:
    chunks: list[Chunk] = []
    for ws_dir in sorted(p for p in corpus_dir.iterdir() if p.is_dir()):
        for path in sorted(ws_dir.glob("*.md")):
            chunks.extend(parse_doc(path, ws_dir.name))
    return tuple(chunks)


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        docs = [tokenize(c.text) for c in chunks]
        self.lengths = [len(d) for d in docs]
        self.avgdl = sum(self.lengths) / max(len(docs), 1)
        self.tf = [Counter(d) for d in docs]
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def _score(self, query_tokens: list[str], i: int) -> float:
        tf, dl = self.tf[i], self.lengths[i]
        score = 0.0
        for t in query_tokens:
            if t not in tf:
                continue
            f = tf[t]
            score += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return score

    def search(self, query: str, k: int) -> list[Hit]:
        q = tokenize(query)
        scored = [(self._score(q, i), i) for i in range(len(self.chunks))]
        top = sorted((s for s in scored if s[0] > 0), key=lambda s: (-s[0], s[1]))[:k]
        return [Hit(self.chunks[i], round(score, 4), rank) for rank, (score, i) in enumerate(top, 1)]


@lru_cache
def _index(workspace: str | None) -> BM25:
    chunks = [c for c in load_chunks() if workspace is None or c.workspace == workspace]
    return BM25(chunks)


def configure(stem_len: int) -> None:
    global STEM_LEN
    STEM_LEN = stem_len
    _index.cache_clear()


def workspaces() -> list[str]:
    return sorted({c.workspace for c in load_chunks()})


def search(query: str, workspace: str = "nordwind", k: int = 4, *, enforce_tenant: bool = True) -> list[Hit]:
    # enforce_tenant=False reproduces a classic multi-tenant bug: one shared index, no workspace filter.
    return _index(workspace if enforce_tenant else None).search(query, k)
