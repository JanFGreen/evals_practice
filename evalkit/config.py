import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RUNS_DIR = Path(os.environ.get("EVAL_RUNS_DIR", ROOT / "runs"))

LLM_MODES = ("live", "cache", "replay", "fake")


def app_model() -> str:
    return os.environ.get("APP_MODEL", "claude-opus-5-5")


def judge_model() -> str:
    return os.environ.get("JUDGE_MODEL", "claude-opus-5-5")


def app_effort() -> str:
    return os.environ.get("APP_EFFORT", "low")


def judge_effort() -> str:
    return os.environ.get("JUDGE_EFFORT", "medium")


def llm_mode() -> str:
    mode = os.environ.get("EVAL_LLM_MODE", "fake")
    if mode not in LLM_MODES:
        raise ValueError(f"EVAL_LLM_MODE={mode!r}, allowed: {', '.join(LLM_MODES)}")
    return mode


def cache_dir() -> Path:
    return Path(os.environ.get("EVAL_CACHE_DIR", ROOT / ".cache" / "llm"))
