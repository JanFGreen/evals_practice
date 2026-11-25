"""Extra module D3: security evals. Tests: uv run pytest tests/test_d3_security.py"""


def find_canaries(text: str, canaries: list[str]) -> list[str]:
    """Canary markers present in the text (case-insensitive)."""
    raise NotImplementedError


def retrieval_leaks(retrieved: list[dict], workspace: str) -> list[str]:
    """Ids of chunks from a workspace other than the asker's workspace."""
    raise NotImplementedError
