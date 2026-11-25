"""Module 5: building and cleaning the dataset. Tests: uv run pytest tests/test_m05_dataset.py"""


def jaccard(a: set, b: set) -> float:
    """|a ∩ b| / |a ∪ b|; 1.0 for two empty sets."""
    raise NotImplementedError


def near_duplicates(items: list[tuple[str, str]], threshold: float = 0.8) -> list[tuple[str, str, float]]:
    """Pairs (id_a, id_b, similarity) with Jaccard of word sets (lowercase, \\w+) >= threshold."""
    raise NotImplementedError


def sample_tuples(dimensions: dict[str, list[str]], n: int, seed: int = 0) -> list[dict[str, str]]:
    """n distinct combinations of dimension values; if n allows it, every value appears at least once."""
    raise NotImplementedError
