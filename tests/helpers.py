import importlib
import os
from types import SimpleNamespace


def load(exercise: str, *solutions: str):
    """Tests run against your exercise module; EVAL_TARGET=solution runs them against evalkit instead."""
    if os.environ.get("EVAL_TARGET") != "solution":
        return importlib.import_module(exercise)
    merged = {}
    for name in solutions:
        merged.update(vars(importlib.import_module(name)))
    return SimpleNamespace(**merged)
