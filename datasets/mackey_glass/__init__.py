"""Mackey–Glass dataset subpackage.

Public API: `load()` -> (train_tensor, test_tensor)
Implementation lives in `generator.py` to keep the namespace clean.
"""
from .generator import load  # re-export

__all__ = ["load"]
