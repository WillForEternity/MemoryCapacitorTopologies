"""Dataset utilities package.

Each subpackage inside ``datasets`` defines generators or PyTorch ``Dataset``
wrappers for a particular benchmark.  The parent package exposes a simple
``load_dataset(name, **kwargs)`` helper that forwards loading to the appropriate
subpackage.
"""
from __future__ import annotations

from types import ModuleType
from importlib import import_module
from typing import Any


def load_dataset(name: str, **kwargs: Any):
    """Load a dataset by *name*.

    Parameters
    ----------
    name : str
        Sub-module name inside ``datasets`` (e.g. ``"mackey_glass"``).
    **kwargs
        Keyword arguments passed through to the dataset module's ``load``
        function.
    """
    module: ModuleType = import_module(f"{__name__}.{name}")
    if not hasattr(module, "load"):
        raise AttributeError(
            f"Dataset module '{name}' must implement a top-level `load(**kw)` function"
        )
    return module.load(**kwargs)
