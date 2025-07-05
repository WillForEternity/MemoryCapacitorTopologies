"""Registry and loader for reservoir topology generators.

Each generator resides in its own module inside this package and registers
itself via the ``@register("name")`` decorator.  This file automatically
imports all sibling modules so that the registry is populated at import time.
"""
from __future__ import annotations

import importlib
import pkgutil
import pathlib
from typing import Callable, Dict

import numpy as np

# -----------------------------------------------------------------------------
# Registry machinery
# -----------------------------------------------------------------------------
TOPOLOGY_REGISTRY: Dict[str, Callable[..., np.ndarray]] = {}


def register(name: str):
    """Decorator used by individual topology modules to register themselves."""

    def _decorator(func: Callable[..., np.ndarray]):
        TOPOLOGY_REGISTRY[name] = func
        return func

    return _decorator


def make_topology(name: str, **kwargs) -> np.ndarray:
    """Instantiate a topology by *name* using keyword arguments forwarded to the
    registered generator.
    """
    try:
        fn = TOPOLOGY_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown topology '{name}'. Available: {list(TOPOLOGY_REGISTRY.keys())}"
        ) from exc
    return fn(**kwargs)


# -----------------------------------------------------------------------------
# Auto-import sibling modules so they can register themselves.
# -----------------------------------------------------------------------------
_pkg_path = pathlib.Path(__file__).parent
for _module in pkgutil.iter_modules([str(_pkg_path)]):
    if _module.name.startswith("_"):
        # Skip private/dunder modules.
        continue
    importlib.import_module(f"{__name__}.{_module.name}")
