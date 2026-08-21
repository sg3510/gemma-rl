"""Adapter registry.

Adding an eval format means writing a module with `detect(path) -> bool` and
`load(path, name=None) -> Run`, then listing it here. Nothing else in the
viewer changes.
"""
from __future__ import annotations

from pathlib import Path

from viewer.adapters import tau1, tau2
from viewer.model import Run

ADAPTERS = [tau2, tau1]


def detect(path: Path):
    for adapter in ADAPTERS:
        if adapter.detect(path):
            return adapter
    return None


def load(path: Path, name: str | None = None) -> Run:
    adapter = detect(path)
    if adapter is None:
        raise ValueError(
            f"no adapter recognises {path}. Expected a tau2 results.json (or "
            "its directory), or a tau1 sweep directory with meta.json."
        )
    return adapter.load(path, name=name)
