from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "run_compute_effect":
        from .runner import run_compute_effect

        return run_compute_effect
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["run_compute_effect"]
