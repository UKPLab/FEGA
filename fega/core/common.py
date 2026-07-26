from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import Any, Iterable, Iterator

from fega.config_schema import FEGAPipelineConfig


def require_single_entity_attr(config: FEGAPipelineConfig) -> tuple[str, str]:
    """Return the sole (entity, attribute) pair or raise if config has more."""
    if len(config.entity_attribute_selection) != 1:
        raise ValueError("Pipeline currently supports exactly one entity selection.")
    entity, attrs = next(iter(config.entity_attribute_selection.items()))
    if len(attrs) != 1:
        raise ValueError("Pipeline currently supports exactly one attribute selection.")
    return entity, attrs[0]


def selection_seed(config: FEGAPipelineConfig) -> int:
    """Selection seed with fallback to the global seed."""
    return config.seed.selection_seed or config.seed.global_


@contextmanager
def patched_attr(obj: Any, name: str, replacement: Any) -> Iterator[None]:
    """Temporarily replace an attribute on an object or module."""
    original = getattr(obj, name)
    setattr(obj, name, replacement)
    try:
        yield
    finally:
        setattr(obj, name, original)


@contextmanager
def stacked(managers: Iterable[Any]) -> Iterator[None]:
    """Enter a collection of context managers in order."""
    with ExitStack() as stack:
        for manager in managers:
            stack.enter_context(manager)
        yield
