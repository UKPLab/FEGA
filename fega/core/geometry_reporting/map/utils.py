from __future__ import annotations

import math
from typing import Any

import numpy as np


def finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def row_label_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get(key))
        counts[label] = counts.get(label, 0) + 1
    return counts


def median_record_value(records: list[dict[str, Any]], key: str) -> float | None:
    values = [finite_float(record.get(key)) for record in records]
    finite = [value for value in values if value is not None]
    return None if not finite else float(np.median(finite))


def format_optional_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"
