from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path


def default_discovery_root(result_root: Path) -> Path:
    return Path("data/induction_feature_outputs") / result_root.name


def slugify(value: object | None) -> str:
    raw = "" if value is None else str(value).strip()
    slug = re.sub(r"[^A-Za-z0-9.+-]+", "-", raw)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug


def infer_sae_arch(sae_uid: str | None) -> str | None:
    if not sae_uid:
        return None
    lowered = sae_uid.lower()
    if "gemma-scope" in lowered or "gemmascope" in lowered:
        return "gemmascope"
    if "matryoshka" in lowered:
        return "matryoshka-batch-topk"
    if "top_k" in lowered or "topk" in lowered:
        return "topk"
    if "standard" in lowered or "relu" in lowered:
        return "relu"
    return None


def infer_sae_width(value: str | None) -> str | None:
    if not value:
        return None
    lowered = value.lower()
    match = re.search(r"width[-_]?2pow(\d+)", lowered)
    if match:
        return f"2pow{match.group(1)}"
    match = re.search(r"width[-_]?(\d+)k", lowered)
    if match:
        return f"{match.group(1)}k"
    return None


def infer_sae_l0(value: str | None) -> str | None:
    if not value:
        return None
    lowered = value.lower()
    match = re.search(r"average[-_]?l0[-_]?(\d+)", lowered)
    if match:
        return f"l0-{match.group(1)}"
    return None


def artifact_tag(
    *,
    model_name: str | None = None,
    sae_uid: str | None = None,
    sae_width: str | None = None,
    task: str | None = None,
    aggregate: bool = False,
) -> str:
    parts: list[str] = []
    if model_name:
        parts.append(slugify(model_name))
    if aggregate:
        width = sae_width or infer_sae_width(sae_uid)
        if width:
            parts.append(slugify(width))
        parts.append("aggregate")
    elif sae_uid:
        arch = infer_sae_arch(sae_uid)
        width = sae_width or infer_sae_width(sae_uid)
        l0 = infer_sae_l0(sae_uid)
        if arch:
            parts.append(slugify(arch))
        if width:
            parts.append(slugify(width))
        if l0:
            parts.append(slugify(l0))
        if not any((arch, width, l0)):
            parts.append(slugify(sae_uid)[-80:])
    elif sae_width:
        parts.append(slugify(sae_width))
    if task:
        parts.append(slugify(task))
    return "__".join(part for part in parts if part)


def aggregate_artifact_tag(
    *,
    model_name: str | None = None,
    sae_uids: Iterable[str] = (),
    sae_width: str | None = None,
) -> str:
    uids = [uid for uid in sae_uids if uid]
    widths = sorted({width for uid in uids if (width := infer_sae_width(uid))})
    if sae_width:
        width = sae_width
    elif len(widths) == 1:
        width = widths[0]
    elif widths:
        width = "-".join(widths)
    else:
        width = None
    return artifact_tag(model_name=model_name, sae_width=width, aggregate=True)


def tagged_path(path: Path, tag: str | None) -> Path:
    if not tag:
        return path
    suffix = "".join(path.suffixes)
    if suffix:
        stem = path.name[: -len(suffix)]
    else:
        stem = path.name
    return path.with_name(f"{stem}__{tag}{suffix}")


def tagged_paths(path: Path, tag: str | None) -> list[Path]:
    tagged = tagged_path(path, tag)
    if tagged == path:
        return [path]
    return [path, tagged]
