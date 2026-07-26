from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sae_bench.evals.icl_features.artifact_naming import (
    aggregate_artifact_tag,
    tagged_paths,
)


def _parse_spec(value: str) -> tuple[str, str, Path, int | None]:
    parts = value.split("::")
    if len(parts) not in {3, 4}:
        raise argparse.ArgumentTypeError(
            "Expected LABEL::SAE_LOCATION::MASK_OR_FEATURE_JSON[::EXPECTED_COUNT]"
        )
    expected = int(parts[3]) if len(parts) == 4 else None
    return parts[0], parts[1], Path(parts[2]).expanduser(), expected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RAVEL MDBM masks into an IoU-compatible feature summary."
    )
    parser.add_argument("--model-name", default="gemma-2-2b")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--feature-set", action="append", type=_parse_spec, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _feature_ids(path: Path) -> list[int]:
    if not path.is_file():
        raise FileNotFoundError(f"RAVEL mask/feature file not found: {path}")
    if path.suffix.lower() == ".json":
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("feature_ids")
        if not isinstance(payload, list):
            raise ValueError(f"{path}: expected a list or a feature_ids list")
        return sorted({int(value) for value in payload})

    from fega.core.utils.models import load_mdbm_mask

    mask = load_mdbm_mask(path)
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu()
    return [
        index
        for index, value in enumerate(mask)
        if float(value) > 0.0
    ]


def _sae_uid(repo_id: str, location: str) -> str:
    return f"{repo_id.replace('/', '_')}__{location.replace('/', '_')}"


def main() -> None:
    args = parse_args()
    feature_sets = {}
    for label, location, path, expected_count in args.feature_set:
        ids = _feature_ids(path)
        if expected_count is not None and len(ids) != expected_count:
            raise ValueError(
                f"{label}: found {len(ids)} RAVEL features in {path}, "
                f"expected {expected_count}"
            )
        uid = _sae_uid(args.repo_id, location)
        feature_sets[uid] = {
            "label": label,
            "sae_release": args.repo_id,
            "sae_id": location,
            "threshold_feature_ids": ids,
            "threshold_feature_count": len(ids),
            "candidate_feature_ids": ids,
            "candidate_feature_count": len(ids),
            "source_path": str(path.resolve()),
            "selection_method": "RAVEL MDBM positive binary mask",
        }
    payload = {
        "schema_version": 1,
        "model_name": args.model_name,
        "task": "ravel",
        "artifact_tag": aggregate_artifact_tag(
            model_name=args.model_name,
            sae_uids=feature_sets,
        ),
        "feature_sets": feature_sets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path in tagged_paths(args.output, payload["artifact_tag"]):
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote RAVEL feature summary: {args.output}")


if __name__ == "__main__":
    main()
