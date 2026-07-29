"""Export a compact, traceable RAVEL geometry dataset for the static project page.

The script reads raw artifacts under ``results/`` and writes only derived web
assets below ``page/assets/generated/``. It never modifies raw results or paper
sources. Run it from the repository root after regenerating FEGA results.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPOSITORY_ROOT / "results" / "fega" / "ravel"
OUTPUT_ROOT = REPOSITORY_ROOT / "page" / "assets" / "generated"

ARCHITECTURES = {
    "matryoshka": "Matryoshka Batch TopK",
    "relu": "ReLU",
    "topk": "TopK",
}


def architecture_key(path: Path) -> str:
    name = path.name.lower()
    if "matryoshka" in name:
        return "matryoshka"
    if "standard_new" in name:
        return "relu"
    if "top_k" in name:
        return "topk"
    raise ValueError(f"Unrecognized RAVEL run: {path.name}")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_feature(feature: dict[str, Any]) -> dict[str, Any]:
    embedding = feature.get("embedding") or {}
    return {
        "id": feature["feature_id"],
        "label": feature["primary_label"],
        "x": embedding.get("x"),
        "y": embedding.get("y"),
        "n": feature.get("n_valid"),
        "magnitude": feature.get("m_median"),
        "flags": feature.get("secondary_flags", []),
        "confidence": feature.get("label_confidence"),
        "evidence": feature.get("evidence_status"),
    }


def record_metrics(record: dict[str, Any], map_feature: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["feature_id"],
        "label": map_feature["primary_label"],
        "n": record.get("n_valid"),
        "magnitude": record.get("m_median"),
        "c_ray": record.get("c_ray"),
        "r2": record.get("r2"),
        "span_2": record.get("s_span_2"),
        "residual_energy": record.get("e_res"),
        "selected_k": map_feature.get("selected_k"),
        "flags": map_feature.get("secondary_flags", []),
        "evidence": map_feature.get("evidence_status"),
        "decision": map_feature.get("selected_family_decision"),
        "blocker": map_feature.get("strongest_blocker"),
    }


def copy_feature_assets(run_root: Path, label: str, feature_id: int, architecture: str) -> dict[str, str]:
    candidate_root = run_root / "city_Country" / "visualizations" / "candidates" / label
    matches = sorted(candidate_root.glob(f"rank_*_f{feature_id}"))
    if not matches:
        return {}

    source_dir = matches[0]
    destination_dir = OUTPUT_ROOT / "features" / architecture / label / f"f{feature_id}"
    destination_dir.mkdir(parents=True, exist_ok=True)
    assets: dict[str, str] = {}
    for source_name, target_name in (("card.png", "card.png"), ("projection_2d.png", "projection.png")):
        source = source_dir / source_name
        if source.exists():
            destination = destination_dir / target_name
            shutil.copy2(source, destination)
            assets[target_name.removesuffix(".png")] = destination.relative_to(REPOSITORY_ROOT / "page").as_posix()
    return assets


def export_run(run_root: Path) -> dict[str, Any]:
    architecture = architecture_key(run_root)
    report_root = run_root / "city_Country" / "geometry_reporting"
    records = read_json(report_root / "geometry_feature_records.json")
    map_data = read_json(report_root / "geometry_map_data.json")
    record_by_id = {record["feature_id"]: record for record in records["features"]}
    map_by_id = {feature["feature_id"]: feature for feature in map_data["features"]}
    features = [compact_feature(feature) for feature in map_data["features"]]

    featured: dict[str, dict[str, Any]] = {}
    candidates_root = run_root / "city_Country" / "visualizations" / "candidates"
    for label_directory in sorted(path for path in candidates_root.iterdir() if path.is_dir()):
        candidates = sorted(label_directory.glob("rank_*_f*"))
        if not candidates:
            continue
        feature_id = int(candidates[0].name.rsplit("_f", 1)[1])
        if feature_id not in record_by_id or feature_id not in map_by_id:
            continue
        featured[label_directory.name] = {
            "metrics": record_metrics(record_by_id[feature_id], map_by_id[feature_id]),
            "assets": copy_feature_assets(run_root, label_directory.name, feature_id, architecture),
        }

    summary = records["summary"]
    return {
        "id": architecture,
        "name": ARCHITECTURES[architecture],
        "dataset": "RAVEL City-Country",
        "total": summary["features_total"],
        "labels": summary["primary_label_counts"],
        "flags": summary.get("global_flag_counts", {}),
        "features": features,
        "featured": featured,
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    runs = [export_run(run_root) for run_root in sorted(RESULTS_ROOT.iterdir()) if run_root.is_dir()]
    runs.sort(key=lambda run: ("topk", "matryoshka", "relu").index(run["id"]))
    output = {
        "dataset": "RAVEL City-Country",
        "source": "results/fega/ravel",
        "architectures": runs,
    }
    (OUTPUT_ROOT / "ravel_atlas.json").write_text(json.dumps(output, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {OUTPUT_ROOT / 'ravel_atlas.json'} for {len(runs)} architectures.")


if __name__ == "__main__":
    main()
