from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def gram_magnitude(delta: torch.Tensor, gram: torch.Tensor) -> torch.Tensor:
    """Return exact finite Gram magnitudes, rejecting persistent negative forms."""
    # Accumulate q in float64 and fail closed rather than clamping invalid values.
    delta_f = delta.to(dtype=torch.float64)
    gram_f = gram.to(device=delta_f.device, dtype=torch.float64)
    q = torch.sum((delta_f @ gram_f) * delta_f, dim=-1)
    if not torch.isfinite(q).all():
        raise ValueError("Nonfinite final_resid Gram quadratic form.")
    if torch.any(q < 0):
        raise ArithmeticError("Persistently negative final_resid Gram quadratic form.")
    return torch.sqrt(q)


def effect_direction(delta: torch.Tensor, magnitude: torch.Tensor) -> torch.Tensor:
    """Normalize retained deltas by their exact positive Gram magnitude."""
    # Divide without an additive epsilon after the caller applies tau_zero.
    return delta.to(dtype=torch.float64) / magnitude.to(
        device=delta.device, dtype=torch.float64
    ).unsqueeze(-1)


def summarize_magnitudes(values: list[float]) -> dict[str, float | None]:
    """Compute stable magnitude summary stats for one feature."""
    if not values:
        return {
            "mean_magnitude": None,
            "std_magnitude": None,
            "median_magnitude": None,
            "q10_magnitude": None,
            "q90_magnitude": None,
            "cv_magnitude": None,
        }
    tensor = torch.tensor(values, dtype=torch.float32)
    mean = float(tensor.mean().item())
    std = float(tensor.std(unbiased=False).item()) if tensor.numel() > 1 else 0.0
    median = float(torch.quantile(tensor, 0.5).item())
    q10 = float(torch.quantile(tensor, 0.1).item())
    q90 = float(torch.quantile(tensor, 0.9).item())
    return {
        "mean_magnitude": mean,
        "std_magnitude": std,
        "median_magnitude": median,
        "q10_magnitude": q10,
        "q90_magnitude": q90,
        "cv_magnitude": (std / mean) if mean > 0 else None,
    }


class EffectArtifactWriter:
    """Write flattened compute_effect shards and final metadata artifacts."""

    def __init__(
        self,
        output_dir: Path,
        shard_size: int,
        *,
        include_magnitude_direction: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.shard_size = int(shard_size)
        self.include_magnitude_direction = bool(include_magnitude_direction)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._shard_idx = 0
        self._rows: list[dict[str, Any]] = []
        self._group_feature_ids: list[int] = []
        self._row_offsets: list[int] = []
        self._candidate_identities: list[list[dict[str, Any]]] = []
        self._retained_masks: list[list[bool]] = []
        self.shard_records: list[dict[str, Any]] = []
        self.total_rows = 0

    def add_feature_rows(
        self,
        feature_id: int,
        rows: list[dict[str, Any]],
        *,
        candidate_identity: list[dict[str, Any]],
        retained_mask: list[bool],
    ) -> dict[str, Any] | None:
        """Append rows for one feature, preserving one feature per shard pointer."""
        if not rows:
            return None
        if self._rows and len(self._rows) + len(rows) > self.shard_size:
            self.flush()
        row_start = len(self._rows)
        self._group_feature_ids.append(int(feature_id))
        self._row_offsets.append(row_start)
        self._candidate_identities.append(candidate_identity)
        self._retained_masks.append(retained_mask)
        self._rows.extend(rows)
        return {
            "tensor_shard": self._shard_name(self._shard_idx),
            "row_start": row_start,
            "row_end": len(self._rows),
        }

    def flush(self) -> None:
        """Write the current shard if it has rows."""
        if not self._rows:
            return
        shard_name = self._shard_name(self._shard_idx)
        shard_path = self.output_dir / shard_name
        tmp_path = shard_path.with_name(shard_path.name + ".tmp")
        payload = {
            "feature_ids": torch.tensor(self._group_feature_ids, dtype=torch.long),
            "row_offsets": torch.tensor(
                [*self._row_offsets, len(self._rows)], dtype=torch.long
            ),
            "context_indices": torch.tensor(
                [row["context_index"] for row in self._rows], dtype=torch.long
            ),
            "pair_indices": torch.tensor(
                [row["pair_index"] for row in self._rows], dtype=torch.long
            ),
            "attribute_labels": [row["attribute_label"] for row in self._rows],
            "pair_roles": [row["pair_role"] for row in self._rows],
            "candidate_identity": self._candidate_identities,
            "retained_mask": self._retained_masks,
            "feature_activations": torch.tensor(
                [row["feature_activation"] for row in self._rows],
                dtype=torch.float32,
            ),
            "delta": torch.stack(
                [
                    row["delta"].detach().cpu().to(dtype=torch.float32)
                    for row in self._rows
                ]
            ),
        }
        if self.include_magnitude_direction:
            payload["magnitude"] = torch.tensor(
                [row["magnitude"] for row in self._rows], dtype=torch.float32
            )
            payload["direction"] = torch.stack(
                [
                    row["direction"].detach().cpu().to(dtype=torch.float32)
                    for row in self._rows
                ]
            )
            payload["unit_gram_norm_error"] = torch.tensor(
                [row["unit_gram_norm_error"] for row in self._rows],
                dtype=torch.float64,
            )
        torch.save(payload, tmp_path)
        tmp_path.replace(shard_path)
        rows = len(self._rows)
        self.shard_records.append(
            {
                "shard": self._shard_idx,
                "path": str(shard_path),
                "rows": rows,
                "feature_ids": list(self._group_feature_ids),
                "row_start": self.total_rows,
                "row_end": self.total_rows + rows,
            }
        )
        self.total_rows += rows
        self._shard_idx += 1
        self._rows = []
        self._group_feature_ids = []
        self._row_offsets = []
        self._candidate_identities = []
        self._retained_masks = []

    def write_summary(self, path: Path, summary: dict[str, Any]) -> None:
        """Write summary JSON after shards have been flushed."""
        tmp_path = Path(path).with_name(Path(path).name + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(summary, f, indent=2)
        tmp_path.replace(path)

    def write_manifest(self, path: Path, manifest: dict[str, Any]) -> None:
        """Write final manifest JSON last to avoid advertising partial runs."""
        tmp_path = Path(path).with_name(Path(path).name + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(manifest, f, indent=2)
        tmp_path.replace(path)

    @staticmethod
    def _shard_name(idx: int) -> str:
        return f"effect_tensors_{idx:05d}.pt"


def validate_manifest_summary_consistency(
    manifest: dict[str, Any], summary: dict[str, Any]
) -> None:
    """Raise if final manifest and summary disagree on row/shard accounting."""
    manifest_counts = manifest.get("counts", {})
    summary_counts = summary.get("summary", {})
    shard_rows = sum(int(shard.get("rows", 0)) for shard in manifest.get("shards", []))
    manifest_total = int(manifest_counts.get("total_effect_rows", -1))
    summary_total = int(summary_counts.get("total_effect_rows", -1))
    if shard_rows != manifest_total or summary_total != manifest_total:
        raise ValueError(
            "Manifest/summary row mismatch: "
            f"manifest={manifest_total}, summary={summary_total}, shards={shard_rows}"
        )
    if int(manifest_counts.get("shard_count", -1)) != len(manifest.get("shards", [])):
        raise ValueError("Manifest shard_count does not match shard records.")
    pointed_rows = 0
    for feature_id, record in summary.get("per_feature", {}).items():
        # Prove each retained block is the single mask applied to ordered identity.
        candidate_identity = record.get("candidate_identity")
        retained_mask = record.get("retained_mask")
        if not isinstance(candidate_identity, list) or not isinstance(
            retained_mask, list
        ):
            raise ValueError(f"Feature {feature_id} missing identity or retained mask.")
        if len(candidate_identity) != len(retained_mask):
            raise ValueError(f"Feature {feature_id} identity/mask length mismatch.")
        if any(type(value) is not bool for value in retained_mask):
            raise ValueError(f"Feature {feature_id} retained mask must be boolean.")
        required_identity = {"attribute_label", "pair_role", "pair_index"}
        if any(
            not isinstance(identity, dict)
            or set(identity) != required_identity
            for identity in candidate_identity
        ):
            raise ValueError(f"Feature {feature_id} has incomplete candidate identity.")
        tensor_shard = record.get("tensor_shard")
        row_start = record.get("row_start")
        row_end = record.get("row_end")
        usable = int(record.get("usable_effects") or 0)
        if tensor_shard is None:
            if row_start is not None or row_end is not None:
                raise ValueError(
                    f"Skipped feature {feature_id} advertises row pointers."
                )
            continue
        if row_start is None or row_end is None or row_end < row_start:
            raise ValueError(f"Feature {feature_id} has invalid row range.")
        if row_end - row_start != usable:
            raise ValueError(
                f"Feature {feature_id} row range does not match usable_effects."
            )
        if sum(retained_mask) != usable:
            raise ValueError(
                f"Feature {feature_id} retained mask does not select usable_effects."
            )
        pointed_rows += usable
    if pointed_rows != manifest_total:
        raise ValueError(
            f"Per-feature row ranges total {pointed_rows}, expected {manifest_total}."
        )
