from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from fega.core.visualizations.mode_colors import multi_mode_point_colors
from fega.core.visualizations.projection import (
    project_directions,
    surface_coordinates,
)
from fega.core.visualizations.runner import (
    _directed_ray_display_coordinates,
    _orient_axis_for_display,
    discover_candidates,
    run_visualizations,
)


def test_gram_projection_matches_explicit_linear_readout() -> None:
    """Preserve exact logit-space geometry and the centered residual view."""
    # Build a synthetic readout whose explicit logits define the comparison kernel.
    generator = torch.Generator().manual_seed(13)
    readout = torch.randn(4, 3, generator=generator, dtype=torch.float64)
    directions = torch.randn(5, 4, generator=generator, dtype=torch.float64)
    gram = readout @ readout.T
    logits = directions @ readout
    expected_kernel = logits @ logits.T

    # Recover the full kernel and its centered counterpart from Gram-only coordinates.
    projection = project_directions(directions, gram, dimensions=5)
    assert projection.kernel == pytest.approx(expected_kernel, abs=1.0e-10)
    assert projection.coordinates @ projection.coordinates.T == pytest.approx(
        expected_kernel, abs=1.0e-10
    )
    centered = project_directions(directions, gram, dimensions=5, centered=True)
    expected_centered = (
        expected_kernel
        - expected_kernel.mean(dim=1, keepdim=True)
        - expected_kernel.mean(dim=0, keepdim=True)
        + expected_kernel.mean()
    )
    assert centered.kernel == pytest.approx(expected_centered, abs=1.0e-10)
    assert centered.coordinates @ centered.coordinates.T == pytest.approx(
        expected_centered, abs=1.0e-10
    )

    # Confirm sign canonicalization and display-only surface normalization.
    for column in projection.coordinates.T:
        anchor = int(torch.argmax(torch.abs(column)).item())
        assert float(column[anchor].item()) >= 0.0
    surface, keep = surface_coordinates(projection.coordinates[:, :3])
    assert keep.any()
    assert torch.linalg.vector_norm(surface, dim=1) == pytest.approx(
        torch.ones(surface.shape[0], dtype=torch.float64), abs=1.0e-12
    )


def test_cached_runner_ranks_renders_and_reports_unavailable_terminal(
    tmp_path: Path,
) -> None:
    """Render top cached candidates while preserving a truthful no-row terminal."""
    # Create one minimal completed run with two ranked features and one terminal feature.
    run_dir = tmp_path / "run"
    effect_dir = run_dir / "compute_effect" / "final_resid"
    report_dir = run_dir / "geometry_reporting"
    gram_dir = run_dir / "data_prep" / "gram_cache"
    effect_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    gram_dir.mkdir(parents=True)
    identities = {
        feature_id: [
            {
                "attribute_label": f"label-{feature_id}-{row}",
                "pair_role": "cause_base_prompts",
                "pair_index": row,
            }
            for row in range(2)
        ]
        for feature_id in (1, 2)
    }
    shard_name = "effect_tensors_00000.pt"
    torch.save(
        {
            "feature_ids": torch.tensor([1, 2]),
            "row_offsets": torch.tensor([0, 2, 4]),
            "pair_indices": torch.tensor([0, 1, 0, 1]),
            "attribute_labels": [
                identity["attribute_label"]
                for feature_id in (1, 2)
                for identity in identities[feature_id]
            ],
            "pair_roles": ["cause_base_prompts"] * 4,
            "candidate_identity": [identities[1], identities[2]],
            "retained_mask": [[True, True], [True, True]],
            "direction": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] * 2),
        },
        effect_dir / shard_name,
    )
    per_feature = {
        str(feature_id): {
            "feature_id": feature_id,
            "usable_effects": 2,
            "candidate_identity": identities[feature_id],
            "retained_mask": [True, True],
            "tensor_shard": shard_name,
            "row_start": 0 if feature_id == 1 else 2,
            "row_end": 2 if feature_id == 1 else 4,
        }
        for feature_id in (1, 2)
    }
    per_feature["3"] = {
        "feature_id": 3,
        "usable_effects": 0,
        "candidate_identity": [],
        "retained_mask": [],
        "tensor_shard": None,
        "row_start": None,
        "row_end": None,
        "skipped_reason": "no_retained_directions",
    }
    _write_json(
        effect_dir / "effect_tensors_manifest.json",
        {
            "counts": {"total_effect_rows": 4, "shard_count": 1},
            "shards": [{"path": shard_name, "rows": 4}],
        },
    )
    _write_json(
        effect_dir / "effect_summary.json",
        {"summary": {"total_effect_rows": 4}, "per_feature": per_feature},
    )
    _write_json(
        report_dir / "geometry_feature_records.json",
        {
            "features": [
                _report_record(1, "directed_ray", m_median=1.0),
                _report_record(2, "directed_ray", m_median=2.0),
                _report_record(
                    3,
                    "insufficient_effect_evidence",
                    m_median=None,
                    terminal_reason="insufficient_effect_evidence",
                ),
            ]
        },
    )
    torch.save(torch.eye(3), gram_dir / "gram.pt")
    palette_path = tmp_path / "palette.json"
    _write_json(palette_path, {"directed_ray": "#123ABC"})

    # Run the cached generator and inspect scientific selection and output availability.
    index_path = run_visualizations(run_dir, top_n=1, palette_path=palette_path, dpi=72)
    index = json.loads(index_path.read_text())
    directed = index["families"]["directed_ray"][0]
    assert directed["feature_id"] == 2
    assert directed["color"] == "#123ABC"
    assert directed["visualization_status"] == "available"
    directed_dir = run_dir / Path(directed["metrics_path"]).parent
    for name in (
        "sphere_ball.png",
        "sphere_surface.png",
        "projection_2d.png",
        "card.png",
    ):
        assert (directed_dir / name).stat().st_size > 0

    terminal = index["families"]["insufficient_effect_evidence"][0]
    assert terminal["feature_id"] == 3
    assert terminal["visualization_status"] == "unavailable"
    terminal_dir = run_dir / Path(terminal["metrics_path"]).parent
    terminal_metrics = json.loads((terminal_dir / "metrics.json").read_text())
    assert terminal_metrics["reason"] == "no_retained_directions"
    assert list(terminal_dir.iterdir()) == [terminal_dir / "metrics.json"]


def test_candidate_ranking_prefers_more_valid_contexts() -> None:
    """Prefer better-supported examples before using effect strength as a tie-breaker."""
    # Give the lower-effect feature more valid contexts and require it to rank first.
    records = [
        _report_record(1, "directed_ray", m_median=1.0, n_valid=12),
        _report_record(2, "directed_ray", m_median=2.0, n_valid=8),
        _report_record(3, "directed_ray", m_median=2.0, n_valid=12),
    ]

    # Within the largest context count, preserve effect magnitude as the next criterion.
    ranked = discover_candidates(records, top_n=3)["directed_ray"]
    assert [candidate["feature_id"] for candidate in ranked] == [3, 1, 2]


def test_unresolved_ranking_prefers_supported_high_rank_diffuse_examples() -> None:
    """Choose unresolved examples that visibly express the reported family."""
    # Keep context count primary, then prefer broader rank before weaker coherence.
    records = [
        _report_record(
            1, "unresolved_high_dimensional_or_diffuse", m_median=9.0, n_valid=32
        ),
        _report_record(
            2, "unresolved_high_dimensional_or_diffuse", m_median=1.0, n_valid=64
        ),
        _report_record(
            3, "unresolved_high_dimensional_or_diffuse", m_median=2.0, n_valid=64
        ),
    ]
    records[0].update({"r_span_pr": 30.0, "c_ray": 0.01, "r_ctr_pr": 30.0})
    records[1].update({"r_span_pr": 12.0, "c_ray": 0.20, "r_ctr_pr": 11.0})
    records[2].update({"r_span_pr": 12.0, "c_ray": 0.10, "r_ctr_pr": 10.0})

    # More contexts win first; equal-rank examples then use lower C_ray.
    ranked = discover_candidates(records, top_n=3)[
        "unresolved_high_dimensional_or_diffuse"
    ]
    assert [candidate["feature_id"] for candidate in ranked] == [3, 2, 1]


def test_axis_display_orientation_preserves_geometry_and_side_balance() -> None:
    """Orient an unsigned axis visually without centering or changing its geometry."""
    # Give the positive side the majority so the display convention performs one flip.
    coordinates = torch.tensor(
        [[0.9, 0.1], [0.8, -0.2], [0.7, 0.3], [-0.6, 0.2]],
        dtype=torch.float64,
    )

    # A global first-axis sign flip must preserve the kernel and the 3-to-1 side split.
    displayed, flipped = _orient_axis_for_display(coordinates)
    assert flipped is True
    assert int((displayed[:, 0] < 0.0).sum().item()) == 3
    assert int((displayed[:, 0] > 0.0).sum().item()) == 1
    assert displayed @ displayed.T == pytest.approx(
        coordinates @ coordinates.T, abs=1.0e-12
    )
    assert displayed.mean(dim=0)[1] == coordinates.mean(dim=0)[1]


def test_directed_ray_display_centres_and_preserves_projected_geometry() -> None:
    """Centre a ray detail view and align its local spread by a rigid transformation."""
    # Keep a copy so display preparation cannot mutate the analytical projection.
    coordinates = torch.tensor(
        [[0.8, -0.2], [0.9, 0.1], [1.0, 0.4], [1.1, -0.1]],
        dtype=torch.float64,
    )
    original = coordinates.clone()

    # Translation plus orthogonal rotation preserves the complete local 2D geometry.
    displayed, centroid = _directed_ray_display_coordinates(coordinates)
    assert centroid == pytest.approx(coordinates.mean(dim=0), abs=1.0e-12)
    assert displayed.mean(dim=0) == pytest.approx(
        torch.zeros(2, dtype=torch.float64), abs=1.0e-12
    )
    assert torch.cdist(displayed, displayed) == pytest.approx(
        torch.cdist(coordinates, coordinates), abs=1.0e-12
    )
    assert float(displayed[:, 0].square().sum().item()) >= float(
        displayed[:, 1].square().sum().item()
    )
    assert torch.equal(coordinates, original)


def test_multi_mode_colors_preserve_membership_under_canonical_display_order() -> None:
    """Color cached vMF memberships without treating raw component IDs as semantic."""
    # Put raw mode 1 left of raw mode 0 so display order differs from label order.
    coordinates = torch.tensor(
        [[0.8, 0.1], [0.9, -0.1], [-0.7, 0.2], [-0.8, -0.2]],
        dtype=torch.float64,
    )
    assignments = [0, 0, 1, 1]

    # Membership remains exact while centroid order chooses deterministic display colors.
    colors, metadata = multi_mode_point_colors(coordinates, assignments)
    assert colors[0] == colors[1]
    assert colors[2] == colors[3]
    assert colors[0] != colors[2]
    assert metadata["display_order_raw_modes"] == [1, 0]
    assert metadata["hard_mode_counts"] == {"0": 2, "1": 2}


def _report_record(
    feature_id: int,
    family: str,
    *,
    m_median: float | None,
    n_valid: int = 2,
    terminal_reason: str | None = None,
) -> dict[str, Any]:
    """Build one concise reporting record for the cached runner test."""
    # Include only fields consumed by candidate discovery, evidence, and card metrics.
    return {
        "feature_id": feature_id,
        "primary_label": family,
        "m_median": m_median,
        "n_valid": n_valid,
        "c_ray": 0.8,
        "s_span_1": 0.9,
        "e_res": 0.1,
        "secondary_flags": [],
        "terminal_reason": terminal_reason,
        "gate_evidence": {family: {"decision": "stable"}},
        "evidence_status": "available",
        "label_confidence": "accepted",
        "missingness": {},
        "selected_k": None,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one small JSON fixture artifact."""
    # Keep fixture construction direct so the test exercises the production loader.
    path.write_text(json.dumps(payload))
