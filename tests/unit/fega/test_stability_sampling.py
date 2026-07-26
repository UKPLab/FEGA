from __future__ import annotations

from fega.core.stability.sampling import (
    SubsetPlan,
    build_subset_plan,
    low_context_protocol,
    subspace_resample_indices,
)


def test_subset_plan_identity_is_complete_and_schedule_independent() -> None:
    first = build_subset_plan(
        global_seed=42,
        feature_id=7,
        protocol="sample_size",
        target_or_group_identity="16",
        replicate_id=3,
        purpose="gate_margins",
        indices=[9, 2, 5, 2],
    )
    second = build_subset_plan(
        global_seed=42,
        feature_id=7,
        protocol="sample_size",
        target_or_group_identity="16",
        replicate_id=3,
        purpose="gate_margins",
        indices=[5, 2, 9, 2],
    )

    assert isinstance(first, SubsetPlan)
    assert first == second
    assert first.indices == (2, 2, 5, 9)
    assert len(first.digest) == 64


def test_subspace_sampling_uses_without_replacement_fraction() -> None:
    samples = subspace_resample_indices(10, 0.7, 5, seed=1)
    assert len(samples) == 5
    assert all(len(sample) == 7 for sample in samples)
    assert all(len(set(sample.tolist())) == len(sample) for sample in samples)


def test_low_context_gate_protocols() -> None:
    assert low_context_protocol(7)["protocol"] == "descriptive"
    assert low_context_protocol(8)["protocol"] == "leave_out_sensitivity"
    assert low_context_protocol(16)["protocol"] == "exploratory_subsampling"
    assert low_context_protocol(32)["protocol"] == "principal_angle"
