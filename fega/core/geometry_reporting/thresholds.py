from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeometryThresholds:
    name: str
    n_min: int
    tau_zero_filter_frac: float
    tau_c_ray: float
    tau_axis: float
    tau_r_2D: float
    tau_b_axis: float
    tau_m_cv: float
    tau_span_k: float
    tau_r: dict[int, float]
    tau_p: dict[int, float]
    tau_gap_k: float
    tau_subspace_angle: dict[str, float]
    tau_mix: float
    tau_mode_mass: float
    tau_mode_c_ray: float
    tau_assignment_stability: float
    tau_res: float
    tau_ctr: dict[int, float]
    tau_r_ctr: dict[int, float]
    tau_longtail: float


_PROFILES = {
    "paper": GeometryThresholds(
        name="paper",
        # insufficient evidence
        n_min=8,
        tau_zero_filter_frac=0.30,

        # directed ray
        tau_c_ray=0.80,
        tau_axis=0.80, # for s_span_1
        tau_r_2D=1.45, # for r_span_pr

        # magnitude unstable
        tau_m_cv=1.00, # for m_cv

        # axis or antipodal
        tau_b_axis=0.15, # for b_axis

        # multi_mode_directional_geometry
        # selected_mode_count > 1
        tau_mix=0.10, # for delta_mix
        tau_mode_mass=0.10, # for mode_mass
        tau_mode_c_ray=0.70, # for min_mode_c_ray
        tau_assignment_stability=0.80, # for assignment_stability

        # global kD
        tau_span_k=0.90, 
        tau_r={2: 1.60, 3: 2.30, 4: 3.00, 8: 5.00},
        tau_p={2: 0.08, 3: 0.05, 4: 0.03, 8: 0.01},
        tau_gap_k=0.60,
        tau_subspace_angle={"1": 30.0, "2": 30.0, "k": 35.0},

        # residual        
        tau_res=0.10,
        tau_ctr={1: 0.80, 2: 0.80, 3: 0.80, 4: 0.80},
        tau_r_ctr={2: 1.50, 3: 2.20, 4: 2.90},
        tau_longtail=1.50,
    ),
}


def get_threshold_profile(name: str) -> GeometryThresholds:
    try:
        return _PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown geometry_reporting threshold profile: {name!r}") from exc


def subspace_angle_threshold(thresholds: GeometryThresholds, k: int) -> float:
    if k == 1:
        return thresholds.tau_subspace_angle["1"]
    if k == 2:
        return thresholds.tau_subspace_angle["2"]
    return thresholds.tau_subspace_angle["k"]
