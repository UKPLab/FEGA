"""Selected-dimensional centered-residual views for FEGA paper cards."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt

from fega.core.visualizations.render import (
    _CARD_CLASS_POINT_SIZE,
    _CARD_SPHERE_POINT_SIZE,
    _STANDALONE_CLASS_POINT_SIZE,
    _color_figure_border,
    _draw_projection_2d,
    _draw_sphere,
    _draw_subspace_plane,
)

RESIDUAL_NEGATIVE_COLOR = "#3B4CC0"
RESIDUAL_NEUTRAL_COLOR = "#F7F7F7"
RESIDUAL_POSITIVE_COLOR = "#B40426"


def residual_point_colors(
    first_coordinates: Sequence[float],
) -> tuple[list[str], float]:
    """Map centered residual PC1 continuously to one deterministic diverging scale."""
    # Normalize only the display color coordinate; analytical positions remain unchanged.
    values = [float(value) for value in first_coordinates]
    scale = max((abs(value) for value in values), default=0.0)
    if scale == 0.0:
        return [RESIDUAL_NEUTRAL_COLOR] * len(values), scale

    # Interpolate through a neutral midpoint without introducing discrete color groups.
    colors = []
    for value in values:
        normalized = value / scale
        if normalized < 0.0:
            colors.append(
                _interpolate_hex(
                    RESIDUAL_NEUTRAL_COLOR,
                    RESIDUAL_NEGATIVE_COLOR,
                    -normalized,
                )
            )
        else:
            colors.append(
                _interpolate_hex(
                    RESIDUAL_NEUTRAL_COLOR,
                    RESIDUAL_POSITIVE_COLOR,
                    normalized,
                )
            )
    return colors, scale


def residual_display_kind(selected_k: int | None) -> str:
    """Name the selected-k residual display without claiming extra structure."""
    # Keep metadata aligned with the literal coordinates shown in the output.
    if selected_k == 2:
        return "centered_residual_pc1_pc2_plane_with_pc3_height"
    if selected_k == 3:
        return "centered_residual_pc1_pc2_pc3"
    if selected_k == 4:
        return "centered_residual_pc12_and_pc34"
    return "centered_residual_pc1_pc2"


def render_residual_view(
    path: Path,
    coordinates: np.ndarray,
    *,
    selected_k: int,
    color: str,
    dpi: int,
    point_colors: Sequence[str],
) -> None:
    """Write the selected-dimensional centered-residual view without clustering."""
    # Choose only a display that can expose the selected residual coordinates honestly.
    if selected_k == 2:
        figure = plt.figure(figsize=(6.2, 5.8))
        axis = figure.add_subplot(111, projection="3d")
        _draw_subspace_plane(
            axis,
            coordinates[:, :3],
            color=color,
            title=None,
            point_colors=point_colors,
            point_size=_STANDALONE_CLASS_POINT_SIZE,
        )
    elif selected_k == 3:
        figure = plt.figure(figsize=(6.2, 5.8))
        axis = figure.add_subplot(111, projection="3d")
        _draw_residual_volume(
            axis,
            coordinates[:, :3],
            title=None,
            point_colors=point_colors,
            point_size=_STANDALONE_CLASS_POINT_SIZE,
        )
    elif selected_k == 4:
        figure, axes = plt.subplots(2, 1, figsize=(6.2, 7.0))
        _draw_residual_pairs(
            axes,
            coordinates,
            point_colors=point_colors,
            point_size=_STANDALONE_CLASS_POINT_SIZE,
            titles=(None, None),
        )
    else:
        figure, axis = plt.subplots(figsize=(6.2, 5.8))
        _draw_projection_2d(
            axis,
            coordinates[:, :2],
            color=color,
            title=None,
            point_colors=point_colors,
            axis_guide=False,
            view_padding=1.35,
            primary_axis_guide=False,
            view_limit=None,
            mode_assignments=None,
            fitted_line=False,
            point_size=_STANDALONE_CLASS_POINT_SIZE,
        )
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    plt.close(figure)


def render_residual_card(
    path: Path,
    sphere_coordinates: np.ndarray,
    residual_coordinates: np.ndarray,
    *,
    selected_k: int,
    color: str,
    title: str,
    plane_title: str,
    footer: str,
    dpi: int,
    point_colors: Sequence[str],
    sphere_mean_vector: np.ndarray,
) -> None:
    """Write a paper card with the true mean and selected residual coordinates."""
    # Give the normalized cloud and residual coordinates equal card space.
    figure = plt.figure(figsize=(11.5, 5.4))
    _color_figure_border(figure, color)
    grid = figure.add_gridspec(1, 2)
    sphere_axis = figure.add_subplot(grid[0, 0], projection="3d")
    _draw_sphere(
        sphere_axis,
        sphere_coordinates,
        color=color,
        title="Normalized directions",
        point_colors=point_colors,
        mean_vector=sphere_mean_vector,
        point_size=_CARD_SPHERE_POINT_SIZE,
    )

    # Match the residual panel to k instead of forcing every result into a plane.
    if selected_k == 2:
        residual_axis = figure.add_subplot(grid[0, 1], projection="3d")
        _draw_subspace_plane(
            residual_axis,
            residual_coordinates[:, :3],
            color=color,
            title=plane_title,
            point_colors=point_colors,
            point_size=_CARD_CLASS_POINT_SIZE,
        )
    elif selected_k == 3:
        residual_axis = figure.add_subplot(grid[0, 1], projection="3d")
        _draw_residual_volume(
            residual_axis,
            residual_coordinates[:, :3],
            title=plane_title,
            point_colors=point_colors,
            point_size=_CARD_CLASS_POINT_SIZE,
        )
    elif selected_k == 4:
        residual_grid = grid[0, 1].subgridspec(2, 1, hspace=0.12)
        residual_axes = (
            figure.add_subplot(residual_grid[0, 0]),
            figure.add_subplot(residual_grid[1, 0]),
        )
        _draw_residual_pairs(
            residual_axes,
            residual_coordinates,
            point_colors=point_colors,
            point_size=_CARD_CLASS_POINT_SIZE,
            titles=("Residual PC1–PC2", "Residual PC3–PC4"),
        )
    else:
        residual_axis = figure.add_subplot(grid[0, 1])
        _draw_projection_2d(
            residual_axis,
            residual_coordinates[:, :2],
            color=color,
            title=plane_title,
            point_colors=point_colors,
            axis_guide=False,
            view_padding=1.35,
            primary_axis_guide=False,
            view_limit=None,
            mode_assignments=None,
            fitted_line=False,
            point_size=_CARD_CLASS_POINT_SIZE,
        )

    # Keep the shared card title and footer consistent with every other family.
    figure.suptitle(title, color=color, fontsize=15, fontweight="bold")
    figure.text(0.5, 0.015, footer, ha="center", va="bottom", fontsize=9)
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 0.94))
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _draw_residual_volume(
    axis: Any,
    coordinates: np.ndarray,
    *,
    title: str | None,
    point_colors: Sequence[str],
    point_size: float,
) -> None:
    """Draw the exact leading three centered-residual coordinates without a plane."""
    # Use one scale and three neutral axes so no component is visually privileged.
    maximum = float(np.max(np.abs(coordinates))) if coordinates.size else 1.0
    limit = max(maximum * 1.12, 1.0e-6)
    axis.plot(
        [-limit, limit],
        [0.0, 0.0],
        [0.0, 0.0],
        color="#C4CDDC",
        linewidth=0.8,
        linestyle="--",
    )
    axis.plot(
        [0.0, 0.0],
        [-limit, limit],
        [0.0, 0.0],
        color="#C4CDDC",
        linewidth=0.8,
        linestyle="--",
    )
    axis.plot(
        [0.0, 0.0],
        [0.0, 0.0],
        [-limit, limit],
        color="#C4CDDC",
        linewidth=0.8,
        linestyle="--",
    )

    # Plot each centered residual at its true leading-three-dimensional coordinate.
    if len(coordinates):
        axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
            c=point_colors,
            edgecolors="white",
            linewidths=0.45,
            s=point_size,
            depthshade=False,
        )
    axis.scatter([0.0], [0.0], [0.0], c="#172B4D", s=18, depthshade=False)

    # Present the residual coordinate system without panes, ticks, or perspective.
    axis.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.set_proj_type("ortho")
    axis.view_init(elev=24.0, azim=-55.0)
    axis.grid(False)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_zticks([])
    for dimension_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
        dimension_axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        dimension_axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        dimension_axis.line.set_color("#202020")
        dimension_axis.line.set_linewidth(0.8)
    if title is not None:
        axis.set_title(title, color="#172B4D", fontweight="normal")


def _draw_residual_pairs(
    axes: Sequence[Any],
    coordinates: np.ndarray,
    *,
    point_colors: Sequence[str],
    point_size: float,
    titles: tuple[str | None, str | None],
) -> None:
    """Draw exact PC1–PC2 and PC3–PC4 residual-coordinate pairs."""
    # Reuse the neutral 2D renderer for both disjoint coordinate pairs.
    for axis, pair, title in zip(
        axes,
        (coordinates[:, :2], coordinates[:, 2:4]),
        titles,
        strict=True,
    ):
        _draw_projection_2d(
            axis,
            pair,
            color="#172B4D",
            title=title,
            point_colors=point_colors,
            axis_guide=False,
            view_padding=1.18,
            primary_axis_guide=False,
            view_limit=None,
            mode_assignments=None,
            fitted_line=False,
            point_size=point_size,
        )


def _interpolate_hex(start: str, end: str, weight: float) -> str:
    """Interpolate two RGB hex colors for deterministic display styling."""
    # Convert each channel directly so the color scale needs no additional dependency.
    start_rgb = tuple(int(start[index : index + 2], 16) for index in (1, 3, 5))
    end_rgb = tuple(int(end[index : index + 2], 16) for index in (1, 3, 5))
    channels = tuple(
        round(start_value + weight * (end_value - start_value))
        for start_value, end_value in zip(start_rgb, end_rgb, strict=True)
    )
    return "#" + "".join(f"{channel:02X}" for channel in channels)
