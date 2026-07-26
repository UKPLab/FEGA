"""Matplotlib renderers for per-feature FEGA paper cards."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

AXIS_NEGATIVE_COLOR = "#4F6D8A"
AXIS_POSITIVE_COLOR = "#C7444E"
AXIS_ZERO_COLOR = "#333333"
_AXIS_GUIDE_COLOR = "#8E44AD"
_CARD_SPHERE_POINT_SIZE = 45
_STANDALONE_SPHERE_POINT_SIZE = 90
_CARD_CLASS_POINT_SIZE = 54
_STANDALONE_CLASS_POINT_SIZE = 108


def render_sphere(
    path: Path,
    coordinates: np.ndarray,
    *,
    color: str,
    dpi: int,
    point_colors: Sequence[str] | None = None,
    mean_vector: np.ndarray | None = None,
) -> None:
    """Write one text-free three-dimensional sphere view of projected directions."""
    # Draw the clean standalone sphere without card titles, ticks, grids, or borders.
    figure = plt.figure(figsize=(6.2, 5.8))
    axis = figure.add_subplot(111, projection="3d")
    _draw_sphere(
        axis,
        coordinates,
        color=color,
        title=None,
        point_colors=point_colors,
        mean_vector=mean_vector,
        point_size=_STANDALONE_SPHERE_POINT_SIZE,
    )
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    plt.close(figure)


def render_projection_2d(
    path: Path,
    coordinates: np.ndarray,
    *,
    color: str,
    dpi: int,
    point_colors: Sequence[str] | None = None,
    axis_guide: bool = False,
    view_padding: float = 1.35,
    primary_axis_guide: bool = True,
    view_limit: float | None = None,
    mode_assignments: Sequence[int] | None = None,
    fitted_line: bool = False,
) -> None:
    """Write one centred, text-free two-dimensional spectral view."""
    # Draw the clean standalone projection without changing its analytical coordinates.
    figure, axis = plt.subplots(figsize=(6.2, 5.8))
    _draw_projection_2d(
        axis,
        coordinates,
        color=color,
        title=None,
        point_colors=point_colors,
        axis_guide=axis_guide,
        view_padding=view_padding,
        primary_axis_guide=primary_axis_guide,
        view_limit=view_limit,
        mode_assignments=mode_assignments,
        fitted_line=fitted_line,
        point_size=_STANDALONE_CLASS_POINT_SIZE,
    )
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    plt.close(figure)


def render_subspace_plane(
    path: Path,
    coordinates: np.ndarray,
    *,
    color: str,
    dpi: int,
    point_colors: Sequence[str] | None = None,
) -> None:
    """Write a text-free view of points around their leading two-dimensional plane."""
    # Draw the existing leading-three-dimensional coordinates without flattening them.
    figure = plt.figure(figsize=(6.2, 5.8))
    axis = figure.add_subplot(111, projection="3d")
    _draw_subspace_plane(
        axis,
        coordinates,
        color=color,
        title=None,
        point_colors=point_colors,
        point_size=_STANDALONE_CLASS_POINT_SIZE,
        show_orthogonal_projection=True,
    )
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    plt.close(figure)


def render_card(
    path: Path,
    sphere_coordinates: np.ndarray,
    plane_coordinates: np.ndarray,
    *,
    color: str,
    title: str,
    plane_title: str,
    footer: str,
    dpi: int,
    point_colors: Sequence[str] | None = None,
    axis_guide: bool = False,
    plane_view_padding: float = 1.35,
    subspace_plane: bool = False,
    plane_primary_axis_guide: bool = True,
    plane_view_limit: float | None = None,
    plane_mode_assignments: Sequence[int] | None = None,
    plane_fitted_line: bool = False,
) -> None:
    """Write a compact paper-card combining a sphere with its class-specific view."""
    # Compose the two faithful analytical panels with only a concise metric footer.
    figure = plt.figure(figsize=(11.5, 5.4))
    _color_figure_border(figure, color)
    sphere_axis = figure.add_subplot(121, projection="3d")
    _draw_sphere(
        sphere_axis,
        sphere_coordinates,
        color=color,
        title="Normalized directions",
        point_colors=point_colors,
        mean_vector=None,
        point_size=_CARD_SPHERE_POINT_SIZE,
    )

    # Show the accepted global plane in 3D; retain the flat class view elsewhere.
    if subspace_plane:
        plane_axis = figure.add_subplot(122, projection="3d")
        _draw_subspace_plane(
            plane_axis,
            plane_coordinates,
            color=color,
            title=plane_title,
            point_colors=point_colors,
            point_size=_CARD_CLASS_POINT_SIZE,
            show_orthogonal_projection=True,
        )
    else:
        plane_axis = figure.add_subplot(122)
        _draw_projection_2d(
            plane_axis,
            plane_coordinates,
            color=color,
            title=plane_title,
            point_colors=point_colors,
            axis_guide=axis_guide,
            view_padding=plane_view_padding,
            primary_axis_guide=plane_primary_axis_guide,
            view_limit=plane_view_limit,
            mode_assignments=plane_mode_assignments,
            fitted_line=plane_fitted_line,
            point_size=_CARD_CLASS_POINT_SIZE,
        )
    figure.suptitle(title, color=color, fontsize=15, fontweight="bold")
    figure.text(0.5, 0.015, footer, ha="center", va="bottom", fontsize=9)
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 0.94))
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _draw_sphere(
    axis: Any,
    coordinates: np.ndarray,
    *,
    color: str,
    title: str | None,
    point_colors: Sequence[str] | None,
    mean_vector: np.ndarray | None,
    point_size: float,
) -> None:
    """Draw projected points and origin rays inside a minimal wireframe sphere."""
    # Build the manuscript-style reference sphere with a fixed orthographic camera.
    azimuth = np.linspace(0.0, 2.0 * np.pi, 32)
    polar = np.linspace(0.0, np.pi, 16)
    x = np.outer(np.cos(azimuth), np.sin(polar))
    y = np.outer(np.sin(azimuth), np.sin(polar))
    z = np.outer(np.ones_like(azimuth), np.cos(polar))
    axis.plot_wireframe(x, y, z, color="#B8C0CC", linewidth=0.35, alpha=0.42)

    # Draw every retained direction as an origin-to-context arrow in its display color.
    colors = [color] * len(coordinates) if point_colors is None else point_colors
    for point, point_color in zip(coordinates, colors, strict=True):
        axis.quiver(
            0.0,
            0.0,
            0.0,
            point[0],
            point[1],
            point[2],
            color=point_color,
            linewidth=0.8,
            alpha=0.55,
            arrow_length_ratio=0.18,
            normalize=False,
        )
    if len(coordinates):
        axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
            c=colors,
            edgecolors=colors,
            s=point_size,
            depthshade=False,
        )

    # Mark the projected sample mean only when the caller requests a residual summary.
    if mean_vector is not None and float(np.linalg.norm(mean_vector)) > 0.0:
        axis.quiver(
            0.0,
            0.0,
            0.0,
            mean_vector[0],
            mean_vector[1],
            mean_vector[2],
            color="#172B4D",
            linewidth=2.6,
            alpha=0.95,
            arrow_length_ratio=0.16,
            normalize=False,
        )
    axis.scatter([0.0], [0.0], [0.0], c="black", s=16)

    # Remove the 3D box and grid while retaining three clean front-facing axes.
    axis.set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), zlim=(-1.05, 1.05))
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.set_proj_type("ortho")
    axis.view_init(elev=18.0, azim=-55.0)
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


def _draw_projection_2d(
    axis: Any,
    coordinates: np.ndarray,
    *,
    color: str,
    title: str | None,
    point_colors: Sequence[str] | None,
    axis_guide: bool,
    view_padding: float,
    primary_axis_guide: bool,
    view_limit: float | None,
    mode_assignments: Sequence[int] | None,
    fitted_line: bool,
    point_size: float,
) -> None:
    """Draw one origin-centred two-dimensional display coordinate cloud."""
    # Use symmetric equal limits so displayed distances and angles remain faithful.
    maximum = float(np.max(np.abs(coordinates))) if coordinates.size else 1.0
    limit = (
        max(maximum * view_padding, 1.0e-6) if view_limit is None else float(view_limit)
    )
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_aspect("equal", adjustable="box")

    # Draw the shared broad class guide, with a display-only band for unsigned axes.
    if axis_guide:
        band_half_width = 0.05 * limit
        axis.axhspan(
            -band_half_width,
            band_half_width,
            color=_AXIS_GUIDE_COLOR,
            alpha=0.12,
            linewidth=0.0,
        )
    if primary_axis_guide:
        axis.axhline(
            0.0,
            color=_AXIS_GUIDE_COLOR,
            linewidth=2.0,
            linestyle=(0, (14, 8)),
        )
    else:
        axis.axhline(0.0, color="#C4CDDC", linewidth=0.8, linestyle="--")
    axis.axvline(0.0, color="#C4CDDC", linewidth=0.8, linestyle="--")

    # Summarize diffuse one-dimensional spread with an orthogonal fit to the shown points.
    if len(coordinates):
        colors = [color] * len(coordinates) if point_colors is None else point_colors
        if fitted_line:
            centroid = coordinates.mean(axis=0)
            centered = coordinates - centroid
            _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
            direction = right_vectors[0]
            positions = centered @ direction
            position_span = positions.max() - positions.min()
            line_positions = np.array(
                [
                    positions.min() - 0.1 * position_span,
                    positions.max() + 0.1 * position_span,
                ]
            )
            endpoints = centroid + np.outer(
                line_positions,
                direction,
            )
            axis.plot(
                endpoints[:, 0],
                endpoints[:, 1],
                color=color,
                linewidth=4.0,
                alpha=0.52,
                solid_capstyle="round",
                zorder=2,
            )

        # Link each fitted mode to its displayed centroid without imposing order.
        if mode_assignments is not None:
            assignments = np.asarray(mode_assignments)
            for mode in np.unique(assignments):
                member_indices = np.flatnonzero(assignments == mode)
                centroid = coordinates[member_indices].mean(axis=0)
                mode_color = colors[int(member_indices[0])]
                for member_index in member_indices:
                    point = coordinates[int(member_index)]
                    axis.plot(
                        [centroid[0], point[0]],
                        [centroid[1], point[1]],
                        color=mode_color,
                        linewidth=0.9,
                        alpha=0.45,
                        zorder=2,
                    )

        # Plot the supplied display coordinates and mark their displayed origin.
        axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=colors,
            edgecolors="white",
            linewidths=0.45,
            s=point_size,
            zorder=3,
        )
    axis.scatter([0.0], [0.0], c="#172B4D", s=18, zorder=4)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    if title is not None:
        axis.set_title(title, color="#172B4D", fontweight="normal")


def _draw_subspace_plane(
    axis: Any,
    coordinates: np.ndarray,
    *,
    color: str,
    title: str | None,
    point_colors: Sequence[str] | None,
    point_size: float,
    show_orthogonal_projection: bool = False,
) -> None:
    """Draw leading-three-dimensional coordinates around the PC1-PC2 plane."""
    # Use one scale for all axes so the visible off-plane displacement is not inflated.
    maximum = float(np.max(np.abs(coordinates))) if coordinates.size else 1.0
    limit = max(maximum * 1.08, 1.0e-6)
    plane_x, plane_y = np.meshgrid(
        np.array([-limit, limit]),
        np.array([-limit, limit]),
    )
    plane_z = np.zeros_like(plane_x)

    # Draw the origin-passing PC1-PC2 plane and its two in-plane spectral axes.
    axis.plot_surface(
        plane_x,
        plane_y,
        plane_z,
        color=color,
        alpha=0.14,
        shade=False,
        linewidth=0.0,
    )
    axis.plot(
        [-limit, limit],
        [0.0, 0.0],
        [0.0, 0.0],
        color=color,
        alpha=0.42,
        linewidth=0.9,
        linestyle="--",
    )
    axis.plot(
        [0.0, 0.0],
        [-limit, limit],
        [0.0, 0.0],
        color=color,
        alpha=0.42,
        linewidth=0.9,
        linestyle="--",
    )

    # Retain every true PC3 height and show its orthogonal foot on the fitted plane.
    if len(coordinates):
        colors = [color] * len(coordinates) if point_colors is None else point_colors
        if show_orthogonal_projection:
            for point, point_color in zip(coordinates, colors, strict=True):
                axis.plot(
                    [point[0], point[0]],
                    [point[1], point[1]],
                    [0.0, point[2]],
                    color=point_color,
                    linewidth=0.9,
                    alpha=0.34,
                )
            axis.scatter(
                coordinates[:, 0],
                coordinates[:, 1],
                np.zeros(len(coordinates)),
                c=colors,
                edgecolors="white",
                linewidths=0.35,
                s=point_size * 0.7,
                alpha=0.32,
                depthshade=False,
            )
        axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
            c=colors,
            edgecolors="white",
            linewidths=0.45,
            s=point_size,
            depthshade=False,
        )
    axis.scatter([0.0], [0.0], [0.0], c="#172B4D", s=18, depthshade=False)

    # Present the plane obliquely with transparent panes and equal metric scaling.
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


def _color_figure_border(figure: Any, color: str) -> None:
    """Apply the atlas family color to the outer figure border."""
    # Keep reference axes neutral while making the candidate family visually explicit.
    figure.patch.set_edgecolor(color)
    figure.patch.set_linewidth(3.0)
