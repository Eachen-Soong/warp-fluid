from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .regular2d import GridSpec, cell_centers


Vec2 = np.ndarray


@dataclass(frozen=True)
class _TeslaUnitGeometry:
    trunk: np.ndarray
    outer_branch: np.ndarray
    inner_obstacle: np.ndarray
    inlet: np.ndarray
    outlet: np.ndarray


def _as_point(x: float, y: float) -> Vec2:
    return np.asarray([float(x), float(y)], dtype=np.float64)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        raise ValueError("Cannot normalize a zero-length vector.")
    return vec / norm


def _cross_2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _reflect_y(points: np.ndarray) -> np.ndarray:
    reflected = np.asarray(points, dtype=np.float64).copy()
    reflected[..., 1] *= -1.0
    return reflected


def _rigid_transform_from_segments(
    source_start: np.ndarray,
    source_end: np.ndarray,
    target_start: np.ndarray,
    target_end: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    src_dir = source_end - source_start
    tgt_dir = target_end - target_start
    src_len = float(np.linalg.norm(src_dir))
    tgt_len = float(np.linalg.norm(tgt_dir))
    if src_len <= 0.0 or tgt_len <= 0.0:
        raise ValueError("Cannot align zero-length segments.")
    if not np.isclose(src_len, tgt_len, rtol=1.0e-6, atol=1.0e-8):
        raise ValueError("Tesla-valve segment alignment changed segment length.")

    src_u = src_dir / src_len
    tgt_u = tgt_dir / tgt_len
    src_v = np.asarray([-src_u[1], src_u[0]], dtype=np.float64)
    tgt_v = np.asarray([-tgt_u[1], tgt_u[0]], dtype=np.float64)
    src_basis = np.stack([src_u, src_v], axis=1)
    tgt_basis = np.stack([tgt_u, tgt_v], axis=1)
    rotation = tgt_basis @ src_basis.T
    translation = target_start - rotation @ source_start
    return rotation, translation


def _apply_transform(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    return pts @ rotation.T + translation


def _grid_extent(grid: GridSpec, dx: float = None, dy: float = None) -> Tuple[float, float]:
    if isinstance(grid, tuple):
        if len(grid) != 4:
            raise ValueError("grid tuple must be (nx, ny, dx, dy)")
        nx, ny, dx_val, dy_val = grid
        return float(nx) * float(dx_val), float(ny) * float(dy_val)
    if dx is None or dy is None:
        if hasattr(grid, "dx") and hasattr(grid, "dy"):
            return float(grid.nx) * float(grid.dx), float(grid.ny) * float(grid.dy)
        raise ValueError("dx and dy must be provided when grid is not a tuple")
    return float(grid.nx) * float(dx), float(grid.ny) * float(dy)


def _sample_semicircle(
    start: np.ndarray,
    end: np.ndarray,
    *,
    num_segments: int,
    reference_point: np.ndarray,
) -> np.ndarray:
    if num_segments < 2:
        raise ValueError("num_segments must be at least 2 for semicircle sampling.")

    center = 0.5 * (start + end)
    radius = 0.5 * float(np.linalg.norm(end - start))
    if radius <= 0.0:
        raise ValueError("Semicircle endpoints must be distinct.")

    start_angle = float(np.arctan2(start[1] - center[1], start[0] - center[0]))
    delta = np.linspace(0.0, np.pi, num_segments + 1, dtype=np.float64)
    angle_sets = (start_angle + delta, start_angle - delta)

    candidates = []
    for angles in angle_sets:
        pts = np.stack(
            [
                center[0] + radius * np.cos(angles),
                center[1] + radius * np.sin(angles),
            ],
            axis=1,
        )
        candidates.append(pts)

    mid_index = num_segments // 2
    distances = [
        float(np.linalg.norm(candidate[mid_index] - reference_point)) for candidate in candidates
    ]
    chosen = candidates[int(distances[1] > distances[0])]
    chosen[0] = start
    chosen[-1] = end
    return chosen


def _polygon_signed_distance(
    x: np.ndarray,
    y: np.ndarray,
    vertices: np.ndarray,
) -> np.ndarray:
    pts = np.asarray(vertices, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
        raise ValueError("Polygon vertices must have shape (N, 2) with N >= 3.")

    signed = np.zeros_like(x, dtype=bool)
    min_dist_sq = np.full_like(x, np.inf, dtype=np.float64)
    x0 = pts[:, 0]
    y0 = pts[:, 1]
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)
    eps = 1.0e-12

    for ax, ay, bx, by in zip(x0, y0, x1, y1):
        intersects = ((ay > y) != (by > y)) & (
            x < (bx - ax) * (y - ay) / ((by - ay) + eps) + ax
        )
        signed ^= intersects

        abx = bx - ax
        aby = by - ay
        ab_len_sq = abx * abx + aby * aby
        if ab_len_sq <= 0.0:
            proj_x = ax
            proj_y = ay
        else:
            t = ((x - ax) * abx + (y - ay) * aby) / ab_len_sq
            t = np.clip(t, 0.0, 1.0)
            proj_x = ax + t * abx
            proj_y = ay + t * aby
        dist_sq = (x - proj_x) ** 2 + (y - proj_y) ** 2
        min_dist_sq = np.minimum(min_dist_sq, dist_sq)

    dist = np.sqrt(min_dist_sq)
    return np.where(signed, -dist, dist).astype(np.float32, copy=False)


def _axis_aligned_box_signed_distance(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> np.ndarray:
    if x_max < x_min:
        raise ValueError("x_max must be >= x_min for a box levelset.")
    if y_max < y_min:
        raise ValueError("y_max must be >= y_min for a box levelset.")
    cx = 0.5 * (float(x_min) + float(x_max))
    cy = 0.5 * (float(y_min) + float(y_max))
    hx = 0.5 * (float(x_max) - float(x_min))
    hy = 0.5 * (float(y_max) - float(y_min))
    dxv = np.abs(x - cx) - hx
    dyv = np.abs(y - cy) - hy
    outside = np.sqrt(np.maximum(dxv, 0.0) ** 2 + np.maximum(dyv, 0.0) ** 2)
    inside = np.minimum(np.maximum(dxv, dyv), 0.0)
    return (outside + inside).astype(np.float32, copy=False)


def _with_domain_pipes(
    fluid_phi: np.ndarray,
    x_local: np.ndarray,
    y_local: np.ndarray,
    *,
    units: Tuple[_TeslaUnitGeometry, ...],
    center_x: float,
    domain_x: float,
) -> np.ndarray:
    inlet = np.asarray(units[0].inlet, dtype=np.float64)
    outlet = np.asarray(units[-1].outlet, dtype=np.float64)

    inlet_x = float(np.mean(inlet[:, 0]))
    outlet_x = float(np.mean(outlet[:, 0]))
    inlet_y_min = float(np.min(inlet[:, 1]))
    inlet_y_max = float(np.max(inlet[:, 1]))
    outlet_y_min = float(np.min(outlet[:, 1]))
    outlet_y_max = float(np.max(outlet[:, 1]))

    left_boundary_x = -float(center_x)
    right_boundary_x = float(domain_x) - float(center_x)
    left_pipe_length = inlet_x - left_boundary_x
    right_pipe_length = right_boundary_x - outlet_x
    if left_pipe_length < 0.0:
        raise ValueError(
            "Left horizontal pipe length is negative. Adjust the Tesla valve center or domain size."
        )
    if right_pipe_length < 0.0:
        raise ValueError(
            "Right horizontal pipe length is negative. Adjust the Tesla valve center or domain size."
        )

    if left_pipe_length > 0.0:
        left_pipe_phi = _axis_aligned_box_signed_distance(
            x_local,
            y_local,
            x_min=left_boundary_x,
            x_max=inlet_x,
            y_min=inlet_y_min,
            y_max=inlet_y_max,
        )
        fluid_phi = np.minimum(fluid_phi, left_pipe_phi)

    if right_pipe_length > 0.0:
        right_pipe_phi = _axis_aligned_box_signed_distance(
            x_local,
            y_local,
            x_min=outlet_x,
            x_max=right_boundary_x,
            y_min=outlet_y_min,
            y_max=outlet_y_max,
        )
        fluid_phi = np.minimum(fluid_phi, right_pipe_phi)

    return fluid_phi.astype(np.float32, copy=False)


def _build_forward_unit(d0: float, d1: float, d2: float, theta: float, arc_segments: int) -> _TeslaUnitGeometry:
    if d0 <= 0.0 or d1 <= 0.0 or d2 <= 0.0:
        raise ValueError("d0, d1, and d2 must be positive.")
    if d2 >= d1:
        raise ValueError("d2 must be smaller than d1 so the return tube has nonzero thickness.")
    if not (0.0 < theta < np.pi):
        raise ValueError("theta must lie in (0, pi) radians.")
    if arc_segments < 8:
        raise ValueError("arc_segments must be at least 8.")

    sin_half = float(np.sin(0.5 * theta))
    cos_half = float(np.cos(0.5 * theta))
    if abs(sin_half) < 1.0e-8 or abs(cos_half) < 1.0e-8:
        raise ValueError("theta is too close to a degenerate Tesla-valve geometry.")

    a0 = _as_point(0.0, 0.0)
    c2 = _as_point(
        d1 * np.cos(theta) / (2.0 * sin_half),
        d1 * np.cos(theta) / (2.0 * cos_half),
    )
    c0 = _as_point(
        d1 / (2.0 * sin_half),
        -d1 / (2.0 * cos_half),
    )

    alpha = (d1 - d2) / (2.0 * d1)
    beta = (d1 + d2) / (2.0 * d1)
    b0 = alpha * c0
    b1 = beta * c0
    d0_pt = alpha * c0 + beta * c2
    d1_pt = beta * c0 + alpha * c2

    trunk_offset = _as_point(0.0, -d0)
    a1 = a0 + trunk_offset
    c1 = c0 + trunk_offset

    outer_arc = _sample_semicircle(
        c2,
        c0,
        num_segments=arc_segments,
        reference_point=a0,
    )
    inner_arc = _sample_semicircle(
        d1_pt,
        d0_pt,
        num_segments=arc_segments,
        reference_point=0.5 * (b0 + b1),
    )

    trunk = np.stack([a0, c0, c1, a1], axis=0)
    outer_branch = np.concatenate([a0[None, :], outer_arc], axis=0)
    inner_obstacle = np.concatenate([b0[None, :], b1[None, :], inner_arc], axis=0)

    inlet = np.stack([a0, a1], axis=0)
    outlet = np.stack([c0, c1], axis=0)
    return _TeslaUnitGeometry(
        trunk=trunk,
        outer_branch=outer_branch,
        inner_obstacle=inner_obstacle,
        inlet=inlet,
        outlet=outlet,
    )


def _mirror_unit(unit: _TeslaUnitGeometry) -> _TeslaUnitGeometry:
    return _TeslaUnitGeometry(
        trunk=_reflect_y(unit.trunk),
        outer_branch=_reflect_y(unit.outer_branch),
        inner_obstacle=_reflect_y(unit.inner_obstacle),
        inlet=_reflect_y(unit.inlet),
        outlet=_reflect_y(unit.outlet),
    )


def _transform_unit(
    unit: _TeslaUnitGeometry,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> _TeslaUnitGeometry:
    return _TeslaUnitGeometry(
        trunk=_apply_transform(unit.trunk, rotation, translation),
        outer_branch=_apply_transform(unit.outer_branch, rotation, translation),
        inner_obstacle=_apply_transform(unit.inner_obstacle, rotation, translation),
        inlet=_apply_transform(unit.inlet, rotation, translation),
        outlet=_apply_transform(unit.outlet, rotation, translation),
    )


def _build_unit_pack(
    *,
    d0: float,
    d1: float,
    d2: float,
    theta: float,
    num_units: int,
    arc_segments: int,
) -> Tuple[_TeslaUnitGeometry, ...]:
    if num_units <= 0:
        raise ValueError("num_units must be positive.")

    forward = _build_forward_unit(d0, d1, d2, theta, arc_segments)
    reverse = _mirror_unit(forward)

    units = [forward]
    previous = forward
    for index in range(1, 2 * num_units):
        template = reverse if index % 2 == 1 else forward
        rotation, translation = _rigid_transform_from_segments(
            template.inlet[0],
            template.inlet[1],
            previous.outlet[1],
            previous.outlet[0],
        )
        placed = _transform_unit(template, rotation, translation)
        units.append(placed)
        previous = placed

    all_vertices = np.concatenate(
        [geom.trunk for geom in units] + [geom.outer_branch for geom in units],
        axis=0,
    )
    bbox_center = 0.5 * (np.min(all_vertices, axis=0) + np.max(all_vertices, axis=0))
    centered_units = []
    for unit in units:
        centered_units.append(
            _transform_unit(unit, np.eye(2, dtype=np.float64), -bbox_center)
        )
    return tuple(centered_units)


def tesla_valve_fluid_levelset(
    grid: GridSpec,
    center: Tuple[float, float],
    d0: float,
    d1: float,
    d2: float,
    theta: float,
    *,
    num_units: int = 1,
    arc_segments: int = 96,
    include_end_pipes: bool = False,
    dx: float = None,
    dy: float = None,
) -> np.ndarray:
    """Signed-distance field for the Tesla-valve fluid region.

    Args:
        grid: Grid object with (nx, ny) attrs or tuple (nx, ny, dx, dy).
        center: Center of the full forward-reverse valve pack, meters.
        d0: Main-channel width, meters.
        d1: Outer diameter of the return tube, meters.
        d2: Inner diameter of the return tube, meters.
        theta: Return-tube angle, radians.
        num_units: Number of forward+reverse Tesla-valve pairs.
        arc_segments: Polyline segments used for each semicircle.
        include_end_pipes: If true, extend the left inlet and right outlet with
            horizontal pipes so the fluid path reaches the west/east domain boundaries.
        dx: Cell size in x for object-based grids, meters.
        dy: Cell size in y for object-based grids, meters.
    """
    x, y = cell_centers(grid, dx, dy)
    cx, cy = center
    x_local = x - float(cx)
    y_local = y - float(cy)

    units = _build_unit_pack(
        d0=float(d0),
        d1=float(d1),
        d2=float(d2),
        theta=float(theta),
        num_units=int(num_units),
        arc_segments=int(arc_segments),
    )
    fluid_phi = np.full_like(x_local, np.inf, dtype=np.float32)
    for unit in units:
        trunk_phi = _polygon_signed_distance(x_local, y_local, unit.trunk)
        outer_phi = _polygon_signed_distance(x_local, y_local, unit.outer_branch)
        inner_phi = _polygon_signed_distance(x_local, y_local, unit.inner_obstacle)
        branch_phi = np.maximum(outer_phi, -inner_phi)
        unit_phi = np.minimum(trunk_phi, branch_phi)
        fluid_phi = np.minimum(fluid_phi, unit_phi)
    if include_end_pipes:
        domain_x, _ = _grid_extent(grid, dx, dy)
        fluid_phi = _with_domain_pipes(
            fluid_phi,
            x_local,
            y_local,
            units=units,
            center_x=float(cx),
            domain_x=domain_x,
        )
    return fluid_phi.astype(np.float32, copy=False)


def tesla_valve_levelset(
    grid: GridSpec,
    center: Tuple[float, float],
    d0: float,
    d1: float,
    d2: float,
    theta: float,
    *,
    num_units: int = 1,
    arc_segments: int = 96,
    include_end_pipes: bool = False,
    dx: float = None,
    dy: float = None,
) -> np.ndarray:
    """Signed-distance field for the solid obstacle outside the Tesla-valve fluid region."""
    fluid_phi = tesla_valve_fluid_levelset(
        grid,
        center,
        d0,
        d1,
        d2,
        theta,
        num_units=num_units,
        arc_segments=arc_segments,
        include_end_pipes=include_end_pipes,
        dx=dx,
        dy=dy,
    )
    return -fluid_phi


__all__ = ["tesla_valve_fluid_levelset", "tesla_valve_levelset"]
