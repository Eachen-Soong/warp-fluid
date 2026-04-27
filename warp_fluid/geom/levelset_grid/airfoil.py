from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .regular2d import GridSpec, cell_centers


def _parse_naca4_code(code: str) -> tuple[float, float, float]:
    digits = str(code).strip()
    if len(digits) != 4 or not digits.isdigit():
        raise ValueError("NACA 4-digit code must contain exactly four digits, e.g. '2412'.")
    max_camber = float(int(digits[0])) / 100.0
    camber_position = float(int(digits[1])) / 10.0
    thickness = float(int(digits[2:])) / 100.0
    return max_camber, camber_position, thickness


def naca4_airfoil_polygon(
    code: str,
    *,
    chord: float,
    leading_edge: tuple[float, float] = (0.0, 0.0),
    angle: float = 0.0,
    samples: int = 256,
    closed_trailing_edge: bool = True,
) -> np.ndarray:
    """Return a closed polygon for a NACA 4-digit airfoil."""

    if chord <= 0.0:
        raise ValueError("chord must be positive.")
    if samples < 16:
        raise ValueError("samples must be >= 16.")

    max_camber, camber_position, thickness = _parse_naca4_code(code)
    beta = np.linspace(0.0, math.pi, samples, dtype=np.float64)
    x = 0.5 * (1.0 - np.cos(beta))

    a4 = -0.1036 if closed_trailing_edge else -0.1015
    y_t = 5.0 * thickness * (
        0.2969 * np.sqrt(np.maximum(x, 0.0))
        - 0.1260 * x
        - 0.3516 * x * x
        + 0.2843 * x * x * x
        + a4 * x * x * x * x
    )

    y_c = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    if max_camber > 0.0 and camber_position > 0.0:
        left = x < camber_position
        right = ~left
        left_denom = camber_position * camber_position
        right_denom = (1.0 - camber_position) * (1.0 - camber_position)
        y_c[left] = max_camber / left_denom * (2.0 * camber_position * x[left] - x[left] * x[left])
        y_c[right] = max_camber / right_denom * (
            (1.0 - 2.0 * camber_position) + 2.0 * camber_position * x[right] - x[right] * x[right]
        )
        dyc_dx[left] = 2.0 * max_camber / left_denom * (camber_position - x[left])
        dyc_dx[right] = 2.0 * max_camber / right_denom * (camber_position - x[right])

    theta = np.arctan(dyc_dx)
    x_u = x - y_t * np.sin(theta)
    y_u = y_c + y_t * np.cos(theta)
    x_l = x + y_t * np.sin(theta)
    y_l = y_c - y_t * np.cos(theta)

    upper = np.column_stack((x_u[::-1], y_u[::-1]))
    lower = np.column_stack((x_l[1:], y_l[1:]))
    polygon = np.vstack((upper, lower)).astype(np.float64, copy=False)

    polygon[:, 0] *= chord
    polygon[:, 1] *= chord
    polygon[:, 0] += float(leading_edge[0])
    polygon[:, 1] += float(leading_edge[1])

    if angle != 0.0:
        pivot = np.array(
            [float(leading_edge[0]) + 0.25 * chord, float(leading_edge[1])],
            dtype=np.float64,
        )
        shifted = polygon - pivot
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        rotated = np.empty_like(shifted)
        rotated[:, 0] = cos_a * shifted[:, 0] - sin_a * shifted[:, 1]
        rotated[:, 1] = sin_a * shifted[:, 0] + cos_a * shifted[:, 1]
        polygon = rotated + pivot

    return polygon.astype(np.float32, copy=False)


def _point_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    inside = np.zeros(points.shape[0], dtype=bool)
    x = points[:, 0]
    y = points[:, 1]
    px = polygon[:, 0]
    py = polygon[:, 1]
    count = polygon.shape[0]
    for i in range(count):
        j = (i + 1) % count
        yi = py[i]
        yj = py[j]
        crosses = (yi > y) != (yj > y)
        if not np.any(crosses):
            continue
        x_intersect = (px[j] - px[i]) * (y[crosses] - yi) / ((yj - yi) + 1.0e-12) + px[i]
        inside[crosses] ^= x[crosses] < x_intersect
    return inside


def _signed_distance_to_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    min_dist_sq = np.full(points.shape[0], np.inf, dtype=np.float64)
    for i in range(polygon.shape[0]):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % polygon.shape[0]]
        edge = p1 - p0
        edge_norm_sq = float(np.dot(edge, edge))
        if edge_norm_sq <= 1.0e-20:
            diff = points - p0
            dist_sq = np.sum(diff * diff, axis=1)
        else:
            rel = points - p0
            t = np.clip((rel @ edge) / edge_norm_sq, 0.0, 1.0)
            projection = p0 + t[:, None] * edge[None, :]
            diff = points - projection
            dist_sq = np.sum(diff * diff, axis=1)
        min_dist_sq = np.minimum(min_dist_sq, dist_sq)

    sign = np.where(_point_in_polygon(points, polygon), -1.0, 1.0)
    return np.sqrt(min_dist_sq) * sign


def naca4_airfoil_levelset(
    grid: GridSpec,
    code: str,
    *,
    chord: float,
    leading_edge: Tuple[float, float],
    angle: float = 0.0,
    dx: float = None,
    dy: float = None,
    samples: int = 256,
    closed_trailing_edge: bool = True,
) -> np.ndarray:
    """Signed-distance levelset for a NACA 4-digit airfoil."""

    polygon = naca4_airfoil_polygon(
        code,
        chord=chord,
        leading_edge=leading_edge,
        angle=angle,
        samples=samples,
        closed_trailing_edge=closed_trailing_edge,
    )
    x, y = cell_centers(grid, dx, dy)
    points = np.column_stack((x.reshape(-1), y.reshape(-1))).astype(np.float64, copy=False)
    sdf = _signed_distance_to_polygon(points, polygon.astype(np.float64, copy=False))
    return sdf.reshape(x.shape).astype(np.float32, copy=False)


__all__ = ["naca4_airfoil_levelset", "naca4_airfoil_polygon"]
