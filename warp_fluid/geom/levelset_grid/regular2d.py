from __future__ import annotations

from typing import Tuple, Union

import numpy as np


GridSpec = Union[Tuple[int, int, float, float], object]


def _grid_params(grid: GridSpec, dx: float = None, dy: float = None) -> Tuple[int, int, float, float]:
    if isinstance(grid, tuple):
        if len(grid) != 4:
            raise ValueError("grid tuple must be (nx, ny, dx, dy)")
        nx, ny, dx_val, dy_val = grid
        return int(nx), int(ny), float(dx_val), float(dy_val)
    if dx is None or dy is None:
        if hasattr(grid, "dx") and hasattr(grid, "dy"):
            return int(grid.nx), int(grid.ny), float(grid.dx), float(grid.dy)
        raise ValueError("dx and dy must be provided when grid is not a tuple")
    return int(grid.nx), int(grid.ny), float(dx), float(dy)


def cell_centers(grid: GridSpec, dx: float = None, dy: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return cell center coordinates for a regular 2D grid.

    Args:
        grid: Grid object with (nx, ny) attributes or tuple (nx, ny, dx, dy).
        dx: Cell size in x for object-based grids, meters.
        dy: Cell size in y for object-based grids, meters.
    """
    nx, ny, dx_val, dy_val = _grid_params(grid, dx, dy)
    x = (np.arange(nx, dtype=np.float32) + 0.5) * dx_val
    y = (np.arange(ny, dtype=np.float32) + 0.5) * dy_val
    return np.meshgrid(x, y, indexing="ij")


def sphere_levelset(
    grid: GridSpec,
    center: Tuple[float, float],
    radius: float,
    dx: float = None,
    dy: float = None,
) -> np.ndarray:
    """Signed-distance levelset for a 2D sphere (circle).

    Args:
        grid: Grid object with (nx, ny) attributes or tuple (nx, ny, dx, dy).
        center: Circle center (x, y), meters.
        radius: Circle radius, meters.
        dx: Cell size in x for object-based grids, meters.
        dy: Cell size in y for object-based grids, meters.
    """
    x, y = cell_centers(grid, dx, dy)
    cx, cy = center
    return np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - float(radius)


def box_levelset(
    grid: GridSpec,
    center: Tuple[float, float],
    half_size: Tuple[float, float],
    dx: float = None,
    dy: float = None,
    angle: float = 0.0,
) -> np.ndarray:
    """Signed-distance levelset for an axis-aligned or rotated box.

    Args:
        grid: Grid object with (nx, ny) attributes or tuple (nx, ny, dx, dy).
        center: Box center (x, y), meters.
        half_size: Half sizes (hx, hy), meters.
        dx: Cell size in x for object-based grids, meters.
        dy: Cell size in y for object-based grids, meters.
        angle: Rotation angle in radians.
    """
    x, y = cell_centers(grid, dx, dy)
    cx, cy = center
    x0 = x - cx
    y0 = y - cy
    if angle != 0.0:
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        xr = cos_a * x0 + sin_a * y0
        yr = -sin_a * x0 + cos_a * y0
    else:
        xr = x0
        yr = y0
    hx, hy = half_size
    dxv = np.abs(xr) - float(hx)
    dyv = np.abs(yr) - float(hy)
    outside = np.sqrt(np.maximum(dxv, 0.0) ** 2 + np.maximum(dyv, 0.0) ** 2)
    inside = np.minimum(np.maximum(dxv, dyv), 0.0)
    return outside + inside


def ellipse_levelset(
    grid: GridSpec,
    center: Tuple[float, float],
    radii: Tuple[float, float],
    dx: float = None,
    dy: float = None,
    angle: float = 0.0,
) -> np.ndarray:
    """Approximate signed-distance levelset for a rotated ellipse.

    Args:
        grid: Grid object with (nx, ny) attributes or tuple (nx, ny, dx, dy).
        center: Ellipse center (x, y), meters.
        radii: Semi-axes (rx, ry), meters.
        dx: Cell size in x for object-based grids, meters.
        dy: Cell size in y for object-based grids, meters.
        angle: Rotation angle in radians.
    """
    x, y = cell_centers(grid, dx, dy)
    cx, cy = center
    x0 = x - cx
    y0 = y - cy
    if angle != 0.0:
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        xr = cos_a * x0 + sin_a * y0
        yr = -sin_a * x0 + cos_a * y0
    else:
        xr = x0
        yr = y0
    rx, ry = radii
    rx = float(rx)
    ry = float(ry)
    if rx <= 0.0 or ry <= 0.0:
        raise ValueError("Ellipse radii must be positive.")
    norm = np.sqrt((xr / rx) ** 2 + (yr / ry) ** 2)
    return (norm - 1.0) * min(rx, ry)
