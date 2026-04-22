from __future__ import annotations

from typing import Optional

import numpy as np
import warp as wp

from ..core.boundary import SolidMask
from ..core.grid import GridSpec


def _smooth_fluid_indicator(values: np.ndarray, thickness: float) -> np.ndarray:
    if thickness > 0.0:
        return 0.5 * (1.0 + np.tanh(values / float(thickness)))
    return (values > 0.0).astype(np.float32)


def _sample_offsets(count: int) -> np.ndarray:
    if count < 1:
        raise ValueError("sample count must be >= 1.")
    return (np.arange(count, dtype=np.float32) + 0.5) / float(count) - 0.5


def _sample_centered_numpy(grid: GridSpec, field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    gx = (x - grid.x0) / grid.dx - 0.5
    gy = (y - grid.y0) / grid.dy - 0.5
    gx = np.clip(gx, 0.0, float(grid.nx - 1))
    gy = np.clip(gy, 0.0, float(grid.ny - 1))
    i0 = np.floor(gx).astype(np.int32)
    j0 = np.floor(gy).astype(np.int32)
    i1 = np.minimum(i0 + 1, grid.nx - 1)
    j1 = np.minimum(j0 + 1, grid.ny - 1)
    tx = gx - i0
    ty = gy - j0
    f00 = field[i0, j0]
    f10 = field[i1, j0]
    f01 = field[i0, j1]
    f11 = field[i1, j1]
    f0 = f00 * (1.0 - tx) + f10 * tx
    f1 = f01 * (1.0 - tx) + f11 * tx
    return (f0 * (1.0 - ty) + f1 * ty).astype(np.float32)


def _cell_solid_fraction_from_levelset(
    grid: GridSpec,
    levelset: np.ndarray,
    thickness: float,
    *,
    samples_per_axis: int = 2,
) -> np.ndarray:
    x_center = grid.x0 + (np.arange(grid.nx, dtype=np.float32)[:, None] + 0.5) * grid.dx
    y_center = grid.y0 + (np.arange(grid.ny, dtype=np.float32)[None, :] + 0.5) * grid.dy
    fluid_acc = np.zeros(grid.shape, dtype=np.float32)
    x_offsets = _sample_offsets(samples_per_axis) * grid.dx
    y_offsets = _sample_offsets(samples_per_axis) * grid.dy
    for ox in x_offsets:
        for oy in y_offsets:
            phi = _sample_centered_numpy(grid, levelset, x_center + ox, y_center + oy)
            fluid_acc += _smooth_fluid_indicator(phi, thickness)
    fluid_fraction = fluid_acc / float(samples_per_axis * samples_per_axis)
    return np.clip(1.0 - fluid_fraction, 0.0, 1.0).astype(np.float32)


def _u_face_open_fraction_from_levelset(
    grid: GridSpec,
    levelset: np.ndarray,
    thickness: float,
    *,
    samples: int = 4,
) -> np.ndarray:
    x = grid.x0 + np.arange(grid.nx + 1, dtype=np.float32)[:, None] * grid.dx
    y_center = grid.y0 + (np.arange(grid.ny, dtype=np.float32)[None, :] + 0.5) * grid.dy
    fluid_acc = np.zeros(grid.shape_u, dtype=np.float32)
    for oy in _sample_offsets(samples) * grid.dy:
        phi = _sample_centered_numpy(grid, levelset, x, y_center + oy)
        fluid_acc += _smooth_fluid_indicator(phi, thickness)
    return np.clip(fluid_acc / float(samples), 0.0, 1.0).astype(np.float32)


def _v_face_open_fraction_from_levelset(
    grid: GridSpec,
    levelset: np.ndarray,
    thickness: float,
    *,
    samples: int = 4,
) -> np.ndarray:
    x_center = grid.x0 + (np.arange(grid.nx, dtype=np.float32)[:, None] + 0.5) * grid.dx
    y = grid.y0 + np.arange(grid.ny + 1, dtype=np.float32)[None, :] * grid.dy
    fluid_acc = np.zeros(grid.shape_v, dtype=np.float32)
    for ox in _sample_offsets(samples) * grid.dx:
        phi = _sample_centered_numpy(grid, levelset, x_center + ox, y)
        fluid_acc += _smooth_fluid_indicator(phi, thickness)
    return np.clip(fluid_acc / float(samples), 0.0, 1.0).astype(np.float32)


def cell_mask_from_levelset(
    levelset: np.ndarray,
    thickness: float = 0.0,
    *,
    fractional: bool = False,
    grid: Optional[GridSpec] = None,
) -> np.ndarray:
    values = np.asarray(levelset, dtype=np.float32)
    if fractional:
        if grid is None:
            raise ValueError("grid is required when fractional=True.")
        return _cell_solid_fraction_from_levelset(grid, values, thickness)
    if thickness <= 0.0:
        mask = (values <= 0.0).astype(np.float32)
    else:
        fluid = _smooth_fluid_indicator(values, thickness)
        mask = (fluid < 0.5).astype(np.float32)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def face_masks_from_cell_mask(cell_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    solid = np.clip(np.asarray(cell_mask, dtype=np.float32), 0.0, 1.0)
    fluid = 1.0 - solid
    nx, ny = solid.shape
    u_mask = np.zeros((nx + 1, ny), dtype=np.float32)
    v_mask = np.zeros((nx, ny + 1), dtype=np.float32)
    u_mask[0, :] = fluid[0, :]
    u_mask[nx, :] = fluid[nx - 1, :]
    u_mask[1:nx, :] = np.minimum(fluid[:-1, :], fluid[1:, :])
    v_mask[:, 0] = fluid[:, 0]
    v_mask[:, ny] = fluid[:, ny - 1]
    v_mask[:, 1:ny] = np.minimum(fluid[:, :-1], fluid[:, 1:])
    return u_mask.astype(np.float32), v_mask.astype(np.float32)


def solid_mask_from_levelset(
    grid: GridSpec,
    levelset: np.ndarray,
    *,
    thickness: float = 0.0,
    device: Optional[str] = None,
    fractional: bool = True,
) -> SolidMask:
    device = device or "cpu"
    values = np.asarray(levelset, dtype=np.float32)
    if values.shape != grid.shape:
        raise ValueError(f"levelset has shape {values.shape}, expected {grid.shape}")
    if fractional:
        cell = _cell_solid_fraction_from_levelset(grid, values, thickness)
        u_mask = _u_face_open_fraction_from_levelset(grid, values, thickness)
        v_mask = _v_face_open_fraction_from_levelset(grid, values, thickness)
    else:
        cell = cell_mask_from_levelset(values, thickness=thickness, fractional=False)
        u_mask, v_mask = face_masks_from_cell_mask(cell)
    return SolidMask(
        grid=grid,
        cell=wp.array(cell, dtype=wp.float32, device=device),
        u=wp.array(u_mask, dtype=wp.float32, device=device),
        v=wp.array(v_mask, dtype=wp.float32, device=device),
        cell_numpy=cell,
        u_numpy=u_mask,
        v_numpy=v_mask,
    )
