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


def _sample_centered_numpy_2d(grid: GridSpec, field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
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


def _sample_centered_numpy_3d(
    grid: GridSpec,
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    assert grid.nz is not None and grid.dz is not None
    gx = (x - grid.x0) / grid.dx - 0.5
    gy = (y - grid.y0) / grid.dy - 0.5
    gz = (z - grid.z0) / grid.dz - 0.5
    gx = np.clip(gx, 0.0, float(grid.nx - 1))
    gy = np.clip(gy, 0.0, float(grid.ny - 1))
    gz = np.clip(gz, 0.0, float(grid.nz - 1))
    i0 = np.floor(gx).astype(np.int32)
    j0 = np.floor(gy).astype(np.int32)
    k0 = np.floor(gz).astype(np.int32)
    i1 = np.minimum(i0 + 1, grid.nx - 1)
    j1 = np.minimum(j0 + 1, grid.ny - 1)
    k1 = np.minimum(k0 + 1, grid.nz - 1)
    tx = gx - i0
    ty = gy - j0
    tz = gz - k0
    c000 = field[i0, j0, k0]
    c100 = field[i1, j0, k0]
    c010 = field[i0, j1, k0]
    c110 = field[i1, j1, k0]
    c001 = field[i0, j0, k1]
    c101 = field[i1, j0, k1]
    c011 = field[i0, j1, k1]
    c111 = field[i1, j1, k1]
    c00 = c000 * (1.0 - tx) + c100 * tx
    c10 = c010 * (1.0 - tx) + c110 * tx
    c01 = c001 * (1.0 - tx) + c101 * tx
    c11 = c011 * (1.0 - tx) + c111 * tx
    c0 = c00 * (1.0 - ty) + c10 * ty
    c1 = c01 * (1.0 - ty) + c11 * ty
    return (c0 * (1.0 - tz) + c1 * tz).astype(np.float32)


def _sample_centered_numpy(grid: GridSpec, field: np.ndarray, *coords: np.ndarray) -> np.ndarray:
    if grid.is_3d:
        if len(coords) != 3:
            raise ValueError("3D sampling expects x, y, z coordinates.")
        return _sample_centered_numpy_3d(grid, field, coords[0], coords[1], coords[2])
    if len(coords) != 2:
        raise ValueError("2D sampling expects x, y coordinates.")
    return _sample_centered_numpy_2d(grid, field, coords[0], coords[1])


def _cell_solid_fraction_from_levelset_2d(
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


def _cell_solid_fraction_from_levelset_3d(
    grid: GridSpec,
    levelset: np.ndarray,
    thickness: float,
    *,
    samples_per_axis: int = 2,
) -> np.ndarray:
    assert grid.nz is not None and grid.dz is not None
    x_center = grid.x0 + (np.arange(grid.nx, dtype=np.float32)[:, None, None] + 0.5) * grid.dx
    y_center = grid.y0 + (np.arange(grid.ny, dtype=np.float32)[None, :, None] + 0.5) * grid.dy
    z_center = grid.z0 + (np.arange(grid.nz, dtype=np.float32)[None, None, :] + 0.5) * grid.dz
    fluid_acc = np.zeros(grid.shape, dtype=np.float32)
    x_offsets = _sample_offsets(samples_per_axis) * grid.dx
    y_offsets = _sample_offsets(samples_per_axis) * grid.dy
    z_offsets = _sample_offsets(samples_per_axis) * grid.dz
    for ox in x_offsets:
        for oy in y_offsets:
            for oz in z_offsets:
                phi = _sample_centered_numpy(grid, levelset, x_center + ox, y_center + oy, z_center + oz)
                fluid_acc += _smooth_fluid_indicator(phi, thickness)
    fluid_fraction = fluid_acc / float(samples_per_axis**3)
    return np.clip(1.0 - fluid_fraction, 0.0, 1.0).astype(np.float32)


def _u_face_open_fraction_from_levelset_2d(
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


def _v_face_open_fraction_from_levelset_2d(
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


def _u_face_open_fraction_from_levelset_3d(
    grid: GridSpec,
    levelset: np.ndarray,
    thickness: float,
    *,
    samples: int = 3,
) -> np.ndarray:
    assert grid.nz is not None and grid.dz is not None
    x = grid.x0 + np.arange(grid.nx + 1, dtype=np.float32)[:, None, None] * grid.dx
    y_center = grid.y0 + (np.arange(grid.ny, dtype=np.float32)[None, :, None] + 0.5) * grid.dy
    z_center = grid.z0 + (np.arange(grid.nz, dtype=np.float32)[None, None, :] + 0.5) * grid.dz
    fluid_acc = np.zeros(grid.shape_u, dtype=np.float32)
    for oy in _sample_offsets(samples) * grid.dy:
        for oz in _sample_offsets(samples) * grid.dz:
            phi = _sample_centered_numpy(grid, levelset, x, y_center + oy, z_center + oz)
            fluid_acc += _smooth_fluid_indicator(phi, thickness)
    return np.clip(fluid_acc / float(samples * samples), 0.0, 1.0).astype(np.float32)


def _v_face_open_fraction_from_levelset_3d(
    grid: GridSpec,
    levelset: np.ndarray,
    thickness: float,
    *,
    samples: int = 3,
) -> np.ndarray:
    assert grid.nz is not None and grid.dz is not None
    x_center = grid.x0 + (np.arange(grid.nx, dtype=np.float32)[:, None, None] + 0.5) * grid.dx
    y = grid.y0 + np.arange(grid.ny + 1, dtype=np.float32)[None, :, None] * grid.dy
    z_center = grid.z0 + (np.arange(grid.nz, dtype=np.float32)[None, None, :] + 0.5) * grid.dz
    fluid_acc = np.zeros(grid.shape_v, dtype=np.float32)
    for ox in _sample_offsets(samples) * grid.dx:
        for oz in _sample_offsets(samples) * grid.dz:
            phi = _sample_centered_numpy(grid, levelset, x_center + ox, y, z_center + oz)
            fluid_acc += _smooth_fluid_indicator(phi, thickness)
    return np.clip(fluid_acc / float(samples * samples), 0.0, 1.0).astype(np.float32)


def _w_face_open_fraction_from_levelset_3d(
    grid: GridSpec,
    levelset: np.ndarray,
    thickness: float,
    *,
    samples: int = 3,
) -> np.ndarray:
    assert grid.nz is not None and grid.dz is not None
    x_center = grid.x0 + (np.arange(grid.nx, dtype=np.float32)[:, None, None] + 0.5) * grid.dx
    y_center = grid.y0 + (np.arange(grid.ny, dtype=np.float32)[None, :, None] + 0.5) * grid.dy
    z = grid.z0 + np.arange(grid.nz + 1, dtype=np.float32)[None, None, :] * grid.dz
    fluid_acc = np.zeros(grid.shape_w, dtype=np.float32)
    for ox in _sample_offsets(samples) * grid.dx:
        for oy in _sample_offsets(samples) * grid.dy:
            phi = _sample_centered_numpy(grid, levelset, x_center + ox, y_center + oy, z)
            fluid_acc += _smooth_fluid_indicator(phi, thickness)
    return np.clip(fluid_acc / float(samples * samples), 0.0, 1.0).astype(np.float32)


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
        return (
            _cell_solid_fraction_from_levelset_3d(grid, values, thickness)
            if grid.is_3d
            else _cell_solid_fraction_from_levelset_2d(grid, values, thickness)
        )
    if thickness <= 0.0:
        mask = (values <= 0.0).astype(np.float32)
    else:
        fluid = _smooth_fluid_indicator(values, thickness)
        mask = (fluid < 0.5).astype(np.float32)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def face_masks_from_cell_mask(cell_mask: np.ndarray) -> tuple[np.ndarray, ...]:
    solid = np.clip(np.asarray(cell_mask, dtype=np.float32), 0.0, 1.0)
    fluid = 1.0 - solid
    if solid.ndim == 2:
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
    if solid.ndim == 3:
        nx, ny, nz = solid.shape
        u_mask = np.zeros((nx + 1, ny, nz), dtype=np.float32)
        v_mask = np.zeros((nx, ny + 1, nz), dtype=np.float32)
        w_mask = np.zeros((nx, ny, nz + 1), dtype=np.float32)
        u_mask[0, :, :] = fluid[0, :, :]
        u_mask[nx, :, :] = fluid[nx - 1, :, :]
        u_mask[1:nx, :, :] = np.minimum(fluid[:-1, :, :], fluid[1:, :, :])
        v_mask[:, 0, :] = fluid[:, 0, :]
        v_mask[:, ny, :] = fluid[:, ny - 1, :]
        v_mask[:, 1:ny, :] = np.minimum(fluid[:, :-1, :], fluid[:, 1:, :])
        w_mask[:, :, 0] = fluid[:, :, 0]
        w_mask[:, :, nz] = fluid[:, :, nz - 1]
        w_mask[:, :, 1:nz] = np.minimum(fluid[:, :, :-1], fluid[:, :, 1:])
        return u_mask.astype(np.float32), v_mask.astype(np.float32), w_mask.astype(np.float32)
    raise ValueError(f"Unsupported cell-mask rank {solid.ndim}.")


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
        cell = (
            _cell_solid_fraction_from_levelset_3d(grid, values, thickness)
            if grid.is_3d
            else _cell_solid_fraction_from_levelset_2d(grid, values, thickness)
        )
        if grid.is_3d:
            u_mask = _u_face_open_fraction_from_levelset_3d(grid, values, thickness)
            v_mask = _v_face_open_fraction_from_levelset_3d(grid, values, thickness)
            w_mask = _w_face_open_fraction_from_levelset_3d(grid, values, thickness)
        else:
            u_mask = _u_face_open_fraction_from_levelset_2d(grid, values, thickness)
            v_mask = _v_face_open_fraction_from_levelset_2d(grid, values, thickness)
            w_mask = None
    else:
        cell = cell_mask_from_levelset(values, thickness=thickness, fractional=False)
        masks = face_masks_from_cell_mask(cell)
        u_mask = masks[0]
        v_mask = masks[1]
        w_mask = masks[2] if grid.is_3d else None
    return SolidMask(
        grid=grid,
        cell=wp.array(cell, dtype=wp.float32, device=device),
        u=wp.array(u_mask, dtype=wp.float32, device=device),
        v=wp.array(v_mask, dtype=wp.float32, device=device),
        w=wp.array(w_mask, dtype=wp.float32, device=device) if w_mask is not None else None,
        cell_numpy=cell,
        u_numpy=u_mask,
        v_numpy=v_mask,
        w_numpy=w_mask,
    )
