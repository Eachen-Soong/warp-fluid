from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import warp as wp

from .field import MACField
from .grid import GridSpec


@dataclass(frozen=True)
class VelocityBoundary:
    """Simple domain velocity boundary conditions."""

    inflow_west: Optional[Tuple[float, float]] = None
    copy_east_outflow: bool = True
    no_slip_south: bool = True
    no_slip_north: bool = True


@dataclass
class SolidMask:
    """Cell and face masks describing solid regions."""

    grid: GridSpec
    cell: object
    u: object
    v: object
    cell_numpy: Optional[np.ndarray] = None
    u_numpy: Optional[np.ndarray] = None
    v_numpy: Optional[np.ndarray] = None
    fluid_cell: Optional[object] = None
    fluid_cell_numpy_cache: Optional[np.ndarray] = None

    @classmethod
    def empty(cls, grid: GridSpec, *, device: Optional[str] = None) -> "SolidMask":
        device = device or "cpu"
        cell = np.zeros(grid.shape, dtype=np.float32)
        u = np.ones(grid.shape_u, dtype=np.float32)
        v = np.ones(grid.shape_v, dtype=np.float32)
        return cls(
            grid=grid,
            cell=wp.array(cell, dtype=wp.float32, device=device),
            u=wp.array(u, dtype=wp.float32, device=device),
            v=wp.array(v, dtype=wp.float32, device=device),
            cell_numpy=cell,
            u_numpy=u,
            v_numpy=v,
            fluid_cell=wp.array(np.ones(grid.shape, dtype=np.float32), dtype=wp.float32, device=device),
            fluid_cell_numpy_cache=np.ones(grid.shape, dtype=np.float32),
        )

    def fluid_cell_mask(self) -> object:
        if self.fluid_cell is not None:
            return self.fluid_cell
        if self.cell_numpy is not None:
            fluid = 1.0 - np.asarray(self.cell_numpy, dtype=np.float32)
            self.fluid_cell_numpy_cache = fluid
            self.fluid_cell = wp.array(fluid, dtype=wp.float32, device=self.cell.device)
            return self.fluid_cell
        self.fluid_cell = 1.0 - self.cell
        return self.fluid_cell

    def fluid_cell_numpy(self) -> Optional[np.ndarray]:
        if self.fluid_cell_numpy_cache is not None:
            return self.fluid_cell_numpy_cache
        if self.cell_numpy is None:
            return None
        self.fluid_cell_numpy_cache = 1.0 - np.asarray(self.cell_numpy, dtype=np.float32)
        return self.fluid_cell_numpy_cache


@wp.kernel
def _apply_inflow_u_kernel(
    u: wp.array2d(dtype=wp.float32),
    u_mask: wp.array2d(dtype=wp.float32),
    j_max: int,
    value: float,
):
    j = wp.tid()
    if j <= j_max and u_mask[0, j] > 0.0:
        u[0, j] = value * u_mask[0, j]


@wp.kernel
def _apply_inflow_v_kernel(
    v: wp.array2d(dtype=wp.float32),
    v_mask: wp.array2d(dtype=wp.float32),
    j_max: int,
    value: float,
):
    j = wp.tid()
    if j <= j_max and v_mask[0, j] > 0.0:
        v[0, j] = value * v_mask[0, j]


@wp.kernel
def _apply_outflow_u_kernel(
    u: wp.array2d(dtype=wp.float32),
    u_mask: wp.array2d(dtype=wp.float32),
    i_out: int,
    i_src: int,
    j_max: int,
):
    j = wp.tid()
    if i_out >= 0 and i_src >= 0 and j <= j_max and u_mask[i_out, j] > 0.0:
        u[i_out, j] = u[i_src, j] * u_mask[i_out, j]


@wp.kernel
def _apply_outflow_v_kernel(
    v: wp.array2d(dtype=wp.float32),
    v_mask: wp.array2d(dtype=wp.float32),
    i_out: int,
    i_src: int,
    j_max: int,
):
    j = wp.tid()
    if i_out >= 0 and i_src >= 0 and j <= j_max and v_mask[i_out, j] > 0.0:
        v[i_out, j] = v[i_src, j] * v_mask[i_out, j]


@wp.kernel
def _apply_wall_v_kernel(
    v: wp.array2d(dtype=wp.float32),
    v_mask: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    zero_bottom: int,
    zero_top: int,
):
    i = wp.tid()
    if i >= nx:
        return
    if zero_bottom != 0 and v_mask[i, 0] > 0.0:
        v[i, 0] = 0.0
    if zero_top != 0 and v_mask[i, ny] > 0.0:
        v[i, ny] = 0.0


@wp.kernel
def _apply_face_mask_kernel(
    face: wp.array2d(dtype=wp.float32),
    face_mask: wp.array2d(dtype=wp.float32),
    ni: int,
    nj: int,
):
    i, j = wp.tid()
    if i < ni and j < nj:
        face[i, j] = face[i, j] * wp.clamp(face_mask[i, j], 0.0, 1.0)


def apply_velocity_boundary(
    velocity: MACField,
    *,
    boundary: Optional[VelocityBoundary] = None,
    solid: Optional[SolidMask] = None,
) -> MACField:
    boundary = boundary or VelocityBoundary()
    solid = solid or SolidMask.empty(velocity.grid, device=velocity.u.device)
    grid = velocity.grid
    if boundary.inflow_west is not None:
        u_in, v_in = boundary.inflow_west
        wp.launch(
            _apply_inflow_u_kernel,
            dim=grid.ny,
            inputs=(velocity.u, solid.u, grid.ny - 1, float(u_in)),
            device=velocity.u.device,
        )
        wp.launch(
            _apply_inflow_v_kernel,
            dim=grid.ny + 1,
            inputs=(velocity.v, solid.v, grid.ny, float(v_in)),
            device=velocity.v.device,
        )
    if boundary.copy_east_outflow:
        wp.launch(
            _apply_outflow_u_kernel,
            dim=grid.ny,
            inputs=(velocity.u, solid.u, grid.nx, grid.nx - 1, grid.ny - 1),
            device=velocity.u.device,
        )
        wp.launch(
            _apply_outflow_v_kernel,
            dim=grid.ny + 1,
            inputs=(velocity.v, solid.v, grid.nx - 1, grid.nx - 2, grid.ny),
            device=velocity.v.device,
        )
    wp.launch(
        _apply_wall_v_kernel,
        dim=grid.nx,
        inputs=(
            velocity.v,
            solid.v,
            grid.nx,
            grid.ny,
            int(boundary.no_slip_south),
            int(boundary.no_slip_north),
        ),
        device=velocity.v.device,
    )
    wp.launch(
        _apply_face_mask_kernel,
        dim=grid.shape_u,
        inputs=(velocity.u, solid.u, grid.shape_u[0], grid.shape_u[1]),
        device=velocity.u.device,
    )
    wp.launch(
        _apply_face_mask_kernel,
        dim=grid.shape_v,
        inputs=(velocity.v, solid.v, grid.shape_v[0], grid.shape_v[1]),
        device=velocity.v.device,
    )
    return velocity
