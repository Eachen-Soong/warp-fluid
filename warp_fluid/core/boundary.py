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

    inflow_west: Optional[Tuple[float, ...]] = None
    copy_east_outflow: bool = True
    no_slip_south: bool = True
    no_slip_north: bool = True
    no_slip_bottom: bool = True
    no_slip_top: bool = True


@dataclass
class SolidMask:
    """Cell and face masks describing solid regions."""

    grid: GridSpec
    cell: object
    u: object
    v: object
    w: Optional[object] = None
    cell_numpy: Optional[np.ndarray] = None
    u_numpy: Optional[np.ndarray] = None
    v_numpy: Optional[np.ndarray] = None
    w_numpy: Optional[np.ndarray] = None
    fluid_cell: Optional[object] = None
    fluid_cell_numpy_cache: Optional[np.ndarray] = None

    @classmethod
    def empty(cls, grid: GridSpec, *, device: Optional[str] = None) -> "SolidMask":
        device = device or "cpu"
        cell = np.zeros(grid.shape, dtype=np.float32)
        u = np.ones(grid.shape_u, dtype=np.float32)
        v = np.ones(grid.shape_v, dtype=np.float32)
        w = np.ones(grid.shape_w, dtype=np.float32) if grid.is_3d else None
        return cls(
            grid=grid,
            cell=wp.array(cell, dtype=wp.float32, device=device),
            u=wp.array(u, dtype=wp.float32, device=device),
            v=wp.array(v, dtype=wp.float32, device=device),
            w=wp.array(w, dtype=wp.float32, device=device) if w is not None else None,
            cell_numpy=cell,
            u_numpy=u,
            v_numpy=v,
            w_numpy=w,
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


@wp.kernel
def _apply_inflow_u_kernel_3d(
    u: wp.array3d(dtype=wp.float32),
    u_mask: wp.array3d(dtype=wp.float32),
    ny: int,
    nz: int,
    value: float,
):
    j, k = wp.tid()
    if j < ny and k < nz and u_mask[0, j, k] > 0.0:
        u[0, j, k] = value * u_mask[0, j, k]


@wp.kernel
def _apply_inflow_v_kernel_3d(
    v: wp.array3d(dtype=wp.float32),
    v_mask: wp.array3d(dtype=wp.float32),
    ny: int,
    nz: int,
    value: float,
):
    j, k = wp.tid()
    if j <= ny and k < nz and v_mask[0, j, k] > 0.0:
        v[0, j, k] = value * v_mask[0, j, k]


@wp.kernel
def _apply_inflow_w_kernel_3d(
    w: wp.array3d(dtype=wp.float32),
    w_mask: wp.array3d(dtype=wp.float32),
    ny: int,
    nz: int,
    value: float,
):
    j, k = wp.tid()
    if j < ny and k <= nz and w_mask[0, j, k] > 0.0:
        w[0, j, k] = value * w_mask[0, j, k]


@wp.kernel
def _apply_outflow_u_kernel_3d(
    u: wp.array3d(dtype=wp.float32),
    u_mask: wp.array3d(dtype=wp.float32),
    i_out: int,
    i_src: int,
    ny: int,
    nz: int,
):
    j, k = wp.tid()
    if i_out >= 0 and i_src >= 0 and j < ny and k < nz and u_mask[i_out, j, k] > 0.0:
        u[i_out, j, k] = u[i_src, j, k] * u_mask[i_out, j, k]


@wp.kernel
def _apply_outflow_v_kernel_3d(
    v: wp.array3d(dtype=wp.float32),
    v_mask: wp.array3d(dtype=wp.float32),
    i_out: int,
    i_src: int,
    ny: int,
    nz: int,
):
    j, k = wp.tid()
    if i_out >= 0 and i_src >= 0 and j <= ny and k < nz and v_mask[i_out, j, k] > 0.0:
        v[i_out, j, k] = v[i_src, j, k] * v_mask[i_out, j, k]


@wp.kernel
def _apply_outflow_w_kernel_3d(
    w: wp.array3d(dtype=wp.float32),
    w_mask: wp.array3d(dtype=wp.float32),
    i_out: int,
    i_src: int,
    ny: int,
    nz: int,
):
    j, k = wp.tid()
    if i_out >= 0 and i_src >= 0 and j < ny and k <= nz and w_mask[i_out, j, k] > 0.0:
        w[i_out, j, k] = w[i_src, j, k] * w_mask[i_out, j, k]


@wp.kernel
def _apply_wall_v_kernel_3d(
    v: wp.array3d(dtype=wp.float32),
    v_mask: wp.array3d(dtype=wp.float32),
    nx: int,
    ny: int,
    nz: int,
    zero_south: int,
    zero_north: int,
):
    i, k = wp.tid()
    if i >= nx or k >= nz:
        return
    if zero_south != 0 and v_mask[i, 0, k] > 0.0:
        v[i, 0, k] = 0.0
    if zero_north != 0 and v_mask[i, ny, k] > 0.0:
        v[i, ny, k] = 0.0


@wp.kernel
def _apply_wall_w_kernel_3d(
    w: wp.array3d(dtype=wp.float32),
    w_mask: wp.array3d(dtype=wp.float32),
    nx: int,
    ny: int,
    nz: int,
    zero_bottom: int,
    zero_top: int,
):
    i, j = wp.tid()
    if i >= nx or j >= ny:
        return
    if zero_bottom != 0 and w_mask[i, j, 0] > 0.0:
        w[i, j, 0] = 0.0
    if zero_top != 0 and w_mask[i, j, nz] > 0.0:
        w[i, j, nz] = 0.0


@wp.kernel
def _apply_face_mask_kernel_3d(
    face: wp.array3d(dtype=wp.float32),
    face_mask: wp.array3d(dtype=wp.float32),
    ni: int,
    nj: int,
    nk: int,
):
    i, j, k = wp.tid()
    if i < ni and j < nj and k < nk:
        face[i, j, k] = face[i, j, k] * wp.clamp(face_mask[i, j, k], 0.0, 1.0)


def apply_velocity_boundary(
    velocity: MACField,
    *,
    boundary: Optional[VelocityBoundary] = None,
    solid: Optional[SolidMask] = None,
) -> MACField:
    boundary = boundary or VelocityBoundary()
    solid = solid or SolidMask.empty(velocity.grid, device=velocity.u.device)
    grid = velocity.grid
    if grid.is_3d:
        inflow = tuple(float(v) for v in (boundary.inflow_west or ()))
        if inflow:
            if len(inflow) != 3:
                raise ValueError("3D inflow_west must provide three velocity components.")
            wp.launch(
                _apply_inflow_u_kernel_3d,
                dim=(grid.ny, int(grid.nz)),
                inputs=(velocity.u, solid.u, grid.ny, int(grid.nz), inflow[0]),
                device=velocity.u.device,
            )
            wp.launch(
                _apply_inflow_v_kernel_3d,
                dim=(grid.ny + 1, int(grid.nz)),
                inputs=(velocity.v, solid.v, grid.ny, int(grid.nz), inflow[1]),
                device=velocity.v.device,
            )
            assert velocity.w is not None and solid.w is not None
            wp.launch(
                _apply_inflow_w_kernel_3d,
                dim=(grid.ny, int(grid.nz) + 1),
                inputs=(velocity.w, solid.w, grid.ny, int(grid.nz), inflow[2]),
                device=velocity.w.device,
            )
        if boundary.copy_east_outflow:
            wp.launch(
                _apply_outflow_u_kernel_3d,
                dim=(grid.ny, int(grid.nz)),
                inputs=(velocity.u, solid.u, grid.nx, grid.nx - 1, grid.ny, int(grid.nz)),
                device=velocity.u.device,
            )
            wp.launch(
                _apply_outflow_v_kernel_3d,
                dim=(grid.ny + 1, int(grid.nz)),
                inputs=(velocity.v, solid.v, grid.nx - 1, grid.nx - 2, grid.ny, int(grid.nz)),
                device=velocity.v.device,
            )
            assert velocity.w is not None and solid.w is not None
            wp.launch(
                _apply_outflow_w_kernel_3d,
                dim=(grid.ny, int(grid.nz) + 1),
                inputs=(velocity.w, solid.w, grid.nx - 1, grid.nx - 2, grid.ny, int(grid.nz)),
                device=velocity.w.device,
            )
        wp.launch(
            _apply_wall_v_kernel_3d,
            dim=(grid.nx, int(grid.nz)),
            inputs=(
                velocity.v,
                solid.v,
                grid.nx,
                grid.ny,
                int(grid.nz),
                int(boundary.no_slip_south),
                int(boundary.no_slip_north),
            ),
            device=velocity.v.device,
        )
        assert velocity.w is not None and solid.w is not None
        wp.launch(
            _apply_wall_w_kernel_3d,
            dim=(grid.nx, grid.ny),
            inputs=(
                velocity.w,
                solid.w,
                grid.nx,
                grid.ny,
                int(grid.nz),
                int(boundary.no_slip_bottom),
                int(boundary.no_slip_top),
            ),
            device=velocity.w.device,
        )
        wp.launch(
            _apply_face_mask_kernel_3d,
            dim=grid.shape_u,
            inputs=(velocity.u, solid.u, grid.shape_u[0], grid.shape_u[1], grid.shape_u[2]),
            device=velocity.u.device,
        )
        wp.launch(
            _apply_face_mask_kernel_3d,
            dim=grid.shape_v,
            inputs=(velocity.v, solid.v, grid.shape_v[0], grid.shape_v[1], grid.shape_v[2]),
            device=velocity.v.device,
        )
        wp.launch(
            _apply_face_mask_kernel_3d,
            dim=grid.shape_w,
            inputs=(velocity.w, solid.w, grid.shape_w[0], grid.shape_w[1], grid.shape_w[2]),
            device=velocity.w.device,
        )
        return velocity
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
