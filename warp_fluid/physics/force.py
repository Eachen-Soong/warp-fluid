from __future__ import annotations

from typing import Optional, Tuple

import warp as wp

from ..core.boundary import SolidMask
from ..core.field import CenteredField, MACField
from ..ops.interp import sample_centered


def _mac_requires_grad(velocity: MACField) -> bool:
    return bool(getattr(velocity.u, "requires_grad", False) or getattr(velocity.v, "requires_grad", False))


@wp.kernel
def _diffuse_face_kernel(
    face: wp.array2d(dtype=wp.float32),
    face_out: wp.array2d(dtype=wp.float32),
    face_mask: wp.array2d(dtype=wp.float32),
    i_max: int,
    j_max: int,
    dx: float,
    dy: float,
    dt: float,
    nu: float,
):
    i, j = wp.tid()
    if i > i_max or j > j_max:
        return
    open_frac = wp.clamp(face_mask[i, j], 0.0, 1.0)
    if open_frac <= 0.0:
        face_out[i, j] = 0.0
        return
    i_l = wp.max(i - 1, 0)
    i_r = wp.min(i + 1, i_max)
    j_d = wp.max(j - 1, 0)
    j_u = wp.min(j + 1, j_max)
    f_c = face[i, j]
    f_l = f_c if face_mask[i_l, j] <= 0.0 else face[i_l, j]
    f_r = f_c if face_mask[i_r, j] <= 0.0 else face[i_r, j]
    f_d = f_c if face_mask[i, j_d] <= 0.0 else face[i, j_d]
    f_u = f_c if face_mask[i, j_u] <= 0.0 else face[i, j_u]
    lap_f = (f_l - 2.0 * f_c + f_r) / (dx * dx) + (f_d - 2.0 * f_c + f_u) / (dy * dy)
    face_out[i, j] = (f_c + dt * nu * lap_f) * open_frac


@wp.kernel
def _copy_face_kernel(
    src: wp.array2d(dtype=wp.float32),
    dst: wp.array2d(dtype=wp.float32),
    ni: int,
    nj: int,
):
    i, j = wp.tid()
    if i < ni and j < nj:
        dst[i, j] = src[i, j]


@wp.kernel
def _add_constant_u_kernel(
    u: wp.array2d(dtype=wp.float32),
    u_mask: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    dt: float,
    value: float,
):
    i, j = wp.tid()
    if i <= nx and j < ny and u_mask[i, j] > 0.0:
        u[i, j] = u[i, j] + dt * value * wp.clamp(u_mask[i, j], 0.0, 1.0)


@wp.kernel
def _add_constant_v_kernel(
    v: wp.array2d(dtype=wp.float32),
    v_mask: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    dt: float,
    value: float,
):
    i, j = wp.tid()
    if i < nx and j <= ny and v_mask[i, j] > 0.0:
        v[i, j] = v[i, j] + dt * value * wp.clamp(v_mask[i, j], 0.0, 1.0)


@wp.kernel
def _add_buoyancy_u_kernel(
    u: wp.array2d(dtype=wp.float32),
    density: wp.array2d(dtype=wp.float32),
    u_mask: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    origin_x: float,
    origin_y: float,
    dx: float,
    dy: float,
    dt: float,
    coeff: float,
):
    i, j = wp.tid()
    if i > nx or j >= ny or u_mask[i, j] <= 0.0:
        return
    x = origin_x + float(i) * dx
    y = origin_y + (float(j) + 0.5) * dy
    rho = sample_centered(density, x, y, origin_x, origin_y, nx, ny, dx, dy)
    u[i, j] = u[i, j] + dt * coeff * rho * wp.clamp(u_mask[i, j], 0.0, 1.0)


@wp.kernel
def _add_buoyancy_v_kernel(
    v: wp.array2d(dtype=wp.float32),
    density: wp.array2d(dtype=wp.float32),
    v_mask: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    origin_x: float,
    origin_y: float,
    dx: float,
    dy: float,
    dt: float,
    coeff: float,
):
    i, j = wp.tid()
    if i >= nx or j > ny or v_mask[i, j] <= 0.0:
        return
    x = origin_x + (float(i) + 0.5) * dx
    y = origin_y + float(j) * dy
    rho = sample_centered(density, x, y, origin_x, origin_y, nx, ny, dx, dy)
    v[i, j] = v[i, j] + dt * coeff * rho * wp.clamp(v_mask[i, j], 0.0, 1.0)


def diffuse_velocity_explicit(
    velocity: MACField,
    viscosity: float,
    dt: float,
    *,
    solid: Optional[SolidMask] = None,
    out: Optional[MACField] = None,
) -> MACField:
    requires_grad = _mac_requires_grad(velocity)
    if out is None and not requires_grad:
        out_field = velocity
    else:
        out_field = out or MACField.zeros(
            velocity.grid,
            device=velocity.u.device,
            requires_grad=requires_grad,
        )
    if viscosity == 0.0 or dt == 0.0:
        if out is not None:
            wp.copy(out.u, velocity.u)
            wp.copy(out.v, velocity.v)
            return out
        return out_field
    solid = solid or SolidMask.empty(velocity.grid, device=velocity.u.device)
    wp.launch(
        _diffuse_face_kernel,
        dim=velocity.grid.shape_u,
        inputs=(
            velocity.u,
            out_field.u,
            solid.u,
            velocity.grid.nx,
            velocity.grid.ny - 1,
            velocity.grid.dx,
            velocity.grid.dy,
            float(dt),
            float(viscosity),
        ),
        device=velocity.u.device,
    )
    wp.launch(
        _diffuse_face_kernel,
        dim=velocity.grid.shape_v,
        inputs=(
            velocity.v,
            out_field.v,
            solid.v,
            velocity.grid.nx - 1,
            velocity.grid.ny,
            velocity.grid.dx,
            velocity.grid.dy,
            float(dt),
            float(viscosity),
        ),
        device=velocity.v.device,
    )
    return out_field


def add_constant_force(
    velocity: MACField,
    dt: float,
    *,
    force: Tuple[float, float],
    solid: Optional[SolidMask] = None,
) -> MACField:
    solid = solid or SolidMask.empty(velocity.grid, device=velocity.u.device)
    fx, fy = force
    wp.launch(
        _add_constant_u_kernel,
        dim=velocity.grid.shape_u,
        inputs=(velocity.u, solid.u, velocity.grid.nx, velocity.grid.ny, float(dt), float(fx)),
        device=velocity.u.device,
    )
    wp.launch(
        _add_constant_v_kernel,
        dim=velocity.grid.shape_v,
        inputs=(velocity.v, solid.v, velocity.grid.nx, velocity.grid.ny, float(dt), float(fy)),
        device=velocity.v.device,
    )
    return velocity


def add_buoyancy(
    velocity: MACField,
    density: CenteredField,
    dt: float,
    *,
    factor: Tuple[float, float] = (0.0, 0.1),
    solid: Optional[SolidMask] = None,
) -> MACField:
    solid = solid or SolidMask.empty(velocity.grid, device=velocity.u.device)
    fx, fy = factor
    wp.launch(
        _add_buoyancy_u_kernel,
        dim=velocity.grid.shape_u,
        inputs=(
            velocity.u,
            density.data,
            solid.u,
            velocity.grid.nx,
            velocity.grid.ny,
            velocity.grid.x0,
            velocity.grid.y0,
            velocity.grid.dx,
            velocity.grid.dy,
            float(dt),
            float(fx),
        ),
        device=velocity.u.device,
    )
    wp.launch(
        _add_buoyancy_v_kernel,
        dim=velocity.grid.shape_v,
        inputs=(
            velocity.v,
            density.data,
            solid.v,
            velocity.grid.nx,
            velocity.grid.ny,
            velocity.grid.x0,
            velocity.grid.y0,
            velocity.grid.dx,
            velocity.grid.dy,
            float(dt),
            float(fy),
        ),
        device=velocity.v.device,
    )
    return velocity
