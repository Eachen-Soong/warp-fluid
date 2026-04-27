from __future__ import annotations

from typing import Optional, Tuple

import warp as wp

from ..core.boundary import SolidMask
from ..core.field import CenteredField, MACField
from ..ops.interp import sample_centered, sample_centered_3d


def _mac_requires_grad(velocity: MACField) -> bool:
    return bool(
        getattr(velocity.u, "requires_grad", False)
        or getattr(velocity.v, "requires_grad", False)
        or (velocity.w is not None and getattr(velocity.w, "requires_grad", False))
    )


@wp.kernel
def _diffuse_face_kernel_2d(
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
def _diffuse_face_kernel_3d(
    face: wp.array3d(dtype=wp.float32),
    face_out: wp.array3d(dtype=wp.float32),
    face_mask: wp.array3d(dtype=wp.float32),
    i_max: int,
    j_max: int,
    k_max: int,
    dx: float,
    dy: float,
    dz: float,
    dt: float,
    nu: float,
):
    i, j, k = wp.tid()
    if i > i_max or j > j_max or k > k_max:
        return
    open_frac = wp.clamp(face_mask[i, j, k], 0.0, 1.0)
    if open_frac <= 0.0:
        face_out[i, j, k] = 0.0
        return
    i_l = wp.max(i - 1, 0)
    i_r = wp.min(i + 1, i_max)
    j_d = wp.max(j - 1, 0)
    j_u = wp.min(j + 1, j_max)
    k_b = wp.max(k - 1, 0)
    k_f = wp.min(k + 1, k_max)
    f_c = face[i, j, k]
    f_l = f_c if face_mask[i_l, j, k] <= 0.0 else face[i_l, j, k]
    f_r = f_c if face_mask[i_r, j, k] <= 0.0 else face[i_r, j, k]
    f_d = f_c if face_mask[i, j_d, k] <= 0.0 else face[i, j_d, k]
    f_u = f_c if face_mask[i, j_u, k] <= 0.0 else face[i, j_u, k]
    f_b = f_c if face_mask[i, j, k_b] <= 0.0 else face[i, j, k_b]
    f_f = f_c if face_mask[i, j, k_f] <= 0.0 else face[i, j, k_f]
    lap_f = (
        (f_l - 2.0 * f_c + f_r) / (dx * dx)
        + (f_d - 2.0 * f_c + f_u) / (dy * dy)
        + (f_b - 2.0 * f_c + f_f) / (dz * dz)
    )
    face_out[i, j, k] = (f_c + dt * nu * lap_f) * open_frac


@wp.kernel
def _add_constant_u_kernel_2d(
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
def _add_constant_v_kernel_2d(
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
def _add_constant_u_kernel_3d(
    u: wp.array3d(dtype=wp.float32),
    u_mask: wp.array3d(dtype=wp.float32),
    nx: int,
    ny: int,
    nz: int,
    dt: float,
    value: float,
):
    i, j, k = wp.tid()
    if i <= nx and j < ny and k < nz and u_mask[i, j, k] > 0.0:
        u[i, j, k] = u[i, j, k] + dt * value * wp.clamp(u_mask[i, j, k], 0.0, 1.0)


@wp.kernel
def _add_constant_v_kernel_3d(
    v: wp.array3d(dtype=wp.float32),
    v_mask: wp.array3d(dtype=wp.float32),
    nx: int,
    ny: int,
    nz: int,
    dt: float,
    value: float,
):
    i, j, k = wp.tid()
    if i < nx and j <= ny and k < nz and v_mask[i, j, k] > 0.0:
        v[i, j, k] = v[i, j, k] + dt * value * wp.clamp(v_mask[i, j, k], 0.0, 1.0)


@wp.kernel
def _add_constant_w_kernel_3d(
    w: wp.array3d(dtype=wp.float32),
    w_mask: wp.array3d(dtype=wp.float32),
    nx: int,
    ny: int,
    nz: int,
    dt: float,
    value: float,
):
    i, j, k = wp.tid()
    if i < nx and j < ny and k <= nz and w_mask[i, j, k] > 0.0:
        w[i, j, k] = w[i, j, k] + dt * value * wp.clamp(w_mask[i, j, k], 0.0, 1.0)


@wp.kernel
def _add_buoyancy_u_kernel_2d(
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
def _add_buoyancy_v_kernel_2d(
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


@wp.kernel
def _add_buoyancy_u_kernel_3d(
    u: wp.array3d(dtype=wp.float32),
    density: wp.array3d(dtype=wp.float32),
    u_mask: wp.array3d(dtype=wp.float32),
    nx: int,
    ny: int,
    nz: int,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    dx: float,
    dy: float,
    dz: float,
    dt: float,
    coeff: float,
):
    i, j, k = wp.tid()
    if i > nx or j >= ny or k >= nz or u_mask[i, j, k] <= 0.0:
        return
    x = origin_x + float(i) * dx
    y = origin_y + (float(j) + 0.5) * dy
    z = origin_z + (float(k) + 0.5) * dz
    rho = sample_centered_3d(density, x, y, z, origin_x, origin_y, origin_z, nx, ny, nz, dx, dy, dz)
    u[i, j, k] = u[i, j, k] + dt * coeff * rho * wp.clamp(u_mask[i, j, k], 0.0, 1.0)


@wp.kernel
def _add_buoyancy_v_kernel_3d(
    v: wp.array3d(dtype=wp.float32),
    density: wp.array3d(dtype=wp.float32),
    v_mask: wp.array3d(dtype=wp.float32),
    nx: int,
    ny: int,
    nz: int,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    dx: float,
    dy: float,
    dz: float,
    dt: float,
    coeff: float,
):
    i, j, k = wp.tid()
    if i >= nx or j > ny or k >= nz or v_mask[i, j, k] <= 0.0:
        return
    x = origin_x + (float(i) + 0.5) * dx
    y = origin_y + float(j) * dy
    z = origin_z + (float(k) + 0.5) * dz
    rho = sample_centered_3d(density, x, y, z, origin_x, origin_y, origin_z, nx, ny, nz, dx, dy, dz)
    v[i, j, k] = v[i, j, k] + dt * coeff * rho * wp.clamp(v_mask[i, j, k], 0.0, 1.0)


@wp.kernel
def _add_buoyancy_w_kernel_3d(
    w: wp.array3d(dtype=wp.float32),
    density: wp.array3d(dtype=wp.float32),
    w_mask: wp.array3d(dtype=wp.float32),
    nx: int,
    ny: int,
    nz: int,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    dx: float,
    dy: float,
    dz: float,
    dt: float,
    coeff: float,
):
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k > nz or w_mask[i, j, k] <= 0.0:
        return
    x = origin_x + (float(i) + 0.5) * dx
    y = origin_y + (float(j) + 0.5) * dy
    z = origin_z + float(k) * dz
    rho = sample_centered_3d(density, x, y, z, origin_x, origin_y, origin_z, nx, ny, nz, dx, dy, dz)
    w[i, j, k] = w[i, j, k] + dt * coeff * rho * wp.clamp(w_mask[i, j, k], 0.0, 1.0)


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
            if velocity.w is not None and out.w is not None:
                wp.copy(out.w, velocity.w)
            return out
        return out_field
    solid = solid or SolidMask.empty(velocity.grid, device=velocity.u.device)
    if velocity.grid.is_3d:
        assert velocity.w is not None and out_field.w is not None and solid.w is not None and velocity.grid.nz is not None and velocity.grid.dz is not None
        wp.launch(
            _diffuse_face_kernel_3d,
            dim=velocity.grid.shape_u,
            inputs=(
                velocity.u,
                out_field.u,
                solid.u,
                velocity.grid.nx,
                velocity.grid.ny - 1,
                int(velocity.grid.nz) - 1,
                velocity.grid.dx,
                velocity.grid.dy,
                velocity.grid.dz,
                float(dt),
                float(viscosity),
            ),
            device=velocity.u.device,
        )
        wp.launch(
            _diffuse_face_kernel_3d,
            dim=velocity.grid.shape_v,
            inputs=(
                velocity.v,
                out_field.v,
                solid.v,
                velocity.grid.nx - 1,
                velocity.grid.ny,
                int(velocity.grid.nz) - 1,
                velocity.grid.dx,
                velocity.grid.dy,
                velocity.grid.dz,
                float(dt),
                float(viscosity),
            ),
            device=velocity.v.device,
        )
        wp.launch(
            _diffuse_face_kernel_3d,
            dim=velocity.grid.shape_w,
            inputs=(
                velocity.w,
                out_field.w,
                solid.w,
                velocity.grid.nx - 1,
                velocity.grid.ny - 1,
                int(velocity.grid.nz),
                velocity.grid.dx,
                velocity.grid.dy,
                velocity.grid.dz,
                float(dt),
                float(viscosity),
            ),
            device=velocity.w.device,
        )
        return out_field
    wp.launch(
        _diffuse_face_kernel_2d,
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
        _diffuse_face_kernel_2d,
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
    force: Tuple[float, ...],
    solid: Optional[SolidMask] = None,
) -> MACField:
    solid = solid or SolidMask.empty(velocity.grid, device=velocity.u.device)
    if velocity.grid.is_3d:
        if len(force) != 3:
            raise ValueError("3D constant force must provide three components.")
        fx, fy, fz = force
        assert velocity.w is not None and solid.w is not None and velocity.grid.nz is not None
        wp.launch(
            _add_constant_u_kernel_3d,
            dim=velocity.grid.shape_u,
            inputs=(velocity.u, solid.u, velocity.grid.nx, velocity.grid.ny, int(velocity.grid.nz), float(dt), float(fx)),
            device=velocity.u.device,
        )
        wp.launch(
            _add_constant_v_kernel_3d,
            dim=velocity.grid.shape_v,
            inputs=(velocity.v, solid.v, velocity.grid.nx, velocity.grid.ny, int(velocity.grid.nz), float(dt), float(fy)),
            device=velocity.v.device,
        )
        wp.launch(
            _add_constant_w_kernel_3d,
            dim=velocity.grid.shape_w,
            inputs=(velocity.w, solid.w, velocity.grid.nx, velocity.grid.ny, int(velocity.grid.nz), float(dt), float(fz)),
            device=velocity.w.device,
        )
        return velocity
    fx, fy = force
    wp.launch(
        _add_constant_u_kernel_2d,
        dim=velocity.grid.shape_u,
        inputs=(velocity.u, solid.u, velocity.grid.nx, velocity.grid.ny, float(dt), float(fx)),
        device=velocity.u.device,
    )
    wp.launch(
        _add_constant_v_kernel_2d,
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
    factor: Tuple[float, ...] = (0.0, 0.1),
    solid: Optional[SolidMask] = None,
) -> MACField:
    solid = solid or SolidMask.empty(velocity.grid, device=velocity.u.device)
    if velocity.grid.is_3d:
        if len(factor) != 3:
            raise ValueError("3D buoyancy factor must provide three components.")
        fx, fy, fz = factor
        assert velocity.w is not None and solid.w is not None and velocity.grid.nz is not None and velocity.grid.dz is not None
        wp.launch(
            _add_buoyancy_u_kernel_3d,
            dim=velocity.grid.shape_u,
            inputs=(
                velocity.u,
                density.data,
                solid.u,
                velocity.grid.nx,
                velocity.grid.ny,
                int(velocity.grid.nz),
                velocity.grid.x0,
                velocity.grid.y0,
                velocity.grid.z0,
                velocity.grid.dx,
                velocity.grid.dy,
                velocity.grid.dz,
                float(dt),
                float(fx),
            ),
            device=velocity.u.device,
        )
        wp.launch(
            _add_buoyancy_v_kernel_3d,
            dim=velocity.grid.shape_v,
            inputs=(
                velocity.v,
                density.data,
                solid.v,
                velocity.grid.nx,
                velocity.grid.ny,
                int(velocity.grid.nz),
                velocity.grid.x0,
                velocity.grid.y0,
                velocity.grid.z0,
                velocity.grid.dx,
                velocity.grid.dy,
                velocity.grid.dz,
                float(dt),
                float(fy),
            ),
            device=velocity.v.device,
        )
        wp.launch(
            _add_buoyancy_w_kernel_3d,
            dim=velocity.grid.shape_w,
            inputs=(
                velocity.w,
                density.data,
                solid.w,
                velocity.grid.nx,
                velocity.grid.ny,
                int(velocity.grid.nz),
                velocity.grid.x0,
                velocity.grid.y0,
                velocity.grid.z0,
                velocity.grid.dx,
                velocity.grid.dy,
                velocity.grid.dz,
                float(dt),
                float(fz),
            ),
            device=velocity.w.device,
        )
        return velocity
    fx, fy = factor
    wp.launch(
        _add_buoyancy_u_kernel_2d,
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
        _add_buoyancy_v_kernel_2d,
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
