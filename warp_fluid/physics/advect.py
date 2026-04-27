from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import warp as wp

from ..core.boundary import SolidMask
from ..core.field import CenteredField, MACField
from ..ops.interp import (
    sample_centered,
    sample_centered_3d,
    sample_mac_velocity,
    sample_mac_velocity_3d,
    sample_u_face,
    sample_u_face_3d,
    sample_v_face,
    sample_v_face_3d,
    sample_w_face_3d,
)


@dataclass
class _MACScratchPair:
    first: MACField
    second: MACField


_CENTERED_SCRATCH: dict[tuple[object, str, bool, Optional[int]], CenteredField] = {}
_MAC_SCRATCH: dict[tuple[object, str, bool], _MACScratchPair] = {}


def _centered_requires_grad(field: CenteredField) -> bool:
    return bool(getattr(field.data, "requires_grad", False))


def _mac_requires_grad(velocity: MACField) -> bool:
    return bool(
        getattr(velocity.u, "requires_grad", False)
        or getattr(velocity.v, "requires_grad", False)
        or (velocity.w is not None and getattr(velocity.w, "requires_grad", False))
    )


def _centered_cache_key(field: CenteredField, requires_grad: bool) -> tuple[object, str, bool, Optional[int]]:
    channels_axis = len(field.grid.shape)
    channels = field.data.shape[channels_axis] if len(field.data.shape) == channels_axis + 1 else None
    return (field.grid, str(field.data.device), requires_grad, channels)


def _mac_cache_key(velocity: MACField, requires_grad: bool) -> tuple[object, str, bool]:
    return (velocity.grid, str(velocity.u.device), requires_grad)


def _scratch_centered_field(field: CenteredField, *, requires_grad: bool) -> CenteredField:
    key = _centered_cache_key(field, requires_grad)
    scratch = _CENTERED_SCRATCH.get(key)
    if scratch is None:
        channels_axis = len(field.grid.shape)
        channels = field.data.shape[channels_axis] if len(field.data.shape) == channels_axis + 1 else None
        scratch = CenteredField.zeros(
            field.grid,
            channels=channels,
            device=field.data.device,
            requires_grad=requires_grad,
        )
        _CENTERED_SCRATCH[key] = scratch
    return scratch


def _scratch_mac_field(velocity: MACField, *, requires_grad: bool, avoid: Optional[MACField] = None) -> MACField:
    key = _mac_cache_key(velocity, requires_grad)
    scratch = _MAC_SCRATCH.get(key)
    if scratch is None:
        scratch = _MACScratchPair(
            first=MACField.zeros(velocity.grid, device=velocity.u.device, requires_grad=requires_grad),
            second=MACField.zeros(velocity.grid, device=velocity.u.device, requires_grad=requires_grad),
        )
        _MAC_SCRATCH[key] = scratch
    if avoid is not scratch.first:
        return scratch.first
    return scratch.second


@wp.kernel
def _advect_centered_kernel_2d(
    src: wp.array2d(dtype=wp.float32),
    u: wp.array2d(dtype=wp.float32),
    v: wp.array2d(dtype=wp.float32),
    solid_cell: wp.array2d(dtype=wp.float32),
    dst: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    origin_x: float,
    origin_y: float,
    dx: float,
    dy: float,
    dt: float,
):
    i, j = wp.tid()
    if i >= nx or j >= ny:
        return
    fluid = 1.0 - wp.clamp(solid_cell[i, j], 0.0, 1.0)
    if fluid <= 0.0:
        dst[i, j] = 0.0
        return
    x = origin_x + (float(i) + 0.5) * dx
    y = origin_y + (float(j) + 0.5) * dy
    vel = sample_mac_velocity(u, v, x, y, origin_x, origin_y, nx, ny, dx, dy)
    x0 = wp.clamp(x - dt * vel[0], origin_x, origin_x + float(nx) * dx)
    y0 = wp.clamp(y - dt * vel[1], origin_y, origin_y + float(ny) * dy)
    dst[i, j] = sample_centered(src, x0, y0, origin_x, origin_y, nx, ny, dx, dy) * fluid


@wp.kernel
def _advect_centered_kernel_3d(
    src: wp.array3d(dtype=wp.float32),
    u: wp.array3d(dtype=wp.float32),
    v: wp.array3d(dtype=wp.float32),
    w: wp.array3d(dtype=wp.float32),
    solid_cell: wp.array3d(dtype=wp.float32),
    dst: wp.array3d(dtype=wp.float32),
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
):
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return
    fluid = 1.0 - wp.clamp(solid_cell[i, j, k], 0.0, 1.0)
    if fluid <= 0.0:
        dst[i, j, k] = 0.0
        return
    x = origin_x + (float(i) + 0.5) * dx
    y = origin_y + (float(j) + 0.5) * dy
    z = origin_z + (float(k) + 0.5) * dz
    vel = sample_mac_velocity_3d(u, v, w, x, y, z, origin_x, origin_y, origin_z, nx, ny, nz, dx, dy, dz)
    x0 = wp.clamp(x - dt * vel[0], origin_x, origin_x + float(nx) * dx)
    y0 = wp.clamp(y - dt * vel[1], origin_y, origin_y + float(ny) * dy)
    z0 = wp.clamp(z - dt * vel[2], origin_z, origin_z + float(nz) * dz)
    dst[i, j, k] = sample_centered_3d(src, x0, y0, z0, origin_x, origin_y, origin_z, nx, ny, nz, dx, dy, dz) * fluid


@wp.kernel
def _advect_u_kernel_2d(
    src: wp.array2d(dtype=wp.float32),
    u: wp.array2d(dtype=wp.float32),
    v: wp.array2d(dtype=wp.float32),
    u_mask: wp.array2d(dtype=wp.float32),
    dst: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    origin_x: float,
    origin_y: float,
    dx: float,
    dy: float,
    dt: float,
):
    i, j = wp.tid()
    if i > nx or j >= ny:
        return
    open_frac = wp.clamp(u_mask[i, j], 0.0, 1.0)
    if open_frac <= 0.0:
        dst[i, j] = 0.0
        return
    x = origin_x + float(i) * dx
    y = origin_y + (float(j) + 0.5) * dy
    vel = sample_mac_velocity(u, v, x, y, origin_x, origin_y, nx, ny, dx, dy)
    x0 = wp.clamp(x - dt * vel[0], origin_x, origin_x + float(nx) * dx)
    y0 = wp.clamp(y - dt * vel[1], origin_y, origin_y + float(ny) * dy)
    dst[i, j] = sample_u_face(src, x0, y0, origin_x, origin_y, nx, ny, dx, dy) * open_frac


@wp.kernel
def _advect_v_kernel_2d(
    src: wp.array2d(dtype=wp.float32),
    u: wp.array2d(dtype=wp.float32),
    v: wp.array2d(dtype=wp.float32),
    v_mask: wp.array2d(dtype=wp.float32),
    dst: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    origin_x: float,
    origin_y: float,
    dx: float,
    dy: float,
    dt: float,
):
    i, j = wp.tid()
    if i >= nx or j > ny:
        return
    open_frac = wp.clamp(v_mask[i, j], 0.0, 1.0)
    if open_frac <= 0.0:
        dst[i, j] = 0.0
        return
    x = origin_x + (float(i) + 0.5) * dx
    y = origin_y + float(j) * dy
    vel = sample_mac_velocity(u, v, x, y, origin_x, origin_y, nx, ny, dx, dy)
    x0 = wp.clamp(x - dt * vel[0], origin_x, origin_x + float(nx) * dx)
    y0 = wp.clamp(y - dt * vel[1], origin_y, origin_y + float(ny) * dy)
    dst[i, j] = sample_v_face(src, x0, y0, origin_x, origin_y, nx, ny, dx, dy) * open_frac


@wp.kernel
def _advect_u_kernel_3d(
    src: wp.array3d(dtype=wp.float32),
    u: wp.array3d(dtype=wp.float32),
    v: wp.array3d(dtype=wp.float32),
    w: wp.array3d(dtype=wp.float32),
    u_mask: wp.array3d(dtype=wp.float32),
    dst: wp.array3d(dtype=wp.float32),
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
):
    i, j, k = wp.tid()
    if i > nx or j >= ny or k >= nz:
        return
    open_frac = wp.clamp(u_mask[i, j, k], 0.0, 1.0)
    if open_frac <= 0.0:
        dst[i, j, k] = 0.0
        return
    x = origin_x + float(i) * dx
    y = origin_y + (float(j) + 0.5) * dy
    z = origin_z + (float(k) + 0.5) * dz
    vel = sample_mac_velocity_3d(u, v, w, x, y, z, origin_x, origin_y, origin_z, nx, ny, nz, dx, dy, dz)
    x0 = wp.clamp(x - dt * vel[0], origin_x, origin_x + float(nx) * dx)
    y0 = wp.clamp(y - dt * vel[1], origin_y, origin_y + float(ny) * dy)
    z0 = wp.clamp(z - dt * vel[2], origin_z, origin_z + float(nz) * dz)
    dst[i, j, k] = sample_u_face_3d(src, x0, y0, z0, origin_x, origin_y, origin_z, nx, ny, nz, dx, dy, dz) * open_frac


@wp.kernel
def _advect_v_kernel_3d(
    src: wp.array3d(dtype=wp.float32),
    u: wp.array3d(dtype=wp.float32),
    v: wp.array3d(dtype=wp.float32),
    w: wp.array3d(dtype=wp.float32),
    v_mask: wp.array3d(dtype=wp.float32),
    dst: wp.array3d(dtype=wp.float32),
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
):
    i, j, k = wp.tid()
    if i >= nx or j > ny or k >= nz:
        return
    open_frac = wp.clamp(v_mask[i, j, k], 0.0, 1.0)
    if open_frac <= 0.0:
        dst[i, j, k] = 0.0
        return
    x = origin_x + (float(i) + 0.5) * dx
    y = origin_y + float(j) * dy
    z = origin_z + (float(k) + 0.5) * dz
    vel = sample_mac_velocity_3d(u, v, w, x, y, z, origin_x, origin_y, origin_z, nx, ny, nz, dx, dy, dz)
    x0 = wp.clamp(x - dt * vel[0], origin_x, origin_x + float(nx) * dx)
    y0 = wp.clamp(y - dt * vel[1], origin_y, origin_y + float(ny) * dy)
    z0 = wp.clamp(z - dt * vel[2], origin_z, origin_z + float(nz) * dz)
    dst[i, j, k] = sample_v_face_3d(src, x0, y0, z0, origin_x, origin_y, origin_z, nx, ny, nz, dx, dy, dz) * open_frac


@wp.kernel
def _advect_w_kernel_3d(
    src: wp.array3d(dtype=wp.float32),
    u: wp.array3d(dtype=wp.float32),
    v: wp.array3d(dtype=wp.float32),
    w: wp.array3d(dtype=wp.float32),
    w_mask: wp.array3d(dtype=wp.float32),
    dst: wp.array3d(dtype=wp.float32),
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
):
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k > nz:
        return
    open_frac = wp.clamp(w_mask[i, j, k], 0.0, 1.0)
    if open_frac <= 0.0:
        dst[i, j, k] = 0.0
        return
    x = origin_x + (float(i) + 0.5) * dx
    y = origin_y + (float(j) + 0.5) * dy
    z = origin_z + float(k) * dz
    vel = sample_mac_velocity_3d(u, v, w, x, y, z, origin_x, origin_y, origin_z, nx, ny, nz, dx, dy, dz)
    x0 = wp.clamp(x - dt * vel[0], origin_x, origin_x + float(nx) * dx)
    y0 = wp.clamp(y - dt * vel[1], origin_y, origin_y + float(ny) * dy)
    z0 = wp.clamp(z - dt * vel[2], origin_z, origin_z + float(nz) * dz)
    dst[i, j, k] = sample_w_face_3d(src, x0, y0, z0, origin_x, origin_y, origin_z, nx, ny, nz, dx, dy, dz) * open_frac


@wp.kernel
def _mac_cormack_correct_kernel_2d(
    src: wp.array2d(dtype=wp.float32),
    fwd: wp.array2d(dtype=wp.float32),
    bwd: wp.array2d(dtype=wp.float32),
    solid_cell: wp.array2d(dtype=wp.float32),
    dst: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    strength: float,
):
    i, j = wp.tid()
    if i >= nx or j >= ny:
        return
    fluid = 1.0 - wp.clamp(solid_cell[i, j], 0.0, 1.0)
    if fluid <= 0.0:
        dst[i, j] = 0.0
        return
    dst[i, j] = (fwd[i, j] + 0.5 * strength * (src[i, j] - bwd[i, j])) * fluid


@wp.kernel
def _mac_cormack_correct_kernel_3d(
    src: wp.array3d(dtype=wp.float32),
    fwd: wp.array3d(dtype=wp.float32),
    bwd: wp.array3d(dtype=wp.float32),
    solid_cell: wp.array3d(dtype=wp.float32),
    dst: wp.array3d(dtype=wp.float32),
    nx: int,
    ny: int,
    nz: int,
    strength: float,
):
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return
    fluid = 1.0 - wp.clamp(solid_cell[i, j, k], 0.0, 1.0)
    if fluid <= 0.0:
        dst[i, j, k] = 0.0
        return
    dst[i, j, k] = (fwd[i, j, k] + 0.5 * strength * (src[i, j, k] - bwd[i, j, k])) * fluid


def advect_centered_semi_lagrangian(
    field: CenteredField,
    velocity: MACField,
    dt: float,
    *,
    solid: Optional[SolidMask] = None,
    out: Optional[CenteredField] = None,
) -> CenteredField:
    solid = solid or SolidMask.empty(field.grid, device=field.data.device)
    if out is None:
        out = _scratch_centered_field(
            field,
            requires_grad=_centered_requires_grad(field) or _mac_requires_grad(velocity),
        )
    if field.grid.is_3d:
        assert velocity.w is not None and field.grid.nz is not None and field.grid.dz is not None
        wp.launch(
            _advect_centered_kernel_3d,
            dim=field.grid.shape,
            inputs=(
                field.data,
                velocity.u,
                velocity.v,
                velocity.w,
                solid.cell,
                out.data,
                field.grid.nx,
                field.grid.ny,
                int(field.grid.nz),
                field.grid.x0,
                field.grid.y0,
                field.grid.z0,
                field.grid.dx,
                field.grid.dy,
                field.grid.dz,
                float(dt),
            ),
            device=field.data.device,
        )
        return out
    wp.launch(
        _advect_centered_kernel_2d,
        dim=field.grid.shape,
        inputs=(
            field.data,
            velocity.u,
            velocity.v,
            solid.cell,
            out.data,
            field.grid.nx,
            field.grid.ny,
            field.grid.x0,
            field.grid.y0,
            field.grid.dx,
            field.grid.dy,
            float(dt),
        ),
        device=field.data.device,
    )
    return out


def advect_mac_semi_lagrangian(
    velocity: MACField,
    dt: float,
    *,
    solid: Optional[SolidMask] = None,
    out: Optional[MACField] = None,
) -> MACField:
    solid = solid or SolidMask.empty(velocity.grid, device=velocity.u.device)
    if out is None:
        out = _scratch_mac_field(
            velocity,
            requires_grad=_mac_requires_grad(velocity),
            avoid=velocity,
        )
    if velocity.grid.is_3d:
        assert velocity.w is not None and solid.w is not None and velocity.grid.nz is not None and velocity.grid.dz is not None
        wp.launch(
            _advect_u_kernel_3d,
            dim=velocity.grid.shape_u,
            inputs=(
                velocity.u,
                velocity.u,
                velocity.v,
                velocity.w,
                solid.u,
                out.u,
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
            ),
            device=velocity.u.device,
        )
        wp.launch(
            _advect_v_kernel_3d,
            dim=velocity.grid.shape_v,
            inputs=(
                velocity.v,
                velocity.u,
                velocity.v,
                velocity.w,
                solid.v,
                out.v,
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
            ),
            device=velocity.v.device,
        )
        assert out.w is not None
        wp.launch(
            _advect_w_kernel_3d,
            dim=velocity.grid.shape_w,
            inputs=(
                velocity.w,
                velocity.u,
                velocity.v,
                velocity.w,
                solid.w,
                out.w,
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
            ),
            device=velocity.w.device,
        )
        return out
    wp.launch(
        _advect_u_kernel_2d,
        dim=velocity.grid.shape_u,
        inputs=(
            velocity.u,
            velocity.u,
            velocity.v,
            solid.u,
            out.u,
            velocity.grid.nx,
            velocity.grid.ny,
            velocity.grid.x0,
            velocity.grid.y0,
            velocity.grid.dx,
            velocity.grid.dy,
            float(dt),
        ),
        device=velocity.u.device,
    )
    wp.launch(
        _advect_v_kernel_2d,
        dim=velocity.grid.shape_v,
        inputs=(
            velocity.v,
            velocity.u,
            velocity.v,
            solid.v,
            out.v,
            velocity.grid.nx,
            velocity.grid.ny,
            velocity.grid.x0,
            velocity.grid.y0,
            velocity.grid.dx,
            velocity.grid.dy,
            float(dt),
        ),
        device=velocity.v.device,
    )
    return out


def advect_centered_mac_cormack(
    field: CenteredField,
    velocity: MACField,
    dt: float,
    *,
    solid: Optional[SolidMask] = None,
    correction_strength: float = 1.0,
) -> CenteredField:
    solid = solid or SolidMask.empty(field.grid, device=field.data.device)
    fwd = advect_centered_semi_lagrangian(field, velocity, dt, solid=solid)
    bwd = advect_centered_semi_lagrangian(fwd, velocity, -dt, solid=solid)
    out = CenteredField.zeros(field.grid, device=field.data.device)
    if field.grid.is_3d:
        assert field.grid.nz is not None
        wp.launch(
            _mac_cormack_correct_kernel_3d,
            dim=field.grid.shape,
            inputs=(
                field.data,
                fwd.data,
                bwd.data,
                solid.cell,
                out.data,
                field.grid.nx,
                field.grid.ny,
                int(field.grid.nz),
                float(correction_strength),
            ),
            device=field.data.device,
        )
        return out
    wp.launch(
        _mac_cormack_correct_kernel_2d,
        dim=field.grid.shape,
        inputs=(
            field.data,
            fwd.data,
            bwd.data,
            solid.cell,
            out.data,
            field.grid.nx,
            field.grid.ny,
            float(correction_strength),
        ),
        device=field.data.device,
    )
    return out
