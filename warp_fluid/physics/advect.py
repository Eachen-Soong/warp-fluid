from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import warp as wp

from ..core.boundary import SolidMask
from ..core.field import CenteredField, MACField
from ..ops.interp import sample_centered, sample_mac_velocity, sample_u_face, sample_v_face


@dataclass
class _MACScratchPair:
    first: MACField
    second: MACField


_CENTERED_SCRATCH: dict[tuple[object, str, bool, Optional[int]], CenteredField] = {}
_MAC_SCRATCH: dict[tuple[object, str, bool], _MACScratchPair] = {}


def _centered_requires_grad(field: CenteredField) -> bool:
    return bool(getattr(field.data, "requires_grad", False))


def _mac_requires_grad(velocity: MACField) -> bool:
    return bool(getattr(velocity.u, "requires_grad", False) or getattr(velocity.v, "requires_grad", False))


def _centered_cache_key(field: CenteredField, requires_grad: bool) -> tuple[object, str, bool, Optional[int]]:
    channels = field.data.shape[2] if len(field.data.shape) == 3 else None
    return (field.grid, str(field.data.device), requires_grad, channels)


def _mac_cache_key(velocity: MACField, requires_grad: bool) -> tuple[object, str, bool]:
    return (velocity.grid, str(velocity.u.device), requires_grad)


def _scratch_centered_field(field: CenteredField, *, requires_grad: bool) -> CenteredField:
    key = _centered_cache_key(field, requires_grad)
    scratch = _CENTERED_SCRATCH.get(key)
    if scratch is None:
        channels = field.data.shape[2] if len(field.data.shape) == 3 else None
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
def _advect_centered_kernel(
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
def _advect_u_kernel(
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
def _advect_v_kernel(
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
def _mac_cormack_correct_kernel(
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
    wp.launch(
        _advect_centered_kernel,
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
    wp.launch(
        _advect_u_kernel,
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
        _advect_v_kernel,
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
    wp.launch(
        _mac_cormack_correct_kernel,
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
