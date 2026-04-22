from __future__ import annotations

from typing import Optional

import warp as wp

from ..core.boundary import SolidMask
from ..core.field import CenteredField, MACField


@wp.kernel
def _cell_center_velocity_kernel(
    u: wp.array2d(dtype=wp.float32),
    v: wp.array2d(dtype=wp.float32),
    solid_cell: wp.array2d(dtype=wp.float32),
    u_mask: wp.array2d(dtype=wp.float32),
    v_mask: wp.array2d(dtype=wp.float32),
    vel: wp.array3d(dtype=wp.float32),
    nx: int,
    ny: int,
):
    i, j = wp.tid()
    if i >= nx or j >= ny:
        return
    fluid = 1.0 - wp.clamp(solid_cell[i, j], 0.0, 1.0)
    if fluid <= 1.0e-6:
        vel[i, j, 0] = 0.0
        vel[i, j, 1] = 0.0
        return
    u_l_w = wp.clamp(u_mask[i, j], 0.0, 1.0)
    u_r_w = wp.clamp(u_mask[i + 1, j], 0.0, 1.0)
    v_b_w = wp.clamp(v_mask[i, j], 0.0, 1.0)
    v_t_w = wp.clamp(v_mask[i, j + 1], 0.0, 1.0)
    u_denom = wp.max(u_l_w + u_r_w, 1.0e-6)
    v_denom = wp.max(v_b_w + v_t_w, 1.0e-6)
    vel[i, j, 0] = (u_l_w * u[i, j] + u_r_w * u[i + 1, j]) / u_denom
    vel[i, j, 1] = (v_b_w * v[i, j] + v_t_w * v[i, j + 1]) / v_denom


@wp.kernel
def _divergence_mac_kernel(
    u: wp.array2d(dtype=wp.float32),
    v: wp.array2d(dtype=wp.float32),
    solid_cell: wp.array2d(dtype=wp.float32),
    u_mask: wp.array2d(dtype=wp.float32),
    v_mask: wp.array2d(dtype=wp.float32),
    div: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    dx: float,
    dy: float,
):
    i, j = wp.tid()
    if i >= nx or j >= ny:
        return
    fluid = 1.0 - wp.clamp(solid_cell[i, j], 0.0, 1.0)
    if fluid <= 1.0e-6:
        div[i, j] = 0.0
        return
    u_l = u[i, j] * wp.clamp(u_mask[i, j], 0.0, 1.0)
    u_r = u[i + 1, j] * wp.clamp(u_mask[i + 1, j], 0.0, 1.0)
    v_b = v[i, j] * wp.clamp(v_mask[i, j], 0.0, 1.0)
    v_t = v[i, j + 1] * wp.clamp(v_mask[i, j + 1], 0.0, 1.0)
    div[i, j] = ((u_r - u_l) / dx + (v_t - v_b) / dy) / fluid


@wp.kernel
def _laplace_centered_kernel(
    src: wp.array2d(dtype=wp.float32),
    solid_cell: wp.array2d(dtype=wp.float32),
    u_mask: wp.array2d(dtype=wp.float32),
    v_mask: wp.array2d(dtype=wp.float32),
    dst: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    dx: float,
    dy: float,
):
    i, j = wp.tid()
    if i >= nx or j >= ny:
        return
    fluid = 1.0 - wp.clamp(solid_cell[i, j], 0.0, 1.0)
    if fluid <= 1.0e-6:
        dst[i, j] = 0.0
        return
    i_l = wp.max(i - 1, 0)
    i_r = wp.min(i + 1, nx - 1)
    j_d = wp.max(j - 1, 0)
    j_u = wp.min(j + 1, ny - 1)
    c = src[i, j]
    w_l = wp.clamp(u_mask[i, j], 0.0, 1.0)
    w_r = wp.clamp(u_mask[i + 1, j], 0.0, 1.0)
    w_d = wp.clamp(v_mask[i, j], 0.0, 1.0)
    w_u = wp.clamp(v_mask[i, j + 1], 0.0, 1.0)
    l = c if i == 0 or (1.0 - wp.clamp(solid_cell[i_l, j], 0.0, 1.0)) <= 1.0e-6 else src[i_l, j]
    r = c if i == nx - 1 or (1.0 - wp.clamp(solid_cell[i_r, j], 0.0, 1.0)) <= 1.0e-6 else src[i_r, j]
    d = c if j == 0 or (1.0 - wp.clamp(solid_cell[i, j_d], 0.0, 1.0)) <= 1.0e-6 else src[i, j_d]
    u_ = c if j == ny - 1 or (1.0 - wp.clamp(solid_cell[i, j_u], 0.0, 1.0)) <= 1.0e-6 else src[i, j_u]
    dst[i, j] = (
        (w_r * (r - c) - w_l * (c - l)) / (dx * dx)
        + (w_u * (u_ - c) - w_d * (c - d)) / (dy * dy)
    ) / fluid


def _solid_or_empty(velocity_or_field, solid: Optional[SolidMask]) -> SolidMask:
    if solid is not None:
        return solid
    grid = velocity_or_field.grid
    return SolidMask.empty(grid, device=velocity_or_field.u.device if hasattr(velocity_or_field, "u") else velocity_or_field.data.device)


def cell_center_velocity(
    velocity: MACField,
    *,
    solid: Optional[SolidMask] = None,
    out: Optional[CenteredField] = None,
) -> CenteredField:
    solid = _solid_or_empty(velocity, solid)
    if out is None:
        out = CenteredField.zeros(velocity.grid, channels=2, device=velocity.u.device)
    wp.launch(
        _cell_center_velocity_kernel,
        dim=velocity.grid.shape,
        inputs=(velocity.u, velocity.v, solid.cell, solid.u, solid.v, out.data, velocity.grid.nx, velocity.grid.ny),
        device=velocity.u.device,
    )
    return out


def divergence(
    velocity: MACField,
    *,
    solid: Optional[SolidMask] = None,
    out: Optional[CenteredField] = None,
) -> CenteredField:
    solid = _solid_or_empty(velocity, solid)
    if out is None:
        out = CenteredField.zeros(velocity.grid, device=velocity.u.device)
    wp.launch(
        _divergence_mac_kernel,
        dim=velocity.grid.shape,
        inputs=(
            velocity.u,
            velocity.v,
            solid.cell,
            solid.u,
            solid.v,
            out.data,
            velocity.grid.nx,
            velocity.grid.ny,
            velocity.grid.dx,
            velocity.grid.dy,
        ),
        device=velocity.u.device,
    )
    return out


def laplace_centered(
    field: CenteredField,
    *,
    solid: Optional[SolidMask] = None,
    out: Optional[CenteredField] = None,
) -> CenteredField:
    solid = solid or SolidMask.empty(field.grid, device=field.data.device)
    if out is None:
        out = CenteredField.zeros(field.grid, device=field.data.device)
    wp.launch(
        _laplace_centered_kernel,
        dim=field.grid.shape,
        inputs=(
            field.data,
            solid.cell,
            solid.u,
            solid.v,
            out.data,
            field.grid.nx,
            field.grid.ny,
            field.grid.dx,
            field.grid.dy,
        ),
        device=field.data.device,
    )
    return out
