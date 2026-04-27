from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import warp as wp

from ..core.boundary import SolidMask, VelocityBoundary, apply_velocity_boundary
from ..core.field import CenteredField, MACField
from ..ops.diff import divergence, laplace_centered
from ..solver.optimize import (
    LinearSolveConfig,
    LinearSolveStats,
    _jacobi_update_kernel,
    _jacobi_update_kernel_3d,
    solve_linear,
)


@dataclass
class _BalanceScratch:
    sum_out: object
    count_out: object
    mean_out: object


@dataclass
class _JacobiPressureScratch:
    rhs: CenteredField
    pressure_a: CenteredField
    pressure_b: CenteredField
    diagonal: CenteredField
    ax: CenteredField


_BALANCE_SCRATCH: dict[str, _BalanceScratch] = {}
_JACOBI_PRESSURE_SCRATCH: dict[tuple[object, str], _JacobiPressureScratch] = {}


@wp.kernel
def _sum_active_kernel_2d(
    values: wp.array2d(dtype=wp.float32),
    active: wp.array2d(dtype=wp.float32),
    sum_out: wp.array(dtype=wp.float32),
    count_out: wp.array(dtype=wp.float32),
    nx: int,
    ny: int,
):
    i, j = wp.tid()
    if i < nx and j < ny and active[i, j] > 0.0:
        wp.atomic_add(sum_out, 0, values[i, j])
        wp.atomic_add(count_out, 0, 1.0)


@wp.kernel
def _sum_active_kernel_3d(
    values: wp.array3d(dtype=wp.float32),
    active: wp.array3d(dtype=wp.float32),
    sum_out: wp.array(dtype=wp.float32),
    count_out: wp.array(dtype=wp.float32),
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    if i < nx and j < ny and k < nz and active[i, j, k] > 0.0:
        wp.atomic_add(sum_out, 0, values[i, j, k])
        wp.atomic_add(count_out, 0, 1.0)


@wp.kernel
def _compute_mean_kernel(
    sum_out: wp.array(dtype=wp.float32),
    count_out: wp.array(dtype=wp.float32),
    mean_out: wp.array(dtype=wp.float32),
):
    if wp.tid() == 0:
        denom = wp.max(count_out[0], 1.0)
        mean_out[0] = sum_out[0] / denom


@wp.kernel
def _fill_scalar_kernel(
    out: wp.array(dtype=wp.float32),
    value: float,
):
    if wp.tid() == 0:
        out[0] = value


@wp.kernel
def _fill_centered_kernel_2d(
    out: wp.array2d(dtype=wp.float32),
    value: float,
    nx: int,
    ny: int,
):
    i, j = wp.tid()
    if i < nx and j < ny:
        out[i, j] = value


@wp.kernel
def _fill_centered_kernel_3d(
    out: wp.array3d(dtype=wp.float32),
    value: float,
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    if i < nx and j < ny and k < nz:
        out[i, j, k] = value


@wp.kernel
def _subtract_mean_kernel_2d(
    values: wp.array2d(dtype=wp.float32),
    active: wp.array2d(dtype=wp.float32),
    mean: wp.array(dtype=wp.float32),
    nx: int,
    ny: int,
):
    i, j = wp.tid()
    if i < nx and j < ny and active[i, j] > 0.0:
        values[i, j] = values[i, j] - mean[0]


@wp.kernel
def _subtract_mean_kernel_3d(
    values: wp.array3d(dtype=wp.float32),
    active: wp.array3d(dtype=wp.float32),
    mean: wp.array(dtype=wp.float32),
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    if i < nx and j < ny and k < nz and active[i, j, k] > 0.0:
        values[i, j, k] = values[i, j, k] - mean[0]


@wp.kernel
def _project_u_kernel_2d(
    u: wp.array2d(dtype=wp.float32),
    p: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    u_mask: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    dx: float,
):
    i, j = wp.tid()
    if i > nx or j >= ny:
        return
    open_frac = wp.clamp(u_mask[i, j], 0.0, 1.0)
    if open_frac <= 0.0:
        u[i, j] = 0.0
        return
    if i == 0 or i == nx:
        u[i, j] = u[i, j] * open_frac
        return
    if fluid[i - 1, j] <= 1.0e-6 or fluid[i, j] <= 1.0e-6:
        u[i, j] = u[i, j] * open_frac
        return
    u[i, j] = (u[i, j] - (p[i, j] - p[i - 1, j]) / dx) * open_frac


@wp.kernel
def _project_v_kernel_2d(
    v: wp.array2d(dtype=wp.float32),
    p: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    v_mask: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    dy: float,
):
    i, j = wp.tid()
    if i >= nx or j > ny:
        return
    open_frac = wp.clamp(v_mask[i, j], 0.0, 1.0)
    if open_frac <= 0.0:
        v[i, j] = 0.0
        return
    if j == 0 or j == ny:
        v[i, j] = v[i, j] * open_frac
        return
    if fluid[i, j - 1] <= 1.0e-6 or fluid[i, j] <= 1.0e-6:
        v[i, j] = v[i, j] * open_frac
        return
    v[i, j] = (v[i, j] - (p[i, j] - p[i, j - 1]) / dy) * open_frac


@wp.kernel
def _project_u_kernel_3d(
    u: wp.array3d(dtype=wp.float32),
    p: wp.array3d(dtype=wp.float32),
    fluid: wp.array3d(dtype=wp.float32),
    u_mask: wp.array3d(dtype=wp.float32),
    nx: int,
    ny: int,
    nz: int,
    dx: float,
):
    i, j, k = wp.tid()
    if i > nx or j >= ny or k >= nz:
        return
    open_frac = wp.clamp(u_mask[i, j, k], 0.0, 1.0)
    if open_frac <= 0.0:
        u[i, j, k] = 0.0
        return
    if i == 0 or i == nx:
        u[i, j, k] = u[i, j, k] * open_frac
        return
    if fluid[i - 1, j, k] <= 1.0e-6 or fluid[i, j, k] <= 1.0e-6:
        u[i, j, k] = u[i, j, k] * open_frac
        return
    u[i, j, k] = (u[i, j, k] - (p[i, j, k] - p[i - 1, j, k]) / dx) * open_frac


@wp.kernel
def _project_v_kernel_3d(
    v: wp.array3d(dtype=wp.float32),
    p: wp.array3d(dtype=wp.float32),
    fluid: wp.array3d(dtype=wp.float32),
    v_mask: wp.array3d(dtype=wp.float32),
    nx: int,
    ny: int,
    nz: int,
    dy: float,
):
    i, j, k = wp.tid()
    if i >= nx or j > ny or k >= nz:
        return
    open_frac = wp.clamp(v_mask[i, j, k], 0.0, 1.0)
    if open_frac <= 0.0:
        v[i, j, k] = 0.0
        return
    if j == 0 or j == ny:
        v[i, j, k] = v[i, j, k] * open_frac
        return
    if fluid[i, j - 1, k] <= 1.0e-6 or fluid[i, j, k] <= 1.0e-6:
        v[i, j, k] = v[i, j, k] * open_frac
        return
    v[i, j, k] = (v[i, j, k] - (p[i, j, k] - p[i, j - 1, k]) / dy) * open_frac


@wp.kernel
def _project_w_kernel_3d(
    w: wp.array3d(dtype=wp.float32),
    p: wp.array3d(dtype=wp.float32),
    fluid: wp.array3d(dtype=wp.float32),
    w_mask: wp.array3d(dtype=wp.float32),
    nx: int,
    ny: int,
    nz: int,
    dz: float,
):
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k > nz:
        return
    open_frac = wp.clamp(w_mask[i, j, k], 0.0, 1.0)
    if open_frac <= 0.0:
        w[i, j, k] = 0.0
        return
    if k == 0 or k == nz:
        w[i, j, k] = w[i, j, k] * open_frac
        return
    if fluid[i, j, k - 1] <= 1.0e-6 or fluid[i, j, k] <= 1.0e-6:
        w[i, j, k] = w[i, j, k] * open_frac
        return
    w[i, j, k] = (w[i, j, k] - (p[i, j, k] - p[i, j, k - 1]) / dz) * open_frac


@wp.kernel
def _apply_active_identity_kernel_2d(
    src: wp.array2d(dtype=wp.float32),
    active: wp.array2d(dtype=wp.float32),
    dst: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
):
    i, j = wp.tid()
    if i < nx and j < ny and active[i, j] <= 1.0e-6:
        dst[i, j] = src[i, j]


@wp.kernel
def _apply_active_identity_kernel_3d(
    src: wp.array3d(dtype=wp.float32),
    active: wp.array3d(dtype=wp.float32),
    dst: wp.array3d(dtype=wp.float32),
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    if i < nx and j < ny and k < nz and active[i, j, k] <= 1.0e-6:
        dst[i, j, k] = src[i, j, k]


@wp.kernel
def _masked_laplace_diagonal_kernel_2d(
    solid_cell: wp.array2d(dtype=wp.float32),
    u_mask: wp.array2d(dtype=wp.float32),
    v_mask: wp.array2d(dtype=wp.float32),
    active: wp.array2d(dtype=wp.float32),
    diagonal: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    dx: float,
    dy: float,
):
    i, j = wp.tid()
    if i >= nx or j >= ny:
        return
    if active[i, j] <= 1.0e-6:
        diagonal[i, j] = 1.0
        return
    fluid = 1.0 - wp.clamp(solid_cell[i, j], 0.0, 1.0)
    if fluid <= 1.0e-6:
        diagonal[i, j] = 1.0
        return
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)
    w_l = wp.clamp(u_mask[i, j], 0.0, 1.0)
    w_r = wp.clamp(u_mask[i + 1, j], 0.0, 1.0)
    w_d = wp.clamp(v_mask[i, j], 0.0, 1.0)
    w_u = wp.clamp(v_mask[i, j + 1], 0.0, 1.0)
    diagonal[i, j] = -((w_l + w_r) * inv_dx2 + (w_d + w_u) * inv_dy2) / fluid


@wp.kernel
def _masked_laplace_diagonal_kernel_3d(
    solid_cell: wp.array3d(dtype=wp.float32),
    u_mask: wp.array3d(dtype=wp.float32),
    v_mask: wp.array3d(dtype=wp.float32),
    w_mask: wp.array3d(dtype=wp.float32),
    active: wp.array3d(dtype=wp.float32),
    diagonal: wp.array3d(dtype=wp.float32),
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
):
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return
    if active[i, j, k] <= 1.0e-6:
        diagonal[i, j, k] = 1.0
        return
    fluid = 1.0 - wp.clamp(solid_cell[i, j, k], 0.0, 1.0)
    if fluid <= 1.0e-6:
        diagonal[i, j, k] = 1.0
        return
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)
    inv_dz2 = 1.0 / (dz * dz)
    w_l = wp.clamp(u_mask[i, j, k], 0.0, 1.0)
    w_r = wp.clamp(u_mask[i + 1, j, k], 0.0, 1.0)
    w_d = wp.clamp(v_mask[i, j, k], 0.0, 1.0)
    w_u = wp.clamp(v_mask[i, j + 1, k], 0.0, 1.0)
    w_b = wp.clamp(w_mask[i, j, k], 0.0, 1.0)
    w_f = wp.clamp(w_mask[i, j, k + 1], 0.0, 1.0)
    diagonal[i, j, k] = -((w_l + w_r) * inv_dx2 + (w_d + w_u) * inv_dy2 + (w_b + w_f) * inv_dz2) / fluid


def _balance_mean(rhs: CenteredField, fluid: object) -> None:
    grid = rhs.grid
    device = rhs.data.device
    scratch = _BALANCE_SCRATCH.get(str(device))
    if scratch is None:
        scratch = _BalanceScratch(
            sum_out=wp.zeros((1,), dtype=wp.float32, device=device),
            count_out=wp.zeros((1,), dtype=wp.float32, device=device),
            mean_out=wp.zeros((1,), dtype=wp.float32, device=device),
        )
        _BALANCE_SCRATCH[str(device)] = scratch
    wp.launch(_fill_scalar_kernel, dim=1, inputs=(scratch.sum_out, 0.0), device=device)
    wp.launch(_fill_scalar_kernel, dim=1, inputs=(scratch.count_out, 0.0), device=device)
    wp.launch(_fill_scalar_kernel, dim=1, inputs=(scratch.mean_out, 0.0), device=device)
    if grid.is_3d:
        wp.launch(
            _sum_active_kernel_3d,
            dim=grid.shape,
            inputs=(rhs.data, fluid, scratch.sum_out, scratch.count_out, grid.nx, grid.ny, int(grid.nz)),
            device=device,
        )
        wp.launch(_compute_mean_kernel, dim=1, inputs=(scratch.sum_out, scratch.count_out, scratch.mean_out), device=device)
        wp.launch(
            _subtract_mean_kernel_3d,
            dim=grid.shape,
            inputs=(rhs.data, fluid, scratch.mean_out, grid.nx, grid.ny, int(grid.nz)),
            device=device,
        )
        return
    wp.launch(
        _sum_active_kernel_2d,
        dim=grid.shape,
        inputs=(rhs.data, fluid, scratch.sum_out, scratch.count_out, grid.nx, grid.ny),
        device=device,
    )
    wp.launch(_compute_mean_kernel, dim=1, inputs=(scratch.sum_out, scratch.count_out, scratch.mean_out), device=device)
    wp.launch(
        _subtract_mean_kernel_2d,
        dim=grid.shape,
        inputs=(rhs.data, fluid, scratch.mean_out, grid.nx, grid.ny),
        device=device,
    )


def _field_requires_grad(field: CenteredField) -> bool:
    return bool(getattr(field.data, "requires_grad", False))


def _mac_requires_grad(velocity: MACField) -> bool:
    return bool(
        getattr(velocity.u, "requires_grad", False)
        or getattr(velocity.v, "requires_grad", False)
        or (velocity.w is not None and getattr(velocity.w, "requires_grad", False))
    )


def _copy_centered_field(field: CenteredField, *, requires_grad: bool) -> CenteredField:
    result = CenteredField.zeros(field.grid, device=field.data.device, requires_grad=requires_grad)
    wp.copy(result.data, field.data)
    return result


def _copy_mac_field(velocity: MACField, *, requires_grad: bool) -> MACField:
    result = MACField.zeros(velocity.grid, device=velocity.u.device, requires_grad=requires_grad)
    wp.copy(result.u, velocity.u)
    wp.copy(result.v, velocity.v)
    if velocity.w is not None and result.w is not None:
        wp.copy(result.w, velocity.w)
    return result


def _jacobi_scratch_for(grid: object, device: object) -> _JacobiPressureScratch:
    key = (grid, str(device))
    scratch = _JACOBI_PRESSURE_SCRATCH.get(key)
    if scratch is None:
        scratch = _JacobiPressureScratch(
            rhs=CenteredField.zeros(grid, device=device),
            pressure_a=CenteredField.zeros(grid, device=device),
            pressure_b=CenteredField.zeros(grid, device=device),
            diagonal=CenteredField.zeros(grid, device=device),
            ax=CenteredField.zeros(grid, device=device),
        )
        _JACOBI_PRESSURE_SCRATCH[key] = scratch
    return scratch


def _jacobi_scratch(rhs: CenteredField) -> _JacobiPressureScratch:
    return _jacobi_scratch_for(rhs.grid, rhs.data.device)


def build_pressure_rhs(
    velocity: MACField,
    *,
    solid: Optional[SolidMask] = None,
    out: Optional[CenteredField] = None,
) -> CenteredField:
    solid = solid or SolidMask.empty(velocity.grid, device=velocity.u.device)
    rhs = divergence(
        velocity,
        solid=solid,
        out=out
        or CenteredField.zeros(
            velocity.grid,
            device=velocity.u.device,
            requires_grad=_mac_requires_grad(velocity),
        ),
    )
    _balance_mean(rhs, solid.fluid_cell_mask())
    return rhs


def masked_laplace(
    pressure: CenteredField,
    solid: SolidMask,
    active: object,
    *,
    out: Optional[CenteredField] = None,
) -> CenteredField:
    out = out or CenteredField.zeros(
        pressure.grid,
        device=pressure.data.device,
        requires_grad=_field_requires_grad(pressure),
    )
    laplace_centered(pressure, solid=solid, out=out)
    if pressure.grid.is_3d:
        wp.launch(
            _apply_active_identity_kernel_3d,
            dim=pressure.grid.shape,
            inputs=(pressure.data, active, out.data, pressure.grid.nx, pressure.grid.ny, int(pressure.grid.nz)),
            device=pressure.data.device,
        )
        return out
    wp.launch(
        _apply_active_identity_kernel_2d,
        dim=pressure.grid.shape,
        inputs=(pressure.data, active, out.data, pressure.grid.nx, pressure.grid.ny),
        device=pressure.data.device,
    )
    return out


def _masked_laplace_diagonal(
    pressure: CenteredField,
    solid: SolidMask,
    active: object,
    *,
    out: Optional[CenteredField] = None,
) -> CenteredField:
    out = out or CenteredField.zeros(
        pressure.grid,
        device=pressure.data.device,
        requires_grad=_field_requires_grad(pressure),
    )
    if pressure.grid.is_3d:
        assert solid.w is not None and pressure.grid.dz is not None and pressure.grid.nz is not None
        wp.launch(
            _masked_laplace_diagonal_kernel_3d,
            dim=pressure.grid.shape,
            inputs=(
                solid.cell,
                solid.u,
                solid.v,
                solid.w,
                active,
                out.data,
                pressure.grid.nx,
                pressure.grid.ny,
                int(pressure.grid.nz),
                pressure.grid.dx,
                pressure.grid.dy,
                pressure.grid.dz,
            ),
            device=pressure.data.device,
        )
        return out
    wp.launch(
        _masked_laplace_diagonal_kernel_2d,
        dim=pressure.grid.shape,
        inputs=(
            solid.cell,
            solid.u,
            solid.v,
            active,
            out.data,
            pressure.grid.nx,
            pressure.grid.ny,
            pressure.grid.dx,
            pressure.grid.dy,
        ),
        device=pressure.data.device,
    )
    return out


masked_laplace.jacobi_diagonal = _masked_laplace_diagonal
masked_laplace.is_symmetric = True


def _fill_pressure_zero(pressure_field: CenteredField) -> None:
    grid = pressure_field.grid
    if grid.is_3d:
        wp.launch(
            _fill_centered_kernel_3d,
            dim=grid.shape,
            inputs=(pressure_field.data, 0.0, grid.nx, grid.ny, int(grid.nz)),
            device=pressure_field.data.device,
        )
        return
    wp.launch(
        _fill_centered_kernel_2d,
        dim=grid.shape,
        inputs=(pressure_field.data, 0.0, grid.nx, grid.ny),
        device=pressure_field.data.device,
    )


def _solve_pressure_jacobi_fast(
    rhs: CenteredField,
    *,
    solid: SolidMask,
    pressure: Optional[CenteredField],
    solve: LinearSolveConfig,
) -> tuple[CenteredField, LinearSolveStats]:
    active = solid.fluid_cell_mask()
    scratch = _jacobi_scratch(rhs)
    if pressure is None:
        pressure_field = scratch.pressure_a
        _fill_pressure_zero(pressure_field)
        next_pressure = scratch.pressure_b
    elif pressure is scratch.pressure_a:
        pressure_field = scratch.pressure_a
        next_pressure = scratch.pressure_b
    elif pressure is scratch.pressure_b:
        pressure_field = scratch.pressure_b
        next_pressure = scratch.pressure_a
    else:
        pressure_field = pressure
        next_pressure = scratch.pressure_a
    masked_laplace.jacobi_diagonal(pressure_field, solid, active, out=scratch.diagonal)
    for _ in range(int(solve.max_iterations)):
        masked_laplace(pressure_field, solid, active, out=scratch.ax)
        if rhs.grid.is_3d:
            wp.launch(
                _jacobi_update_kernel_3d,
                dim=rhs.grid.shape,
                inputs=(
                    pressure_field.data,
                    scratch.ax.data,
                    rhs.data,
                    scratch.diagonal.data,
                    next_pressure.data,
                    rhs.grid.nx,
                    rhs.grid.ny,
                    int(rhs.grid.nz),
                    float(solve.omega),
                ),
                device=rhs.data.device,
            )
        else:
            wp.launch(
                _jacobi_update_kernel,
                dim=rhs.grid.shape,
                inputs=(
                    pressure_field.data,
                    scratch.ax.data,
                    rhs.data,
                    scratch.diagonal.data,
                    next_pressure.data,
                    rhs.grid.nx,
                    rhs.grid.ny,
                    float(solve.omega),
                ),
                device=rhs.data.device,
            )
        pressure_field, next_pressure = next_pressure, pressure_field
    _balance_mean(pressure_field, active)
    return pressure_field, LinearSolveStats(
        iterations=int(solve.max_iterations),
        converged=False,
        residual=None,
    )


def solve_pressure(
    rhs: CenteredField,
    *,
    solid: Optional[SolidMask] = None,
    pressure: Optional[CenteredField] = None,
    solve_config: Optional[LinearSolveConfig] = None,
) -> tuple[CenteredField, LinearSolveStats]:
    solid = solid or SolidMask.empty(rhs.grid, device=rhs.data.device)
    active = solid.fluid_cell_mask()
    solve = solve_config or LinearSolveConfig()
    if rhs.grid.is_3d:
        if solve.method.strip().lower() != "jacobi":
            raise ValueError("3D pressure solve currently only supports method='jacobi'.")
        if _field_requires_grad(rhs) or (pressure is not None and _field_requires_grad(pressure)):
            pressure_field = (
                CenteredField.zeros(rhs.grid, device=rhs.data.device, requires_grad=True)
                if pressure is None
                else _copy_centered_field(pressure, requires_grad=True)
            )
            diagonal = CenteredField.zeros(rhs.grid, device=rhs.data.device, requires_grad=True)
            ax = CenteredField.zeros(rhs.grid, device=rhs.data.device, requires_grad=True)
            next_pressure = CenteredField.zeros(rhs.grid, device=rhs.data.device, requires_grad=True)
            masked_laplace.jacobi_diagonal(pressure_field, solid, active, out=diagonal)
            for _ in range(int(solve.max_iterations)):
                masked_laplace(pressure_field, solid, active, out=ax)
                wp.launch(
                    _jacobi_update_kernel_3d,
                    dim=rhs.grid.shape,
                    inputs=(
                        pressure_field.data,
                        ax.data,
                        rhs.data,
                        diagonal.data,
                        next_pressure.data,
                        rhs.grid.nx,
                        rhs.grid.ny,
                        int(rhs.grid.nz),
                        float(solve.omega),
                    ),
                    device=rhs.data.device,
                )
                pressure_field, next_pressure = next_pressure, pressure_field
            _balance_mean(pressure_field, active)
            return pressure_field, LinearSolveStats(
                iterations=int(solve.max_iterations),
                converged=False,
                residual=None,
            )
        return _solve_pressure_jacobi_fast(rhs, solid=solid, pressure=pressure, solve=solve)
    requires_grad = _field_requires_grad(rhs) or (pressure is not None and _field_requires_grad(pressure))
    if requires_grad:
        if solve.method.strip().lower() != "jacobi":
            raise ValueError("Differentiable pressure solve currently only supports method='jacobi'.")
        pressure_field = (
            CenteredField.zeros(rhs.grid, device=rhs.data.device, requires_grad=True)
            if pressure is None
            else _copy_centered_field(pressure, requires_grad=True)
        )
        diagonal = CenteredField.zeros(rhs.grid, device=rhs.data.device, requires_grad=True)
        ax = CenteredField.zeros(rhs.grid, device=rhs.data.device, requires_grad=True)
        next_pressure = CenteredField.zeros(rhs.grid, device=rhs.data.device, requires_grad=True)
        masked_laplace.jacobi_diagonal(pressure_field, solid, active, out=diagonal)
        for _ in range(int(solve.max_iterations)):
            masked_laplace(pressure_field, solid, active, out=ax)
            wp.launch(
                _jacobi_update_kernel,
                dim=rhs.grid.shape,
                inputs=(
                    pressure_field.data,
                    ax.data,
                    rhs.data,
                    diagonal.data,
                    next_pressure.data,
                    rhs.grid.nx,
                    rhs.grid.ny,
                    float(solve.omega),
                ),
                device=rhs.data.device,
            )
            pressure_field, next_pressure = next_pressure, pressure_field
        _balance_mean(pressure_field, active)
        return pressure_field, LinearSolveStats(
            iterations=int(solve.max_iterations),
            converged=False,
            residual=None,
        )
    if solve.method.strip().lower() == "jacobi":
        return _solve_pressure_jacobi_fast(rhs, solid=solid, pressure=pressure, solve=solve)
    return solve_linear(masked_laplace, rhs, solve_config, solid, active, x0=pressure)


def project(
    velocity: MACField,
    *,
    solid: Optional[SolidMask] = None,
    pressure: Optional[CenteredField] = None,
    solve_config: Optional[LinearSolveConfig] = None,
    boundary: Optional[VelocityBoundary] = None,
) -> tuple[MACField, CenteredField, LinearSolveStats]:
    solid = solid or SolidMask.empty(velocity.grid, device=velocity.u.device)
    rhs_out: Optional[CenteredField] = None
    if not (_mac_requires_grad(velocity) or (pressure is not None and _field_requires_grad(pressure))):
        rhs_out = _jacobi_scratch_for(velocity.grid, velocity.u.device).rhs
    rhs = build_pressure_rhs(velocity, solid=solid, out=rhs_out)
    pressure, stats = solve_pressure(rhs, solid=solid, pressure=pressure, solve_config=solve_config)
    fluid = solid.fluid_cell_mask()
    projected = (
        _copy_mac_field(velocity, requires_grad=True)
        if _mac_requires_grad(velocity) or _field_requires_grad(pressure)
        else velocity
    )
    if projected.grid.is_3d:
        assert projected.w is not None and solid.w is not None and projected.grid.dz is not None and projected.grid.nz is not None
        wp.launch(
            _project_u_kernel_3d,
            dim=projected.grid.shape_u,
            inputs=(projected.u, pressure.data, fluid, solid.u, projected.grid.nx, projected.grid.ny, int(projected.grid.nz), projected.grid.dx),
            device=projected.u.device,
        )
        wp.launch(
            _project_v_kernel_3d,
            dim=projected.grid.shape_v,
            inputs=(projected.v, pressure.data, fluid, solid.v, projected.grid.nx, projected.grid.ny, int(projected.grid.nz), projected.grid.dy),
            device=projected.v.device,
        )
        wp.launch(
            _project_w_kernel_3d,
            dim=projected.grid.shape_w,
            inputs=(projected.w, pressure.data, fluid, solid.w, projected.grid.nx, projected.grid.ny, int(projected.grid.nz), projected.grid.dz),
            device=projected.w.device,
        )
    else:
        wp.launch(
            _project_u_kernel_2d,
            dim=projected.grid.shape_u,
            inputs=(projected.u, pressure.data, fluid, solid.u, projected.grid.nx, projected.grid.ny, projected.grid.dx),
            device=projected.u.device,
        )
        wp.launch(
            _project_v_kernel_2d,
            dim=projected.grid.shape_v,
            inputs=(projected.v, pressure.data, fluid, solid.v, projected.grid.nx, projected.grid.ny, projected.grid.dy),
            device=projected.v.device,
        )
    apply_velocity_boundary(projected, boundary=boundary, solid=solid)
    return projected, pressure, stats


def make_incompressible(
    velocity: MACField,
    *,
    solid: Optional[SolidMask] = None,
    pressure: Optional[CenteredField] = None,
    solve_config: Optional[LinearSolveConfig] = None,
    boundary: Optional[VelocityBoundary] = None,
) -> tuple[MACField, CenteredField, LinearSolveStats]:
    """Project a MAC velocity field to a divergence-free state."""

    return project(
        velocity,
        solid=solid,
        pressure=pressure,
        solve_config=solve_config,
        boundary=boundary,
    )
