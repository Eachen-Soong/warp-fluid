from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import warp as wp

from ..core.field import CenteredField


@wp.kernel
def _fill_f32_buffer_kernel(value: float, out: wp.array(dtype=wp.float32), count: int):
    index = wp.tid()
    if index < count:
        out[index] = value


@wp.kernel
def _jacobi_update_kernel(
    x: wp.array2d(dtype=wp.float32),
    ax: wp.array2d(dtype=wp.float32),
    rhs: wp.array2d(dtype=wp.float32),
    diagonal: wp.array2d(dtype=wp.float32),
    x_next: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    omega: float,
):
    i, j = wp.tid()
    if i >= nx or j >= ny:
        return
    diag = diagonal[i, j]
    if wp.abs(diag) <= 1.0e-6:
        x_next[i, j] = x[i, j]
        return
    correction = (rhs[i, j] - ax[i, j]) / diag
    x_next[i, j] = x[i, j] + omega * correction


@wp.kernel
def _axpy_kernel(
    dst: wp.array2d(dtype=wp.float32),
    src: wp.array2d(dtype=wp.float32),
    scale: float,
    nx: int,
    ny: int,
):
    i, j = wp.tid()
    if i < nx and j < ny:
        dst[i, j] = dst[i, j] + scale * src[i, j]


@wp.kernel
def _combine_kernel(
    a: wp.array2d(dtype=wp.float32),
    b: wp.array2d(dtype=wp.float32),
    out: wp.array2d(dtype=wp.float32),
    scale_a: float,
    scale_b: float,
    nx: int,
    ny: int,
):
    i, j = wp.tid()
    if i < nx and j < ny:
        out[i, j] = scale_a * a[i, j] + scale_b * b[i, j]


@wp.kernel
def _update_direction_kernel(
    residual: wp.array2d(dtype=wp.float32),
    direction: wp.array2d(dtype=wp.float32),
    beta: float,
    nx: int,
    ny: int,
):
    i, j = wp.tid()
    if i < nx and j < ny:
        direction[i, j] = residual[i, j] + beta * direction[i, j]


@wp.kernel
def _bicgstab_direction_kernel(
    residual: wp.array2d(dtype=wp.float32),
    p: wp.array2d(dtype=wp.float32),
    v: wp.array2d(dtype=wp.float32),
    beta: float,
    omega: float,
    nx: int,
    ny: int,
):
    i, j = wp.tid()
    if i < nx and j < ny:
        p[i, j] = residual[i, j] + beta * (p[i, j] - omega * v[i, j])


@wp.kernel
def _dot_kernel(
    a: wp.array2d(dtype=wp.float32),
    b: wp.array2d(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
    nx: int,
    ny: int,
):
    i, j = wp.tid()
    if i < nx and j < ny:
        wp.atomic_add(out, 0, a[i, j] * b[i, j])


@wp.kernel
def _max_abs_kernel(
    values: wp.array2d(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
    nx: int,
    ny: int,
):
    i, j = wp.tid()
    if i >= nx or j >= ny:
        return
    value = wp.abs(values[i, j])
    if wp.isnan(value) or wp.isinf(value):
        return
    wp.atomic_max(out, 0, value)


@wp.kernel
def _bicgstab_two_dot_kernel(
    a0: wp.array2d(dtype=wp.float32),
    b0: wp.array2d(dtype=wp.float32),
    a1: wp.array2d(dtype=wp.float32),
    b1: wp.array2d(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
    nx: int,
    ny: int,
):
    i, j = wp.tid()
    if i < nx and j < ny:
        wp.atomic_add(out, 0, a0[i, j] * b0[i, j])
        wp.atomic_add(out, 1, a1[i, j] * b1[i, j])


@wp.kernel
def _bicgstab_update_xs_reduce_kernel(
    residual: wp.array2d(dtype=wp.float32),
    v: wp.array2d(dtype=wp.float32),
    p: wp.array2d(dtype=wp.float32),
    x: wp.array2d(dtype=wp.float32),
    s: wp.array2d(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
    alpha: float,
    nx: int,
    ny: int,
):
    i, j = wp.tid()
    if i >= nx or j >= ny:
        return
    s_value = residual[i, j] - alpha * v[i, j]
    s[i, j] = s_value
    x[i, j] = x[i, j] + alpha * p[i, j]
    value = wp.abs(s_value)
    if not wp.isnan(value) and not wp.isinf(value):
        wp.atomic_max(out, 0, value)


@wp.kernel
def _bicgstab_update_xr_reduce_kernel(
    x: wp.array2d(dtype=wp.float32),
    s: wp.array2d(dtype=wp.float32),
    t: wp.array2d(dtype=wp.float32),
    residual: wp.array2d(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
    omega: float,
    nx: int,
    ny: int,
):
    i, j = wp.tid()
    if i >= nx or j >= ny:
        return
    residual_value = s[i, j] - omega * t[i, j]
    residual[i, j] = residual_value
    x[i, j] = x[i, j] + omega * s[i, j]
    value = wp.abs(residual_value)
    if not wp.isnan(value) and not wp.isinf(value):
        wp.atomic_max(out, 0, value)


@dataclass(frozen=True)
class LinearSolveConfig:
    """Configuration for linear iterative solvers."""

    method: str = "jacobi"
    max_iterations: int = 200
    tolerance: float = 1.0e-5
    omega: float = 1.0

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1.")
        if self.tolerance < 0.0:
            raise ValueError("tolerance must be non-negative.")
        canonical = _canonical_method(self.method, None)
        supported = {
            "jacobi",
            "CG",
            "CG-adaptive",
            "biCG-stab",
            "auto",
            "rbgs",
            "multigrid",
        }
        if canonical not in supported:
            raise ValueError(f"Unsupported linear solve method '{self.method}'.")


@dataclass(frozen=True)
class LinearSolveStats:
    iterations: int
    converged: bool
    residual: Optional[float] = None


def _canonical_method(method: str, operator: Optional[Callable[..., CenteredField]]) -> str:
    normalized = method.strip()
    lowered = normalized.lower()
    if lowered == "auto":
        if operator is not None and getattr(operator, "is_symmetric", False):
            return "CG-adaptive"
        return "biCG-stab"
    if lowered == "jacobi":
        return "jacobi"
    if lowered == "cg":
        return "CG"
    if lowered == "cg-adaptive":
        return "CG-adaptive"
    if lowered == "bicg":
        return "biCG"
    if lowered == "bicg-stab":
        return "biCG-stab"
    if lowered == "rbgs":
        return "rbgs"
    if lowered == "multigrid":
        return "multigrid"
    return normalized


def _new_field_like(reference: CenteredField) -> CenteredField:
    return CenteredField.zeros(
        reference.grid,
        device=reference.data.device,
        requires_grad=bool(getattr(reference.data, "requires_grad", False)),
    )


def _copy_field(field: CenteredField) -> CenteredField:
    result = _new_field_like(field)
    wp.copy(result.data, field.data)
    return result


def _apply_operator(
    f: Callable[..., CenteredField],
    x: CenteredField,
    out: CenteredField,
    f_args: tuple,
    f_kwargs: dict,
) -> CenteredField:
    return f(x, *f_args, out=out, **f_kwargs)


def _dot(a: CenteredField, b: CenteredField, scratch: object) -> float:
    wp.launch(_fill_f32_buffer_kernel, dim=1, inputs=(0.0, scratch, 1), device=a.data.device)
    wp.launch(
        _dot_kernel,
        dim=a.grid.shape,
        inputs=(a.data, b.data, scratch, a.grid.nx, a.grid.ny),
        device=a.data.device,
    )
    return float(scratch.numpy()[0])


def _max_abs(field: CenteredField, scratch: object) -> float:
    wp.launch(_fill_f32_buffer_kernel, dim=1, inputs=(0.0, scratch, 1), device=field.data.device)
    wp.launch(
        _max_abs_kernel,
        dim=field.grid.shape,
        inputs=(field.data, scratch, field.grid.nx, field.grid.ny),
        device=field.data.device,
    )
    return float(scratch.numpy()[0])


def _two_dot(
    a0: CenteredField,
    b0: CenteredField,
    a1: CenteredField,
    b1: CenteredField,
    scratch: object,
) -> tuple[float, float]:
    wp.launch(_fill_f32_buffer_kernel, dim=2, inputs=(0.0, scratch, 2), device=a0.data.device)
    wp.launch(
        _bicgstab_two_dot_kernel,
        dim=a0.grid.shape,
        inputs=(a0.data, b0.data, a1.data, b1.data, scratch, a0.grid.nx, a0.grid.ny),
        device=a0.data.device,
    )
    values = scratch.numpy()
    return float(values[0]), float(values[1])


def _bicgstab_update_xs_reduce(
    residual: CenteredField,
    v: CenteredField,
    p: CenteredField,
    x: CenteredField,
    s: CenteredField,
    scratch: object,
    alpha: float,
) -> float:
    wp.launch(_fill_f32_buffer_kernel, dim=1, inputs=(0.0, scratch, 1), device=x.data.device)
    wp.launch(
        _bicgstab_update_xs_reduce_kernel,
        dim=x.grid.shape,
        inputs=(residual.data, v.data, p.data, x.data, s.data, scratch, alpha, x.grid.nx, x.grid.ny),
        device=x.data.device,
    )
    return float(scratch.numpy()[0])


def _bicgstab_update_xr_reduce(
    x: CenteredField,
    s: CenteredField,
    t: CenteredField,
    residual: CenteredField,
    scratch: object,
    omega: float,
) -> float:
    wp.launch(_fill_f32_buffer_kernel, dim=1, inputs=(0.0, scratch, 1), device=x.data.device)
    wp.launch(
        _bicgstab_update_xr_reduce_kernel,
        dim=x.grid.shape,
        inputs=(x.data, s.data, t.data, residual.data, scratch, omega, x.grid.nx, x.grid.ny),
        device=x.data.device,
    )
    return float(scratch.numpy()[0])


def _compute_residual(
    f: Callable[..., CenteredField],
    x: CenteredField,
    y: CenteredField,
    ax: CenteredField,
    residual: CenteredField,
    f_args: tuple,
    f_kwargs: dict,
) -> CenteredField:
    _apply_operator(f, x, ax, f_args, f_kwargs)
    wp.launch(
        _combine_kernel,
        dim=x.grid.shape,
        inputs=(y.data, ax.data, residual.data, 1.0, -1.0, x.grid.nx, x.grid.ny),
        device=x.data.device,
    )
    return residual


def _solve_jacobi(
    f: Callable[..., CenteredField],
    y: CenteredField,
    solve: LinearSolveConfig,
    f_args: tuple,
    f_kwargs: dict,
    x0: Optional[CenteredField],
) -> tuple[CenteredField, LinearSolveStats]:
    diagonal_fn = getattr(f, "jacobi_diagonal", None)
    if diagonal_fn is None:
        raise ValueError(
            f"Linear operator '{getattr(f, '__name__', type(f).__name__)}' does not define a 'jacobi_diagonal' helper."
        )
    grid = y.grid
    device = y.data.device
    x = _copy_field(x0) if x0 is not None else _new_field_like(y)
    x_next = _new_field_like(y)
    ax = _new_field_like(y)
    residual = _new_field_like(y)
    diagonal = _new_field_like(y)
    max_buffer = wp.zeros((1,), dtype=wp.float32, device=device)
    diagonal_fn(x, *f_args, out=diagonal, **f_kwargs)
    converged = False
    iterations = 0
    residual_value = None
    for iteration in range(solve.max_iterations):
        _apply_operator(f, x, ax, f_args, f_kwargs)
        wp.launch(
            _jacobi_update_kernel,
            dim=grid.shape,
            inputs=(x.data, ax.data, y.data, diagonal.data, x_next.data, grid.nx, grid.ny, solve.omega),
            device=device,
        )
        x, x_next = x_next, x
        _compute_residual(f, x, y, ax, residual, f_args, f_kwargs)
        iterations = iteration + 1
        residual_value = _max_abs(residual, max_buffer)
        if residual_value <= solve.tolerance:
            converged = True
            break
    if residual_value is None:
        _compute_residual(f, x, y, ax, residual, f_args, f_kwargs)
        residual_value = _max_abs(residual, max_buffer)
    return x, LinearSolveStats(iterations=iterations, converged=converged, residual=residual_value)


def _solve_cg(
    f: Callable[..., CenteredField],
    y: CenteredField,
    solve: LinearSolveConfig,
    f_args: tuple,
    f_kwargs: dict,
    x0: Optional[CenteredField],
) -> tuple[CenteredField, LinearSolveStats]:
    x = _copy_field(x0) if x0 is not None else _new_field_like(y)
    ax = _new_field_like(y)
    residual = _new_field_like(y)
    direction = _new_field_like(y)
    ad = _new_field_like(y)
    dot_buffer = wp.zeros((1,), dtype=wp.float32, device=y.data.device)
    max_buffer = wp.zeros((1,), dtype=wp.float32, device=y.data.device)
    _compute_residual(f, x, y, ax, residual, f_args, f_kwargs)
    wp.copy(direction.data, residual.data)
    rr = _dot(residual, residual, dot_buffer)
    residual_value = _max_abs(residual, max_buffer)
    if residual_value <= solve.tolerance:
        return x, LinearSolveStats(iterations=0, converged=True, residual=residual_value)
    converged = False
    iterations = 0
    for iteration in range(solve.max_iterations):
        _apply_operator(f, direction, ad, f_args, f_kwargs)
        denom = _dot(direction, ad, dot_buffer)
        if abs(denom) <= 1.0e-12:
            break
        alpha = rr / denom
        wp.launch(_axpy_kernel, dim=y.grid.shape, inputs=(x.data, direction.data, alpha, y.grid.nx, y.grid.ny), device=y.data.device)
        wp.launch(_axpy_kernel, dim=y.grid.shape, inputs=(residual.data, ad.data, -alpha, y.grid.nx, y.grid.ny), device=y.data.device)
        residual_value = _max_abs(residual, max_buffer)
        iterations = iteration + 1
        if residual_value <= solve.tolerance:
            converged = True
            break
        rr_new = _dot(residual, residual, dot_buffer)
        if rr <= 1.0e-20:
            rr = rr_new
            break
        beta = rr_new / rr
        wp.launch(
            _update_direction_kernel,
            dim=y.grid.shape,
            inputs=(residual.data, direction.data, beta, y.grid.nx, y.grid.ny),
            device=y.data.device,
        )
        rr = rr_new
    _compute_residual(f, x, y, ax, residual, f_args, f_kwargs)
    residual_value = _max_abs(residual, max_buffer)
    return x, LinearSolveStats(iterations=iterations, converged=converged, residual=residual_value)


def _solve_cg_adaptive(
    f: Callable[..., CenteredField],
    y: CenteredField,
    solve: LinearSolveConfig,
    f_args: tuple,
    f_kwargs: dict,
    x0: Optional[CenteredField],
) -> tuple[CenteredField, LinearSolveStats]:
    x = _copy_field(x0) if x0 is not None else _new_field_like(y)
    ax = _new_field_like(y)
    residual = _new_field_like(y)
    direction = _new_field_like(y)
    ad = _new_field_like(y)
    dot_buffer = wp.zeros((1,), dtype=wp.float32, device=y.data.device)
    max_buffer = wp.zeros((1,), dtype=wp.float32, device=y.data.device)
    _compute_residual(f, x, y, ax, residual, f_args, f_kwargs)
    wp.copy(direction.data, residual.data)
    _apply_operator(f, direction, ad, f_args, f_kwargs)
    residual_value = _max_abs(residual, max_buffer)
    if residual_value <= solve.tolerance:
        return x, LinearSolveStats(iterations=0, converged=True, residual=residual_value)
    converged = False
    iterations = 0
    for iteration in range(solve.max_iterations):
        dx_dy = _dot(direction, ad, dot_buffer)
        if abs(dx_dy) <= 1.0e-12:
            break
        alpha = _dot(direction, residual, dot_buffer) / dx_dy
        wp.launch(_axpy_kernel, dim=y.grid.shape, inputs=(x.data, direction.data, alpha, y.grid.nx, y.grid.ny), device=y.data.device)
        wp.launch(_axpy_kernel, dim=y.grid.shape, inputs=(residual.data, ad.data, -alpha, y.grid.nx, y.grid.ny), device=y.data.device)
        residual_value = _max_abs(residual, max_buffer)
        iterations = iteration + 1
        if residual_value <= solve.tolerance:
            converged = True
            break
        beta = -_dot(residual, ad, dot_buffer) / dx_dy
        wp.launch(
            _update_direction_kernel,
            dim=y.grid.shape,
            inputs=(residual.data, direction.data, beta, y.grid.nx, y.grid.ny),
            device=y.data.device,
        )
        _apply_operator(f, direction, ad, f_args, f_kwargs)
    _compute_residual(f, x, y, ax, residual, f_args, f_kwargs)
    residual_value = _max_abs(residual, max_buffer)
    return x, LinearSolveStats(iterations=iterations, converged=converged, residual=residual_value)


def _solve_bicgstab(
    f: Callable[..., CenteredField],
    y: CenteredField,
    solve: LinearSolveConfig,
    f_args: tuple,
    f_kwargs: dict,
    x0: Optional[CenteredField],
) -> tuple[CenteredField, LinearSolveStats]:
    x = _copy_field(x0) if x0 is not None else _new_field_like(y)
    ax = _new_field_like(y)
    residual = _new_field_like(y)
    residual_hat = _new_field_like(y)
    p = _new_field_like(y)
    v = _new_field_like(y)
    s = _new_field_like(y)
    t = _new_field_like(y)
    dot_buffer = wp.zeros((1,), dtype=wp.float32, device=y.data.device)
    two_dot_buffer = wp.zeros((2,), dtype=wp.float32, device=y.data.device)
    max_buffer = wp.zeros((1,), dtype=wp.float32, device=y.data.device)
    _compute_residual(f, x, y, ax, residual, f_args, f_kwargs)
    wp.copy(residual_hat.data, residual.data)
    residual_value = _max_abs(residual, max_buffer)
    if residual_value <= solve.tolerance:
        return x, LinearSolveStats(iterations=0, converged=True, residual=residual_value)
    rho_prev = 1.0
    alpha = 1.0
    omega = 1.0
    converged = False
    iterations = 0
    for iteration in range(solve.max_iterations):
        rho = _dot(residual_hat, residual, dot_buffer)
        if abs(rho) <= 1.0e-12:
            break
        if iteration == 0:
            wp.copy(p.data, residual.data)
        else:
            if abs(omega) <= 1.0e-12:
                break
            beta = (rho / rho_prev) * (alpha / omega)
            wp.launch(
                _bicgstab_direction_kernel,
                dim=y.grid.shape,
                inputs=(residual.data, p.data, v.data, beta, omega, y.grid.nx, y.grid.ny),
                device=y.data.device,
            )
        _apply_operator(f, p, v, f_args, f_kwargs)
        denom = _dot(residual_hat, v, dot_buffer)
        if abs(denom) <= 1.0e-12:
            break
        alpha = rho / denom
        residual_value = _bicgstab_update_xs_reduce(residual, v, p, x, s, max_buffer, alpha)
        iterations = iteration + 1
        if residual_value <= solve.tolerance:
            wp.copy(residual.data, s.data)
            converged = True
            break
        _apply_operator(f, s, t, f_args, f_kwargs)
        tt, ts = _two_dot(t, t, t, s, two_dot_buffer)
        if abs(tt) <= 1.0e-12:
            break
        omega = ts / tt
        residual_value = _bicgstab_update_xr_reduce(x, s, t, residual, max_buffer, omega)
        if residual_value <= solve.tolerance:
            converged = True
            break
        rho_prev = rho
    _compute_residual(f, x, y, ax, residual, f_args, f_kwargs)
    residual_value = _max_abs(residual, max_buffer)
    return x, LinearSolveStats(iterations=iterations, converged=converged, residual=residual_value)


def solve_linear(
    f: Callable[..., CenteredField],
    y: CenteredField,
    solve: Optional[LinearSolveConfig] = None,
    *f_args,
    x0: Optional[CenteredField] = None,
    f_kwargs: Optional[dict] = None,
    **f_kwargs_,
) -> tuple[CenteredField, LinearSolveStats]:
    """Solve the linear system `f(x, *f_args, **f_kwargs) = y`."""

    solve = solve or LinearSolveConfig()
    method = _canonical_method(solve.method, f)
    f_kwargs = dict(f_kwargs or {})
    f_kwargs.update(f_kwargs_)
    if method == "jacobi":
        return _solve_jacobi(f, y, solve, f_args, f_kwargs, x0)
    if method == "CG":
        return _solve_cg(f, y, solve, f_args, f_kwargs, x0)
    if method == "CG-adaptive":
        return _solve_cg_adaptive(f, y, solve, f_args, f_kwargs, x0)
    if method == "biCG-stab":
        return _solve_bicgstab(f, y, solve, f_args, f_kwargs, x0)
    if method == "biCG":
        raise NotImplementedError("Generic biCG is not implemented yet. Use 'biCG-stab' instead.")
    raise NotImplementedError(
        f"Linear solve method '{solve.method}' is not implemented yet. Available methods: jacobi, CG, CG-adaptive, biCG-stab."
    )
