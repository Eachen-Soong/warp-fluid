from __future__ import annotations

import numpy as np
import warp as wp
from typing import Optional

from ..core import CenteredField, GridSpec, MACField, SolidMask
from ..physics.incompressible import build_pressure_rhs, solve_pressure
from .optimize import LinearSolveConfig, solve_linear


@wp.kernel
def _diagonal_operator_kernel(
    src: wp.array2d(dtype=wp.float32),
    dst: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
):
    i, j = wp.tid()
    if i < nx and j < ny:
        diagonal = 2.0 + float(i) + 0.5 * float(j)
        dst[i, j] = diagonal * src[i, j]


@wp.kernel
def _diagonal_only_kernel(
    dst: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
):
    i, j = wp.tid()
    if i < nx and j < ny:
        dst[i, j] = 2.0 + float(i) + 0.5 * float(j)


def diagonal_operator(
    x: CenteredField,
    *,
    out: Optional[CenteredField] = None,
) -> CenteredField:
    out = out or CenteredField.zeros(x.grid, device=x.data.device)
    wp.launch(
        _diagonal_operator_kernel,
        dim=x.grid.shape,
        inputs=(x.data, out.data, x.grid.nx, x.grid.ny),
        device=x.data.device,
    )
    return out


def _diagonal_operator_diagonal(
    x: CenteredField,
    *,
    out: Optional[CenteredField] = None,
) -> CenteredField:
    out = out or CenteredField.zeros(x.grid, device=x.data.device)
    wp.launch(
        _diagonal_only_kernel,
        dim=x.grid.shape,
        inputs=(out.data, x.grid.nx, x.grid.ny),
        device=x.data.device,
    )
    return out


diagonal_operator.jacobi_diagonal = _diagonal_operator_diagonal
diagonal_operator.is_symmetric = True


def test_solve_linear_supports_multiple_iterative_methods() -> None:
    wp.init()
    grid = GridSpec(nx=4, ny=3, dx=1.0, dy=1.0)
    exact = CenteredField.from_numpy(
        grid,
        np.array(
            [
                [0.5, -1.0, 0.75],
                [1.25, 0.25, -0.5],
                [0.0, 2.0, -1.5],
                [1.5, -0.25, 0.5],
            ],
            dtype=np.float32,
        ),
        device="cpu",
    )
    rhs = diagonal_operator(exact)
    methods = ["jacobi", "CG", "CG-adaptive", "biCG-stab", "auto"]

    for method in methods:
        solution, stats = solve_linear(
            diagonal_operator,
            rhs,
            LinearSolveConfig(method=method, max_iterations=50, tolerance=1.0e-6),
        )
        np.testing.assert_allclose(solution.numpy(), exact.numpy(), atol=2.0e-5, rtol=0.0)
        assert stats.converged, method
        assert stats.residual is not None and stats.residual <= 1.0e-5, method


def test_solve_pressure_accepts_cg_family_methods() -> None:
    wp.init()
    grid = GridSpec(nx=4, ny=4, dx=1.0, dy=1.0)
    velocity = MACField.zeros(grid, device="cpu")
    solid = SolidMask.empty(grid, device="cpu")
    rhs = build_pressure_rhs(velocity, solid=solid)

    for method in ["CG", "CG-adaptive", "biCG-stab", "auto"]:
        pressure, stats = solve_pressure(
            rhs,
            solid=solid,
            solve_config=LinearSolveConfig(method=method, max_iterations=20, tolerance=1.0e-6),
        )
        np.testing.assert_allclose(pressure.numpy(), 0.0)
        assert stats.converged, method
        assert stats.residual == 0.0, method
