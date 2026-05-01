from __future__ import annotations

import numpy as np
import pytest


def test_density_based_warp_solver_smoke() -> None:
    pytest.importorskip("warp")
    from ..core.grid import GridSpec
    from ..geom.levelset_grid.airfoil import naca4_airfoil_levelset
    from .density_based_wp import DensityBasedSolverConfig, run_density_based_solver_warp
    from .pressure_based_np import FreestreamCondition

    grid = GridSpec.from_extent(48, 24, 6.0, 3.0)
    levelset = naca4_airfoil_levelset(
        grid,
        "0012",
        chord=0.8,
        leading_edge=(1.5, 1.5),
        samples=96,
    )
    state, history = run_density_based_solver_warp(
        grid,
        levelset,
        chord=0.8,
        freestream=FreestreamCondition(
            mach=1.2,
            static_pressure=101325.0,
            static_temperature=288.15,
            angle_of_attack_deg=2.0,
            reynolds_number=2.0e5,
            turbulence_intensity=0.0,
            turbulent_viscosity_ratio=1.0,
        ),
        config=DensityBasedSolverConfig(
            pseudo_steps=3,
            cfl=0.2,
            convergence_window=2,
        ),
        device="cpu",
    )

    primitive = state.primitive_fields(1.4, 287.05)
    assert history
    assert primitive["pressure"].shape == grid.shape
    assert np.all(np.isfinite(primitive["pressure"][primitive["fluid"]]))
    assert np.all(np.isfinite(primitive["mach"][primitive["fluid"]]))
    assert np.isfinite(history[-1].density_residual)
    assert np.isfinite(history[-1].drag_coefficient)
