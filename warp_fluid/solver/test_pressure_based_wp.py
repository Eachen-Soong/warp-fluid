from __future__ import annotations

import numpy as np
import pytest


def test_pressure_based_warp_solver_smoke() -> None:
    pytest.importorskip("warp")
    from ..core.grid import GridSpec
    from ..geom.levelset_grid.airfoil import naca4_airfoil_levelset
    from .pressure_based_np import FreestreamCondition, PressureBasedSolverConfig
    from .pressure_based_wp import run_pressure_based_solver_warp

    grid = GridSpec.from_extent(48, 24, 6.0, 3.0)
    levelset = naca4_airfoil_levelset(
        grid,
        "0012",
        chord=0.8,
        leading_edge=(1.5, 1.5),
        samples=96,
    )
    state, history = run_pressure_based_solver_warp(
        grid,
        levelset,
        chord=0.8,
        freestream=FreestreamCondition(
            mach=0.15,
            static_pressure=101325.0,
            static_temperature=288.15,
            angle_of_attack_deg=2.0,
            reynolds_number=2.0e5,
            turbulence_intensity=0.01,
            turbulent_viscosity_ratio=5.0,
        ),
        config=PressureBasedSolverConfig(
            pseudo_steps=2,
            cfl=0.4,
            simple_iterations=1,
            pressure_correction_iterations=6,
            velocity_relaxation=0.4,
            pressure_relaxation=0.2,
            temperature_relaxation=0.3,
            turbulence_relaxation=0.2,
            pressure_linear_solver="SOR",
            pressure_sor_omega=1.4,
            outlet_static_pressure=101325.0,
            convergence_window=2,
        ),
        device="cpu",
    )

    host = state.to_numpy_state()
    assert history
    assert host.pressure.shape == grid.shape
    assert np.all(np.isfinite(host.pressure[host.fluid]))
    assert np.all(np.isfinite(host.density[host.fluid]))
    assert np.all(np.isfinite(host.temperature[host.fluid]))
    assert np.isfinite(history[-1].pressure_correction)
    assert np.isfinite(history[-1].drag_coefficient)
    assert np.isfinite(history[-1].lift_coefficient)
    assert np.isfinite(history[-1].moment_coefficient)
