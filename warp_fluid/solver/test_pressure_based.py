from __future__ import annotations

import numpy as np

from ..core.grid import GridSpec
from ..geom.levelset_grid.airfoil import naca4_airfoil_levelset
from .pressure_based import FreestreamCondition, PressureBasedSolverConfig, run_pressure_based_solver


def test_pressure_based_config_rejects_unknown_scheme() -> None:
    try:
        PressureBasedSolverConfig(convection_scheme="first_order_upwind")
    except ValueError as exc:
        assert "Unsupported" in str(exc)
    else:
        raise AssertionError("Expected unsupported scheme to raise ValueError.")


def test_pressure_based_config_rejects_invalid_sor_omega() -> None:
    try:
        PressureBasedSolverConfig(pressure_sor_omega=2.1)
    except ValueError as exc:
        assert "pressure_sor_omega" in str(exc)
    else:
        raise AssertionError("Expected invalid SOR omega to raise ValueError.")


def test_pressure_based_config_rejects_unknown_pressure_solver() -> None:
    try:
        PressureBasedSolverConfig(pressure_linear_solver="GMRES")
    except ValueError as exc:
        assert "pressure_linear_solver" in str(exc)
    else:
        raise AssertionError("Expected unsupported pressure solver to raise ValueError.")


def test_pressure_based_solver_smoke() -> None:
    grid = GridSpec.from_extent(48, 24, 6.0, 3.0)
    levelset = naca4_airfoil_levelset(
        grid,
        "0012",
        chord=0.8,
        leading_edge=(1.5, 1.5),
        samples=96,
    )
    state, history = run_pressure_based_solver(
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
            pseudo_steps=3,
            cfl=0.4,
            simple_iterations=1,
            pressure_correction_iterations=8,
            velocity_relaxation=0.4,
            pressure_relaxation=0.2,
            temperature_relaxation=0.3,
            turbulence_relaxation=0.2,
            pressure_linear_solver="SOR",
            pressure_sor_omega=1.4,
            outlet_static_pressure=101325.0,
            convergence_window=2,
        ),
    )

    assert history
    assert state.pressure.shape == grid.shape
    assert np.all(np.isfinite(state.pressure[state.fluid]))
    assert np.all(np.isfinite(state.density[state.fluid]))
    assert np.all(np.isfinite(state.temperature[state.fluid]))
    assert np.isfinite(history[-1].pressure_correction)
    assert np.isfinite(history[-1].drag_coefficient)
    assert np.isfinite(history[-1].lift_coefficient)
    assert np.isfinite(history[-1].moment_coefficient)
    assert np.isfinite(history[-1].normalized_mass)
