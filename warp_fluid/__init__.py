"""Warp-only fluid simulation toolkit."""

from __future__ import annotations

import os

os.environ.setdefault("WARP_CACHE_DIR", "/tmp/warp_cache")

import warp as wp

wp.config.kernel_cache_dir = os.environ["WARP_CACHE_DIR"]

from .core import CenteredField, GridSpec, MACField, SolidMask, VelocityBoundary, apply_velocity_boundary
from .physics.incompressible import make_incompressible
from .solver import (
    AerodynamicCoefficients,
    DensityBasedSnapshot,
    DensityBasedSolverConfig,
    DensityBasedState,
    FreestreamCondition,
    LinearSolveConfig,
    LinearSolveStats,
    PressureBasedSolverConfig,
    PressureBasedState,
    PressureBasedWarpState,
    ResidualSnapshot,
    initialize_density_based_state_warp,
    initialize_pressure_based_state,
    initialize_pressure_based_state_warp,
    run_density_based_solver_warp,
    run_pressure_based_solver,
    run_pressure_based_solver_warp,
    solve_linear,
)

__version__ = "0.1.0"

__all__ = [
    "CenteredField",
    "AerodynamicCoefficients",
    "DensityBasedSnapshot",
    "DensityBasedSolverConfig",
    "DensityBasedState",
    "GridSpec",
    "LinearSolveConfig",
    "LinearSolveStats",
    "MACField",
    "FreestreamCondition",
    "PressureBasedSolverConfig",
    "PressureBasedState",
    "PressureBasedWarpState",
    "ResidualSnapshot",
    "SolidMask",
    "VelocityBoundary",
    "__version__",
    "apply_velocity_boundary",
    "initialize_density_based_state_warp",
    "initialize_pressure_based_state",
    "initialize_pressure_based_state_warp",
    "make_incompressible",
    "run_density_based_solver_warp",
    "run_pressure_based_solver",
    "run_pressure_based_solver_warp",
    "solve_linear",
]
