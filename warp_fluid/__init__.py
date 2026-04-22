"""Warp-only fluid simulation toolkit."""

from __future__ import annotations

import os

os.environ.setdefault("WARP_CACHE_DIR", "/tmp/warp_cache")

import warp as wp

wp.config.kernel_cache_dir = os.environ["WARP_CACHE_DIR"]

from .core import CenteredField, GridSpec, MACField, SolidMask, VelocityBoundary, apply_velocity_boundary
from .physics.incompressible import make_incompressible
from .solver import LinearSolveConfig, LinearSolveStats, solve_linear

__version__ = "0.1.0"

__all__ = [
    "CenteredField",
    "GridSpec",
    "LinearSolveConfig",
    "LinearSolveStats",
    "MACField",
    "SolidMask",
    "VelocityBoundary",
    "__version__",
    "apply_velocity_boundary",
    "make_incompressible",
    "solve_linear",
]
