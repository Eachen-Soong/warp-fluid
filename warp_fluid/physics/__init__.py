"""Physics building blocks for Warp-only fluid simulations."""

from .advect import advect_centered_mac_cormack, advect_centered_semi_lagrangian, advect_mac_semi_lagrangian
from .force import add_buoyancy, add_constant_force, diffuse_velocity_explicit
from .incompressible import make_incompressible

__all__ = [
    "add_buoyancy",
    "add_constant_force",
    "advect_centered_mac_cormack",
    "advect_centered_semi_lagrangian",
    "advect_mac_semi_lagrangian",
    "diffuse_velocity_explicit",
    "make_incompressible",
]

try:
    from .smoke import SmokeState, smoke_step
except ModuleNotFoundError:
    SmokeState = None
    smoke_step = None
else:
    __all__.extend(["SmokeState", "smoke_step"])
