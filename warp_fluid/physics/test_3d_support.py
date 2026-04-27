from __future__ import annotations

import math

import numpy as np
import warp as wp

from ..core import CenteredField, GridSpec, MACField, VelocityBoundary
from ..geom import tesla_valve_levelset_3d
from ..ops.mask import solid_mask_from_levelset
from .advect import advect_mac_semi_lagrangian
from .force import diffuse_velocity_explicit
from .incompressible import make_incompressible


def test_tesla_valve_levelset_3d_produces_expected_shape() -> None:
    grid = GridSpec(nx=24, ny=12, nz=8, dx=1.0, dy=1.0, dz=1.0)
    levelset = tesla_valve_levelset_3d(
        grid,
        center=(12.0, 6.0, 4.0),
        d0=4.0,
        d1=6.0,
        d2=2.0,
        theta=math.radians(35.0),
        depth=6.0,
        num_units=1,
        include_end_pipes=True,
    )

    assert levelset.shape == (24, 12, 8)
    assert np.any(levelset > 0.0)
    assert np.any(levelset <= 0.0)
    assert np.any(levelset[0, :, :] > 0.0)
    assert np.any(levelset[-1, :, :] > 0.0)


def test_3d_velocity_step_smoke() -> None:
    wp.init()
    grid = GridSpec(nx=32, ny=16, nz=6, dx=1.0, dy=1.0, dz=1.0)
    levelset = tesla_valve_levelset_3d(
        grid,
        center=(16.0, 8.0, 3.0),
        d0=3.0,
        d1=5.0,
        d2=2.0,
        theta=math.radians(35.0),
        depth=4.0,
        num_units=1,
        include_end_pipes=True,
    )
    solid = solid_mask_from_levelset(grid, levelset, thickness=0.25)
    velocity = MACField.zeros(grid)
    pressure = CenteredField.zeros(grid)

    velocity.u.fill_(0.1)
    velocity = advect_mac_semi_lagrangian(velocity, 0.1, solid=solid)
    velocity = diffuse_velocity_explicit(velocity, 1.0e-3, 0.1, solid=solid)
    velocity, pressure, _ = make_incompressible(
        velocity,
        solid=solid,
        pressure=pressure,
        boundary=VelocityBoundary(inflow_west=(0.1, 0.0, 0.0)),
    )

    assert velocity.u.shape == grid.shape_u
    assert velocity.v.shape == grid.shape_v
    assert velocity.w is not None and velocity.w.shape == grid.shape_w
    assert pressure.data.shape == grid.shape
