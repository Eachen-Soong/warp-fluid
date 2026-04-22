from __future__ import annotations

import numpy as np

from ..core import GridSpec
from ..geom.levelset_grid import sphere_levelset
from .mask import solid_mask_from_levelset


def test_fractional_solid_mask_contains_partial_cell_and_face_fractions() -> None:
    grid = GridSpec.from_extent(32, 24, 32.0, 24.0)
    levelset = sphere_levelset(grid, center=(10.3, 12.1), radius=4.7)
    solid = solid_mask_from_levelset(grid, levelset, fractional=True)

    assert np.any((solid.cell_numpy > 0.0) & (solid.cell_numpy < 1.0))
    assert np.any((solid.u_numpy > 0.0) & (solid.u_numpy < 1.0))
    assert np.any((solid.v_numpy > 0.0) & (solid.v_numpy < 1.0))


def test_binary_solid_mask_stays_binary() -> None:
    grid = GridSpec.from_extent(32, 24, 32.0, 24.0)
    levelset = sphere_levelset(grid, center=(10.3, 12.1), radius=4.7)
    solid = solid_mask_from_levelset(grid, levelset, fractional=False)

    assert set(np.unique(solid.cell_numpy)).issubset({0.0, 1.0})
    assert set(np.unique(solid.u_numpy)).issubset({0.0, 1.0})
    assert set(np.unique(solid.v_numpy)).issubset({0.0, 1.0})
