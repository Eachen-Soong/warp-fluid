from __future__ import annotations

import math

import numpy as np

from .airfoil import naca4_airfoil_levelset, naca4_airfoil_polygon
from .tesla_valve import tesla_valve_fluid_levelset, tesla_valve_levelset


def test_tesla_valve_levelsets_are_opposites() -> None:
    grid = (180, 120, 0.5, 0.5)
    fluid = tesla_valve_fluid_levelset(
        grid,
        center=(45.0, 30.0),
        d0=18.0,
        d1=24.0,
        d2=12.0,
        theta=math.radians(35.0),
        num_units=1,
        arc_segments=32,
    )
    obstacle = tesla_valve_levelset(
        grid,
        center=(45.0, 30.0),
        d0=18.0,
        d1=24.0,
        d2=12.0,
        theta=math.radians(35.0),
        num_units=1,
        arc_segments=32,
    )

    assert fluid.shape == obstacle.shape == (180, 120)
    assert np.allclose(obstacle, -fluid)
    assert np.any(fluid < 0.0)
    assert np.any(obstacle < 0.0)


def test_tesla_valve_more_pairs_increase_fluid_area() -> None:
    grid = (320, 180, 0.5, 0.5)
    kwargs = dict(
        center=(80.0, 45.0),
        d0=18.0,
        d1=24.0,
        d2=12.0,
        theta=math.radians(35.0),
        arc_segments=32,
    )

    single = tesla_valve_fluid_levelset(grid, num_units=1, **kwargs)
    double = tesla_valve_fluid_levelset(grid, num_units=2, **kwargs)

    assert np.count_nonzero(double < 0.0) > np.count_nonzero(single < 0.0)


def test_tesla_valve_end_pipes_reach_domain_boundaries() -> None:
    grid = (480, 160, 1.0, 1.0)
    fluid = tesla_valve_fluid_levelset(
        grid,
        center=(240.0, 80.0),
        d0=18.0,
        d1=24.0,
        d2=12.0,
        theta=math.radians(35.0),
        num_units=2,
        arc_segments=32,
        include_end_pipes=True,
    )

    assert np.any(fluid[0, :] < 0.0)
    assert np.any(fluid[-1, :] < 0.0)


def test_tesla_valve_end_pipes_reject_negative_pipe_length() -> None:
    grid = (80, 120, 0.5, 0.5)

    try:
        tesla_valve_fluid_levelset(
            grid,
            center=(5.0, 30.0),
            d0=18.0,
            d1=24.0,
            d2=12.0,
            theta=math.radians(35.0),
            num_units=2,
            arc_segments=32,
            include_end_pipes=True,
        )
    except ValueError as exc:
        assert "negative" in str(exc)
    else:
        raise AssertionError("Expected a negative end-pipe length to raise ValueError.")


def test_naca4_airfoil_polygon_spans_expected_chord() -> None:
    polygon = naca4_airfoil_polygon("2412", chord=1.5, leading_edge=(2.0, 3.0), samples=128)

    assert polygon.shape[1] == 2
    assert np.isclose(float(np.min(polygon[:, 0])), 2.0, atol=2.0e-3)
    assert np.isclose(float(np.max(polygon[:, 0])), 3.5, atol=2.0e-3)


def test_naca4_airfoil_levelset_marks_airfoil_interior() -> None:
    grid = (320, 160, 0.02, 0.02)
    levelset = naca4_airfoil_levelset(
        grid,
        "0012",
        chord=1.0,
        leading_edge=(2.0, 1.6),
        samples=256,
    )

    assert np.any(levelset < 0.0)
    mid_i = int((2.3 / 0.02) - 0.5)
    mid_j = int((1.6 / 0.02) - 0.5)
    far_i = int((0.5 / 0.02) - 0.5)
    far_j = int((0.5 / 0.02) - 0.5)
    assert levelset[mid_i, mid_j] < 0.0
    assert levelset[far_i, far_j] > 0.0


def test_naca4_airfoil_rotation_changes_vertical_extent() -> None:
    base = naca4_airfoil_polygon("0012", chord=1.0, leading_edge=(0.0, 0.0), angle=0.0, samples=128)
    rotated = naca4_airfoil_polygon(
        "0012",
        chord=1.0,
        leading_edge=(0.0, 0.0),
        angle=math.radians(12.0),
        samples=128,
    )

    base_height = float(np.max(base[:, 1]) - np.min(base[:, 1]))
    rotated_height = float(np.max(rotated[:, 1]) - np.min(rotated[:, 1]))
    assert rotated_height > base_height
