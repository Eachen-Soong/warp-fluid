from __future__ import annotations

import warp as wp


@wp.func
def _bilerp_clamped(
    field: wp.array2d(dtype=wp.float32),
    gx: float,
    gy: float,
    i_max: int,
    j_max: int,
) -> float:
    gx = wp.clamp(gx, 0.0, float(i_max))
    gy = wp.clamp(gy, 0.0, float(j_max))
    i0 = int(wp.floor(gx))
    j0 = int(wp.floor(gy))
    i1 = wp.min(i0 + 1, i_max)
    j1 = wp.min(j0 + 1, j_max)
    tx = gx - float(i0)
    ty = gy - float(j0)
    f00 = field[i0, j0]
    f10 = field[i1, j0]
    f01 = field[i0, j1]
    f11 = field[i1, j1]
    f0 = f00 * (1.0 - tx) + f10 * tx
    f1 = f01 * (1.0 - tx) + f11 * tx
    return f0 * (1.0 - ty) + f1 * ty


@wp.func
def sample_centered(
    field: wp.array2d(dtype=wp.float32),
    x: float,
    y: float,
    origin_x: float,
    origin_y: float,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
) -> float:
    gx = (x - origin_x) / dx - 0.5
    gy = (y - origin_y) / dy - 0.5
    return _bilerp_clamped(field, gx, gy, nx - 1, ny - 1)


@wp.func
def sample_u_face(
    field: wp.array2d(dtype=wp.float32),
    x: float,
    y: float,
    origin_x: float,
    origin_y: float,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
) -> float:
    gx = (x - origin_x) / dx
    gy = (y - origin_y) / dy - 0.5
    return _bilerp_clamped(field, gx, gy, nx, ny - 1)


@wp.func
def sample_v_face(
    field: wp.array2d(dtype=wp.float32),
    x: float,
    y: float,
    origin_x: float,
    origin_y: float,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
) -> float:
    gx = (x - origin_x) / dx - 0.5
    gy = (y - origin_y) / dy
    return _bilerp_clamped(field, gx, gy, nx - 1, ny)


@wp.func
def sample_mac_velocity(
    u: wp.array2d(dtype=wp.float32),
    v: wp.array2d(dtype=wp.float32),
    x: float,
    y: float,
    origin_x: float,
    origin_y: float,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
) -> wp.vec2:
    return wp.vec2(
        sample_u_face(u, x, y, origin_x, origin_y, nx, ny, dx, dy),
        sample_v_face(v, x, y, origin_x, origin_y, nx, ny, dx, dy),
    )
