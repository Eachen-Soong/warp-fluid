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
def _trilerp_clamped(
    field: wp.array3d(dtype=wp.float32),
    gx: float,
    gy: float,
    gz: float,
    i_max: int,
    j_max: int,
    k_max: int,
) -> float:
    gx = wp.clamp(gx, 0.0, float(i_max))
    gy = wp.clamp(gy, 0.0, float(j_max))
    gz = wp.clamp(gz, 0.0, float(k_max))
    i0 = int(wp.floor(gx))
    j0 = int(wp.floor(gy))
    k0 = int(wp.floor(gz))
    i1 = wp.min(i0 + 1, i_max)
    j1 = wp.min(j0 + 1, j_max)
    k1 = wp.min(k0 + 1, k_max)
    tx = gx - float(i0)
    ty = gy - float(j0)
    tz = gz - float(k0)
    c000 = field[i0, j0, k0]
    c100 = field[i1, j0, k0]
    c010 = field[i0, j1, k0]
    c110 = field[i1, j1, k0]
    c001 = field[i0, j0, k1]
    c101 = field[i1, j0, k1]
    c011 = field[i0, j1, k1]
    c111 = field[i1, j1, k1]
    c00 = c000 * (1.0 - tx) + c100 * tx
    c10 = c010 * (1.0 - tx) + c110 * tx
    c01 = c001 * (1.0 - tx) + c101 * tx
    c11 = c011 * (1.0 - tx) + c111 * tx
    c0 = c00 * (1.0 - ty) + c10 * ty
    c1 = c01 * (1.0 - ty) + c11 * ty
    return c0 * (1.0 - tz) + c1 * tz


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


@wp.func
def sample_centered_3d(
    field: wp.array3d(dtype=wp.float32),
    x: float,
    y: float,
    z: float,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
) -> float:
    gx = (x - origin_x) / dx - 0.5
    gy = (y - origin_y) / dy - 0.5
    gz = (z - origin_z) / dz - 0.5
    return _trilerp_clamped(field, gx, gy, gz, nx - 1, ny - 1, nz - 1)


@wp.func
def sample_u_face_3d(
    field: wp.array3d(dtype=wp.float32),
    x: float,
    y: float,
    z: float,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
) -> float:
    gx = (x - origin_x) / dx
    gy = (y - origin_y) / dy - 0.5
    gz = (z - origin_z) / dz - 0.5
    return _trilerp_clamped(field, gx, gy, gz, nx, ny - 1, nz - 1)


@wp.func
def sample_v_face_3d(
    field: wp.array3d(dtype=wp.float32),
    x: float,
    y: float,
    z: float,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
) -> float:
    gx = (x - origin_x) / dx - 0.5
    gy = (y - origin_y) / dy
    gz = (z - origin_z) / dz - 0.5
    return _trilerp_clamped(field, gx, gy, gz, nx - 1, ny, nz - 1)


@wp.func
def sample_w_face_3d(
    field: wp.array3d(dtype=wp.float32),
    x: float,
    y: float,
    z: float,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
) -> float:
    gx = (x - origin_x) / dx - 0.5
    gy = (y - origin_y) / dy - 0.5
    gz = (z - origin_z) / dz
    return _trilerp_clamped(field, gx, gy, gz, nx - 1, ny - 1, nz)


@wp.func
def sample_mac_velocity_3d(
    u: wp.array3d(dtype=wp.float32),
    v: wp.array3d(dtype=wp.float32),
    w: wp.array3d(dtype=wp.float32),
    x: float,
    y: float,
    z: float,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
) -> wp.vec3:
    return wp.vec3(
        sample_u_face_3d(u, x, y, z, origin_x, origin_y, origin_z, nx, ny, nz, dx, dy, dz),
        sample_v_face_3d(v, x, y, z, origin_x, origin_y, origin_z, nx, ny, nz, dx, dy, dz),
        sample_w_face_3d(w, x, y, z, origin_x, origin_y, origin_z, nx, ny, nz, dx, dy, dz),
    )
