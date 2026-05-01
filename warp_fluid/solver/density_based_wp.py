from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import warp as wp

from ..core.grid import GridSpec
from .pressure_based_np import (
    AerodynamicCoefficients,
    FreestreamCondition,
    _dynamic_pressure,
    _face_aperture_x,
    _face_aperture_y,
    _freestream_velocity,
    _reference_point,
    _smooth_heaviside,
)


@dataclass(frozen=True)
class DensityBasedSolverConfig:
    pseudo_steps: int = 4000
    cfl: float = 0.4
    dt_safety: float = 0.7
    gamma: float = 1.4
    gas_constant: float = 287.05
    min_density: float = 1.0e-6
    min_pressure: float = 1.0
    min_fluid_fraction: float = 0.2
    max_velocity_factor: float = 6.0
    convergence_tolerance: float = 1.0e-4
    force_coefficient_tolerance: float = 1.0e-4
    convergence_window: int = 10


@dataclass(frozen=True)
class DensityBasedSnapshot:
    step: int
    dt: float
    density_residual: float
    momentum_residual: float
    energy_residual: float
    normalized_density: float
    normalized_momentum: float
    normalized_energy: float
    drag_coefficient: float
    lift_coefficient: float
    moment_coefficient: float
    force_delta: float


@dataclass
class DensityBasedState:
    density: object
    momentum_x: object
    momentum_y: object
    total_energy: object
    fluid: object
    fluid_fraction: object
    aperture_x: object
    aperture_y: object
    solid_levelset: object
    dt: float = 0.0

    def density_numpy(self) -> np.ndarray:
        return np.asarray(self.density.numpy(), dtype=np.float64)

    def momentum_x_numpy(self) -> np.ndarray:
        return np.asarray(self.momentum_x.numpy(), dtype=np.float64)

    def momentum_y_numpy(self) -> np.ndarray:
        return np.asarray(self.momentum_y.numpy(), dtype=np.float64)

    def total_energy_numpy(self) -> np.ndarray:
        return np.asarray(self.total_energy.numpy(), dtype=np.float64)

    def fluid_numpy(self) -> np.ndarray:
        return np.asarray(self.fluid.numpy(), dtype=np.float32) > 0.5

    def fluid_fraction_numpy(self) -> np.ndarray:
        return np.asarray(self.fluid_fraction.numpy(), dtype=np.float64)

    def aperture_x_numpy(self) -> np.ndarray:
        return np.asarray(self.aperture_x.numpy(), dtype=np.float64)

    def aperture_y_numpy(self) -> np.ndarray:
        return np.asarray(self.aperture_y.numpy(), dtype=np.float64)

    def solid_levelset_numpy(self) -> np.ndarray:
        return np.asarray(self.solid_levelset.numpy(), dtype=np.float64)

    def primitive_fields(self, gamma: float, gas_constant: float) -> dict[str, np.ndarray]:
        rho = self.density_numpy()
        rhou = self.momentum_x_numpy()
        rhov = self.momentum_y_numpy()
        rhoe = self.total_energy_numpy()
        u = np.divide(rhou, rho, out=np.zeros_like(rhou), where=rho > 1.0e-12)
        v = np.divide(rhov, rho, out=np.zeros_like(rhov), where=rho > 1.0e-12)
        kinetic = 0.5 * (rhou * rhou + rhov * rhov) / np.maximum(rho, 1.0e-12)
        pressure = np.maximum((gamma - 1.0) * (rhoe - kinetic), 1.0)
        temperature = pressure / np.maximum(rho * gas_constant, 1.0e-12)
        sound = np.sqrt(np.maximum(gamma * gas_constant * temperature, 1.0e-12))
        mach = np.sqrt(u * u + v * v) / np.maximum(sound, 1.0e-12)
        return {
            "density": rho,
            "u": u,
            "v": v,
            "pressure": pressure,
            "temperature": temperature,
            "mach": mach,
            "fluid": self.fluid_numpy(),
            "fluid_fraction": self.fluid_fraction_numpy(),
            "aperture_x": self.aperture_x_numpy(),
            "aperture_y": self.aperture_y_numpy(),
            "solid_levelset": self.solid_levelset_numpy(),
        }


@dataclass
class _DensityBasedScratch:
    rho_next: object
    rhou_next: object
    rhov_next: object
    rhoe_next: object
    flux_rho_x: object
    flux_rhou_x: object
    flux_rhov_x: object
    flux_rhoe_x: object
    flux_rho_y: object
    flux_rhou_y: object
    flux_rhov_y: object
    flux_rhoe_y: object
    rho_prev: object
    rhou_prev: object
    rhov_prev: object
    rhoe_prev: object
    scalar_buffer: object
    pair_buffer: object
    triple_buffer: object


@wp.kernel
def _fill_scalar_buffer_kernel(value: float, out: wp.array(dtype=wp.float32), count: int):
    index = wp.tid()
    if index < count:
        out[index] = value


@wp.kernel
def _copy_field_kernel(src: wp.array2d(dtype=wp.float32), dst: wp.array2d(dtype=wp.float32), nx: int, ny: int):
    i, j = wp.tid()
    if i < nx and j < ny:
        dst[i, j] = src[i, j]


@wp.func
def _cell_fraction(fluid_fraction: wp.array2d(dtype=wp.float32), fluid: wp.array2d(dtype=wp.float32), i: int, j: int) -> float:
    if fluid[i, j] <= 0.5:
        return 0.0
    return wp.max(fluid_fraction[i, j], 0.2)


@wp.func
def _is_finite_scalar(value: float) -> bool:
    return not wp.isnan(value) and not wp.isinf(value)


@wp.func
def _is_valid_conserved_state(
    rho: float,
    rhou: float,
    rhov: float,
    rhoe: float,
    gamma: float,
    rho_floor: float,
    p_floor: float,
    velocity_limit_sq: float,
) -> bool:
    if not _is_finite_scalar(rho) or not _is_finite_scalar(rhou) or not _is_finite_scalar(rhov) or not _is_finite_scalar(rhoe):
        return False
    if rho < rho_floor:
        return False
    inv_rho = 1.0 / wp.max(rho, 1.0e-12)
    speed_sq = (rhou * rhou + rhov * rhov) * inv_rho * inv_rho
    if not _is_finite_scalar(speed_sq) or speed_sq > velocity_limit_sq:
        return False
    kinetic = 0.5 * (rhou * rhou + rhov * rhov) * inv_rho
    pressure = (gamma - 1.0) * (rhoe - kinetic)
    return _is_finite_scalar(kinetic) and _is_finite_scalar(pressure) and pressure >= p_floor


@wp.func
def _pressure_from_conserved(rho: float, rhou: float, rhov: float, rhoe: float, gamma: float, p_floor: float) -> float:
    kinetic = 0.5 * (rhou * rhou + rhov * rhov) / wp.max(rho, 1.0e-12)
    return wp.max((gamma - 1.0) * (rhoe - kinetic), p_floor)


@wp.func
def _energy_from_primitive(rho: float, u: float, v: float, pressure: float, gamma: float) -> float:
    return pressure / (gamma - 1.0) + 0.5 * rho * (u * u + v * v)


@wp.func
def _copy_state_from_cell(
    density: wp.array2d(dtype=wp.float32),
    momentum_x: wp.array2d(dtype=wp.float32),
    momentum_y: wp.array2d(dtype=wp.float32),
    total_energy: wp.array2d(dtype=wp.float32),
    i: int,
    j: int,
) -> wp.vec4:
    return wp.vec4(density[i, j], momentum_x[i, j], momentum_y[i, j], total_energy[i, j])


@wp.func
def _regular_fluid_cell(fluid: wp.array2d(dtype=wp.float32), fluid_fraction: wp.array2d(dtype=wp.float32), i: int, j: int) -> bool:
    return fluid[i, j] > 0.5 and fluid_fraction[i, j] > 0.999


@wp.func
def _minmod3(a: float, b: float, c: float) -> float:
    if a > 0.0 and b > 0.0 and c > 0.0:
        return wp.min(a, wp.min(b, c))
    if a < 0.0 and b < 0.0 and c < 0.0:
        return wp.max(a, wp.max(b, c))
    return 0.0


@wp.func
def _mc_slope_scalar(delta_minus: float, delta_plus: float) -> float:
    return _minmod3(0.5 * (delta_minus + delta_plus), 2.0 * delta_minus, 2.0 * delta_plus)


@wp.func
def _mc_slope_vec4(left: wp.vec4, center: wp.vec4, right: wp.vec4) -> wp.vec4:
    delta_minus = center - left
    delta_plus = right - center
    return wp.vec4(
        _mc_slope_scalar(delta_minus[0], delta_plus[0]),
        _mc_slope_scalar(delta_minus[1], delta_plus[1]),
        _mc_slope_scalar(delta_minus[2], delta_plus[2]),
        _mc_slope_scalar(delta_minus[3], delta_plus[3]),
    )


@wp.func
def _enforce_state_floors(state: wp.vec4, gamma: float, rho_floor: float, p_floor: float) -> wp.vec4:
    rho = wp.max(state[0], rho_floor)
    rhou = state[1]
    rhov = state[2]
    rhoe = state[3]
    kinetic = 0.5 * (rhou * rhou + rhov * rhov) / wp.max(rho, 1.0e-12)
    rhoe = wp.max(rhoe, kinetic + p_floor / (gamma - 1.0))
    return wp.vec4(rho, rhou, rhov, rhoe)


@wp.func
def _reconstruct_face_states_x(
    density: wp.array2d(dtype=wp.float32),
    momentum_x: wp.array2d(dtype=wp.float32),
    momentum_y: wp.array2d(dtype=wp.float32),
    total_energy: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    fluid_fraction: wp.array2d(dtype=wp.float32),
    i: int,
    j: int,
    nx: int,
    gamma: float,
    rho_floor: float,
    p_floor: float,
) -> wp.vec4:
    il = i - 1
    ir = i
    left_state = _copy_state_from_cell(density, momentum_x, momentum_y, total_energy, il, j)
    right_state = _copy_state_from_cell(density, momentum_x, momentum_y, total_energy, ir, j)

    if i < 2 or i > nx - 2:
        return wp.vec4(left_state[0], left_state[1], left_state[2], left_state[3])

    if not (
        _regular_fluid_cell(fluid, fluid_fraction, il - 1, j)
        and _regular_fluid_cell(fluid, fluid_fraction, il, j)
        and _regular_fluid_cell(fluid, fluid_fraction, ir, j)
        and _regular_fluid_cell(fluid, fluid_fraction, ir + 1, j)
    ):
        return wp.vec4(left_state[0], left_state[1], left_state[2], left_state[3])

    state_ll = _copy_state_from_cell(density, momentum_x, momentum_y, total_energy, il - 1, j)
    state_rr = _copy_state_from_cell(density, momentum_x, momentum_y, total_energy, ir + 1, j)
    slope_left = _mc_slope_vec4(state_ll, left_state, right_state)
    slope_right = _mc_slope_vec4(left_state, right_state, state_rr)
    left_face = _enforce_state_floors(left_state + 0.5 * slope_left, gamma, rho_floor, p_floor)
    right_face = _enforce_state_floors(right_state - 0.5 * slope_right, gamma, rho_floor, p_floor)
    return wp.vec4(left_face[0], left_face[1], left_face[2], left_face[3]) + 0.0 * right_face


@wp.func
def _reconstruct_face_states_x_right(
    density: wp.array2d(dtype=wp.float32),
    momentum_x: wp.array2d(dtype=wp.float32),
    momentum_y: wp.array2d(dtype=wp.float32),
    total_energy: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    fluid_fraction: wp.array2d(dtype=wp.float32),
    i: int,
    j: int,
    nx: int,
    gamma: float,
    rho_floor: float,
    p_floor: float,
) -> wp.vec4:
    il = i - 1
    ir = i
    left_state = _copy_state_from_cell(density, momentum_x, momentum_y, total_energy, il, j)
    right_state = _copy_state_from_cell(density, momentum_x, momentum_y, total_energy, ir, j)

    if i < 2 or i > nx - 2:
        return right_state

    if not (
        _regular_fluid_cell(fluid, fluid_fraction, il - 1, j)
        and _regular_fluid_cell(fluid, fluid_fraction, il, j)
        and _regular_fluid_cell(fluid, fluid_fraction, ir, j)
        and _regular_fluid_cell(fluid, fluid_fraction, ir + 1, j)
    ):
        return right_state

    state_ll = _copy_state_from_cell(density, momentum_x, momentum_y, total_energy, il - 1, j)
    state_rr = _copy_state_from_cell(density, momentum_x, momentum_y, total_energy, ir + 1, j)
    slope_left = _mc_slope_vec4(state_ll, left_state, right_state)
    slope_right = _mc_slope_vec4(left_state, right_state, state_rr)
    left_face = _enforce_state_floors(left_state + 0.5 * slope_left, gamma, rho_floor, p_floor)
    right_face = _enforce_state_floors(right_state - 0.5 * slope_right, gamma, rho_floor, p_floor)
    return right_face + 0.0 * left_face


@wp.func
def _reconstruct_face_states_y_bottom(
    density: wp.array2d(dtype=wp.float32),
    momentum_x: wp.array2d(dtype=wp.float32),
    momentum_y: wp.array2d(dtype=wp.float32),
    total_energy: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    fluid_fraction: wp.array2d(dtype=wp.float32),
    i: int,
    j: int,
    ny: int,
    gamma: float,
    rho_floor: float,
    p_floor: float,
) -> wp.vec4:
    jb = j - 1
    jt = j
    bottom_state = _copy_state_from_cell(density, momentum_x, momentum_y, total_energy, i, jb)
    top_state = _copy_state_from_cell(density, momentum_x, momentum_y, total_energy, i, jt)

    if j < 2 or j > ny - 2:
        return bottom_state

    if not (
        _regular_fluid_cell(fluid, fluid_fraction, i, jb - 1)
        and _regular_fluid_cell(fluid, fluid_fraction, i, jb)
        and _regular_fluid_cell(fluid, fluid_fraction, i, jt)
        and _regular_fluid_cell(fluid, fluid_fraction, i, jt + 1)
    ):
        return bottom_state

    state_bb = _copy_state_from_cell(density, momentum_x, momentum_y, total_energy, i, jb - 1)
    state_tt = _copy_state_from_cell(density, momentum_x, momentum_y, total_energy, i, jt + 1)
    slope_bottom = _mc_slope_vec4(state_bb, bottom_state, top_state)
    slope_top = _mc_slope_vec4(bottom_state, top_state, state_tt)
    bottom_face = _enforce_state_floors(bottom_state + 0.5 * slope_bottom, gamma, rho_floor, p_floor)
    top_face = _enforce_state_floors(top_state - 0.5 * slope_top, gamma, rho_floor, p_floor)
    return bottom_face + 0.0 * top_face


@wp.func
def _reconstruct_face_states_y_top(
    density: wp.array2d(dtype=wp.float32),
    momentum_x: wp.array2d(dtype=wp.float32),
    momentum_y: wp.array2d(dtype=wp.float32),
    total_energy: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    fluid_fraction: wp.array2d(dtype=wp.float32),
    i: int,
    j: int,
    ny: int,
    gamma: float,
    rho_floor: float,
    p_floor: float,
) -> wp.vec4:
    jb = j - 1
    jt = j
    bottom_state = _copy_state_from_cell(density, momentum_x, momentum_y, total_energy, i, jb)
    top_state = _copy_state_from_cell(density, momentum_x, momentum_y, total_energy, i, jt)

    if j < 2 or j > ny - 2:
        return top_state

    if not (
        _regular_fluid_cell(fluid, fluid_fraction, i, jb - 1)
        and _regular_fluid_cell(fluid, fluid_fraction, i, jb)
        and _regular_fluid_cell(fluid, fluid_fraction, i, jt)
        and _regular_fluid_cell(fluid, fluid_fraction, i, jt + 1)
    ):
        return top_state

    state_bb = _copy_state_from_cell(density, momentum_x, momentum_y, total_energy, i, jb - 1)
    state_tt = _copy_state_from_cell(density, momentum_x, momentum_y, total_energy, i, jt + 1)
    slope_bottom = _mc_slope_vec4(state_bb, bottom_state, top_state)
    slope_top = _mc_slope_vec4(bottom_state, top_state, state_tt)
    bottom_face = _enforce_state_floors(bottom_state + 0.5 * slope_bottom, gamma, rho_floor, p_floor)
    top_face = _enforce_state_floors(top_state - 0.5 * slope_top, gamma, rho_floor, p_floor)
    return top_face + 0.0 * bottom_face


@wp.func
def _flux_x_from_state(state: wp.vec4, gamma: float, p_floor: float) -> wp.vec4:
    rho = state[0]
    rhou = state[1]
    rhov = state[2]
    rhoe = state[3]
    u = rhou / wp.max(rho, 1.0e-12)
    v = rhov / wp.max(rho, 1.0e-12)
    p = _pressure_from_conserved(rho, rhou, rhov, rhoe, gamma, p_floor)
    return wp.vec4(rhou, rhou * u + p, rhou * v, u * (rhoe + p))


@wp.func
def _flux_y_from_state(state: wp.vec4, gamma: float, p_floor: float) -> wp.vec4:
    rho = state[0]
    rhou = state[1]
    rhov = state[2]
    rhoe = state[3]
    u = rhou / wp.max(rho, 1.0e-12)
    v = rhov / wp.max(rho, 1.0e-12)
    p = _pressure_from_conserved(rho, rhou, rhov, rhoe, gamma, p_floor)
    return wp.vec4(rhov, rhov * u, rhov * v + p, v * (rhoe + p))


@wp.func
def _sound_speed(state: wp.vec4, gamma: float, p_floor: float) -> float:
    rho = state[0]
    p = _pressure_from_conserved(state[0], state[1], state[2], state[3], gamma, p_floor)
    return wp.sqrt(wp.max(gamma * p / wp.max(rho, 1.0e-12), 1.0e-12))


@wp.func
def _rusanov_flux_x(left: wp.vec4, right: wp.vec4, gamma: float, p_floor: float) -> wp.vec4:
    flux_l = _flux_x_from_state(left, gamma, p_floor)
    flux_r = _flux_x_from_state(right, gamma, p_floor)
    a_l = _sound_speed(left, gamma, p_floor)
    a_r = _sound_speed(right, gamma, p_floor)
    u_l = left[1] / wp.max(left[0], 1.0e-12)
    u_r = right[1] / wp.max(right[0], 1.0e-12)
    s_max = wp.max(wp.abs(u_l) + a_l, wp.abs(u_r) + a_r)
    return 0.5 * (flux_l + flux_r) - 0.5 * s_max * (right - left)


@wp.func
def _rusanov_flux_y(left: wp.vec4, right: wp.vec4, gamma: float, p_floor: float) -> wp.vec4:
    flux_l = _flux_y_from_state(left, gamma, p_floor)
    flux_r = _flux_y_from_state(right, gamma, p_floor)
    a_l = _sound_speed(left, gamma, p_floor)
    a_r = _sound_speed(right, gamma, p_floor)
    v_l = left[2] / wp.max(left[0], 1.0e-12)
    v_r = right[2] / wp.max(right[0], 1.0e-12)
    s_max = wp.max(wp.abs(v_l) + a_l, wp.abs(v_r) + a_r)
    return 0.5 * (flux_l + flux_r) - 0.5 * s_max * (right - left)


@wp.func
def _wall_flux_x(pressure: float) -> wp.vec4:
    return wp.vec4(0.0, pressure, 0.0, 0.0)


@wp.func
def _wall_flux_y(pressure: float) -> wp.vec4:
    return wp.vec4(0.0, 0.0, pressure, 0.0)


@wp.kernel
def _face_flux_x_kernel(
    density: wp.array2d(dtype=wp.float32),
    momentum_x: wp.array2d(dtype=wp.float32),
    momentum_y: wp.array2d(dtype=wp.float32),
    total_energy: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    fluid_fraction: wp.array2d(dtype=wp.float32),
    aperture_x: wp.array2d(dtype=wp.float32),
    flux_rho_x: wp.array2d(dtype=wp.float32),
    flux_rhou_x: wp.array2d(dtype=wp.float32),
    flux_rhov_x: wp.array2d(dtype=wp.float32),
    flux_rhoe_x: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    gamma: float,
    rho_floor: float,
    p_floor: float,
    rho_inf: float,
    rhou_inf: float,
    rhov_inf: float,
    rhoe_inf: float,
):
    i, j = wp.tid()
    if i > nx or j >= ny:
        return

    farfield = wp.vec4(rho_inf, rhou_inf, rhov_inf, rhoe_inf)
    flux = wp.vec4(0.0, 0.0, 0.0, 0.0)
    aperture = aperture_x[i, j]

    if i == 0:
        right = _copy_state_from_cell(density, momentum_x, momentum_y, total_energy, 0, j)
        flux = _rusanov_flux_x(farfield, right, gamma, p_floor)
    elif i == nx:
        left = _copy_state_from_cell(density, momentum_x, momentum_y, total_energy, nx - 1, j)
        flux = _rusanov_flux_x(left, left, gamma, p_floor)
    else:
        left_fluid = fluid[i - 1, j] > 0.5
        right_fluid = fluid[i, j] > 0.5
        if left_fluid and right_fluid:
            left = _reconstruct_face_states_x(
                density, momentum_x, momentum_y, total_energy, fluid, fluid_fraction, i, j, nx, gamma, rho_floor, p_floor
            )
            right = _reconstruct_face_states_x_right(
                density, momentum_x, momentum_y, total_energy, fluid, fluid_fraction, i, j, nx, gamma, rho_floor, p_floor
            )
            p_left = _pressure_from_conserved(left[0], left[1], left[2], left[3], gamma, p_floor)
            p_right = _pressure_from_conserved(right[0], right[1], right[2], right[3], gamma, p_floor)
            open_flux = _rusanov_flux_x(left, right, gamma, p_floor)
            wall_flux = _wall_flux_x(0.5 * (p_left + p_right))
            flux = aperture * open_flux + (1.0 - aperture) * wall_flux
        elif left_fluid:
            left = _copy_state_from_cell(density, momentum_x, momentum_y, total_energy, i - 1, j)
            p_left = _pressure_from_conserved(left[0], left[1], left[2], left[3], gamma, p_floor)
            flux = _wall_flux_x(p_left)
        elif right_fluid:
            right = _copy_state_from_cell(density, momentum_x, momentum_y, total_energy, i, j)
            p_right = _pressure_from_conserved(right[0], right[1], right[2], right[3], gamma, p_floor)
            flux = _wall_flux_x(p_right)

    flux_rho_x[i, j] = flux[0]
    flux_rhou_x[i, j] = flux[1]
    flux_rhov_x[i, j] = flux[2]
    flux_rhoe_x[i, j] = flux[3]


@wp.kernel
def _face_flux_y_kernel(
    density: wp.array2d(dtype=wp.float32),
    momentum_x: wp.array2d(dtype=wp.float32),
    momentum_y: wp.array2d(dtype=wp.float32),
    total_energy: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    fluid_fraction: wp.array2d(dtype=wp.float32),
    aperture_y: wp.array2d(dtype=wp.float32),
    flux_rho_y: wp.array2d(dtype=wp.float32),
    flux_rhou_y: wp.array2d(dtype=wp.float32),
    flux_rhov_y: wp.array2d(dtype=wp.float32),
    flux_rhoe_y: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    gamma: float,
    rho_floor: float,
    p_floor: float,
    rho_inf: float,
    rhou_inf: float,
    rhov_inf: float,
    rhoe_inf: float,
):
    i, j = wp.tid()
    if i >= nx or j > ny:
        return

    farfield = wp.vec4(rho_inf, rhou_inf, rhov_inf, rhoe_inf)
    flux = wp.vec4(0.0, 0.0, 0.0, 0.0)
    aperture = aperture_y[i, j]

    if j == 0:
        top = _copy_state_from_cell(density, momentum_x, momentum_y, total_energy, i, 0)
        flux = _rusanov_flux_y(farfield, top, gamma, p_floor)
    elif j == ny:
        bottom = _copy_state_from_cell(density, momentum_x, momentum_y, total_energy, i, ny - 1)
        flux = _rusanov_flux_y(bottom, farfield, gamma, p_floor)
    else:
        bottom_fluid = fluid[i, j - 1] > 0.5
        top_fluid = fluid[i, j] > 0.5
        if bottom_fluid and top_fluid:
            bottom = _reconstruct_face_states_y_bottom(
                density, momentum_x, momentum_y, total_energy, fluid, fluid_fraction, i, j, ny, gamma, rho_floor, p_floor
            )
            top = _reconstruct_face_states_y_top(
                density, momentum_x, momentum_y, total_energy, fluid, fluid_fraction, i, j, ny, gamma, rho_floor, p_floor
            )
            p_bottom = _pressure_from_conserved(bottom[0], bottom[1], bottom[2], bottom[3], gamma, p_floor)
            p_top = _pressure_from_conserved(top[0], top[1], top[2], top[3], gamma, p_floor)
            open_flux = _rusanov_flux_y(bottom, top, gamma, p_floor)
            wall_flux = _wall_flux_y(0.5 * (p_bottom + p_top))
            flux = aperture * open_flux + (1.0 - aperture) * wall_flux
        elif bottom_fluid:
            bottom = _copy_state_from_cell(density, momentum_x, momentum_y, total_energy, i, j - 1)
            p_bottom = _pressure_from_conserved(bottom[0], bottom[1], bottom[2], bottom[3], gamma, p_floor)
            flux = _wall_flux_y(p_bottom)
        elif top_fluid:
            top = _copy_state_from_cell(density, momentum_x, momentum_y, total_energy, i, j)
            p_top = _pressure_from_conserved(top[0], top[1], top[2], top[3], gamma, p_floor)
            flux = _wall_flux_y(p_top)

    flux_rho_y[i, j] = flux[0]
    flux_rhou_y[i, j] = flux[1]
    flux_rhov_y[i, j] = flux[2]
    flux_rhoe_y[i, j] = flux[3]


@wp.kernel
def _update_conserved_kernel(
    density: wp.array2d(dtype=wp.float32),
    momentum_x: wp.array2d(dtype=wp.float32),
    momentum_y: wp.array2d(dtype=wp.float32),
    total_energy: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    fluid_fraction: wp.array2d(dtype=wp.float32),
    flux_rho_x: wp.array2d(dtype=wp.float32),
    flux_rhou_x: wp.array2d(dtype=wp.float32),
    flux_rhov_x: wp.array2d(dtype=wp.float32),
    flux_rhoe_x: wp.array2d(dtype=wp.float32),
    flux_rho_y: wp.array2d(dtype=wp.float32),
    flux_rhou_y: wp.array2d(dtype=wp.float32),
    flux_rhov_y: wp.array2d(dtype=wp.float32),
    flux_rhoe_y: wp.array2d(dtype=wp.float32),
    rho_next: wp.array2d(dtype=wp.float32),
    rhou_next: wp.array2d(dtype=wp.float32),
    rhov_next: wp.array2d(dtype=wp.float32),
    rhoe_next: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    dt: float,
    gamma: float,
    rho_floor: float,
    p_floor: float,
    min_fluid_fraction: float,
    rho_inf: float,
    rhou_inf: float,
    rhov_inf: float,
    rhoe_inf: float,
):
    i, j = wp.tid()
    if i >= nx or j >= ny:
        return
    if fluid[i, j] <= 0.5:
        rho_next[i, j] = rho_inf
        rhou_next[i, j] = rhou_inf
        rhov_next[i, j] = rhov_inf
        rhoe_next[i, j] = rhoe_inf
        return

    volume = wp.max(wp.max(_cell_fraction(fluid_fraction, fluid, i, j), min_fluid_fraction) * dx * dy, 1.0e-12)
    drho = ((flux_rho_x[i + 1, j] - flux_rho_x[i, j]) * dy + (flux_rho_y[i, j + 1] - flux_rho_y[i, j]) * dx) / volume
    drhou = ((flux_rhou_x[i + 1, j] - flux_rhou_x[i, j]) * dy + (flux_rhou_y[i, j + 1] - flux_rhou_y[i, j]) * dx) / volume
    drhov = ((flux_rhov_x[i + 1, j] - flux_rhov_x[i, j]) * dy + (flux_rhov_y[i, j + 1] - flux_rhov_y[i, j]) * dx) / volume
    drhoe = ((flux_rhoe_x[i + 1, j] - flux_rhoe_x[i, j]) * dy + (flux_rhoe_y[i, j + 1] - flux_rhoe_y[i, j]) * dx) / volume

    rho = wp.max(density[i, j] - dt * drho, rho_floor)
    rhou = momentum_x[i, j] - dt * drhou
    rhov = momentum_y[i, j] - dt * drhov
    rhoe = total_energy[i, j] - dt * drhoe

    kinetic = 0.5 * (rhou * rhou + rhov * rhov) / wp.max(rho, 1.0e-12)
    rhoe = wp.max(rhoe, kinetic + p_floor / (gamma - 1.0))

    rho_next[i, j] = rho
    rhou_next[i, j] = rhou
    rhov_next[i, j] = rhov
    rhoe_next[i, j] = rhoe


@wp.kernel
def _swap_next_kernel(
    rho_next: wp.array2d(dtype=wp.float32),
    rhou_next: wp.array2d(dtype=wp.float32),
    rhov_next: wp.array2d(dtype=wp.float32),
    rhoe_next: wp.array2d(dtype=wp.float32),
    density: wp.array2d(dtype=wp.float32),
    momentum_x: wp.array2d(dtype=wp.float32),
    momentum_y: wp.array2d(dtype=wp.float32),
    total_energy: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
):
    i, j = wp.tid()
    if i < nx and j < ny:
        density[i, j] = rho_next[i, j]
        momentum_x[i, j] = rhou_next[i, j]
        momentum_y[i, j] = rhov_next[i, j]
        total_energy[i, j] = rhoe_next[i, j]


@wp.kernel
def _sanitize_conserved_kernel(
    density: wp.array2d(dtype=wp.float32),
    momentum_x: wp.array2d(dtype=wp.float32),
    momentum_y: wp.array2d(dtype=wp.float32),
    total_energy: wp.array2d(dtype=wp.float32),
    rho_prev: wp.array2d(dtype=wp.float32),
    rhou_prev: wp.array2d(dtype=wp.float32),
    rhov_prev: wp.array2d(dtype=wp.float32),
    rhoe_prev: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    gamma: float,
    rho_floor: float,
    p_floor: float,
    velocity_limit_sq: float,
    rho_inf: float,
    rhou_inf: float,
    rhov_inf: float,
    rhoe_inf: float,
):
    i, j = wp.tid()
    if i >= nx or j >= ny:
        return
    if fluid[i, j] <= 0.5:
        density[i, j] = rho_inf
        momentum_x[i, j] = rhou_inf
        momentum_y[i, j] = rhov_inf
        total_energy[i, j] = rhoe_inf
        return

    rho = density[i, j]
    rhou = momentum_x[i, j]
    rhov = momentum_y[i, j]
    rhoe = total_energy[i, j]
    valid = _is_valid_conserved_state(rho, rhou, rhov, rhoe, gamma, rho_floor, p_floor, velocity_limit_sq)

    if not valid:
        rho = rho_prev[i, j]
        rhou = rhou_prev[i, j]
        rhov = rhov_prev[i, j]
        rhoe = rhoe_prev[i, j]
        valid = _is_valid_conserved_state(rho, rhou, rhov, rhoe, gamma, rho_floor, p_floor, velocity_limit_sq)
        if not valid:
            rho = rho_inf
            rhou = rhou_inf
            rhov = rhov_inf
            rhoe = rhoe_inf

    rho = wp.max(rho, rho_floor)
    inv_rho = 1.0 / wp.max(rho, 1.0e-12)
    speed_sq = (rhou * rhou + rhov * rhov) * inv_rho * inv_rho
    if speed_sq > velocity_limit_sq:
        scale = wp.sqrt(velocity_limit_sq / wp.max(speed_sq, 1.0e-12))
        rhou = rhou * scale
        rhov = rhov * scale
    kinetic = 0.5 * (rhou * rhou + rhov * rhov) / wp.max(rho, 1.0e-12)
    rhoe = wp.max(rhoe, kinetic + p_floor / (gamma - 1.0))

    density[i, j] = rho
    momentum_x[i, j] = rhou
    momentum_y[i, j] = rhov
    total_energy[i, j] = rhoe


@wp.kernel
def _apply_outer_boundaries_kernel(
    density: wp.array2d(dtype=wp.float32),
    momentum_x: wp.array2d(dtype=wp.float32),
    momentum_y: wp.array2d(dtype=wp.float32),
    total_energy: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    rho_inf: float,
    rhou_inf: float,
    rhov_inf: float,
    rhoe_inf: float,
):
    i, j = wp.tid()
    if i >= nx or j >= ny:
        return
    if fluid[i, j] <= 0.5:
        density[i, j] = rho_inf
        momentum_x[i, j] = rhou_inf
        momentum_y[i, j] = rhov_inf
        total_energy[i, j] = rhoe_inf
        return
    if i == 0 or j == 0 or j == ny - 1:
        density[i, j] = rho_inf
        momentum_x[i, j] = rhou_inf
        momentum_y[i, j] = rhov_inf
        total_energy[i, j] = rhoe_inf
    elif i == nx - 1:
        density[i, j] = density[i - 1, j]
        momentum_x[i, j] = momentum_x[i - 1, j]
        momentum_y[i, j] = momentum_y[i - 1, j]
        total_energy[i, j] = total_energy[i - 1, j]


@wp.kernel
def _max_wave_speed_kernel(
    density: wp.array2d(dtype=wp.float32),
    momentum_x: wp.array2d(dtype=wp.float32),
    momentum_y: wp.array2d(dtype=wp.float32),
    total_energy: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
    nx: int,
    ny: int,
    gamma: float,
    p_floor: float,
):
    i, j = wp.tid()
    if i >= nx or j >= ny or fluid[i, j] <= 0.5:
        return
    rho = density[i, j]
    rhou = momentum_x[i, j]
    rhov = momentum_y[i, j]
    rhoe = total_energy[i, j]
    u = rhou / wp.max(rho, 1.0e-12)
    v = rhov / wp.max(rho, 1.0e-12)
    p = _pressure_from_conserved(rho, rhou, rhov, rhoe, gamma, p_floor)
    a = wp.sqrt(wp.max(gamma * p / wp.max(rho, 1.0e-12), 1.0e-12))
    speed = wp.max(wp.abs(u) + a, wp.abs(v) + a)
    if not wp.isnan(speed) and not wp.isinf(speed):
        wp.atomic_max(out, 0, speed)


@wp.kernel
def _density_residual_kernel(
    density: wp.array2d(dtype=wp.float32),
    rho_prev: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
    nx: int,
    ny: int,
):
    i, j = wp.tid()
    if i >= nx or j >= ny or fluid[i, j] <= 0.5:
        return
    value = wp.abs(density[i, j] - rho_prev[i, j])
    if not _is_finite_scalar(value):
        value = 1.0e30
    wp.atomic_max(out, 0, value)


@wp.kernel
def _momentum_residual_kernel(
    momentum_x: wp.array2d(dtype=wp.float32),
    rhou_prev: wp.array2d(dtype=wp.float32),
    momentum_y: wp.array2d(dtype=wp.float32),
    rhov_prev: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
    nx: int,
    ny: int,
):
    i, j = wp.tid()
    if i >= nx or j >= ny or fluid[i, j] <= 0.5:
        return
    value_x = wp.abs(momentum_x[i, j] - rhou_prev[i, j])
    value_y = wp.abs(momentum_y[i, j] - rhov_prev[i, j])
    if not _is_finite_scalar(value_x):
        value_x = 1.0e30
    if not _is_finite_scalar(value_y):
        value_y = 1.0e30
    wp.atomic_max(out, 0, value_x)
    wp.atomic_max(out, 0, value_y)


@wp.kernel
def _energy_residual_kernel(
    total_energy: wp.array2d(dtype=wp.float32),
    rhoe_prev: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
    nx: int,
    ny: int,
):
    i, j = wp.tid()
    if i >= nx or j >= ny or fluid[i, j] <= 0.5:
        return
    value = wp.abs(total_energy[i, j] - rhoe_prev[i, j])
    if not _is_finite_scalar(value):
        value = 1.0e30
    wp.atomic_max(out, 0, value)


@wp.kernel
def _aero_reduce_kernel(
    density: wp.array2d(dtype=wp.float32),
    momentum_x: wp.array2d(dtype=wp.float32),
    momentum_y: wp.array2d(dtype=wp.float32),
    total_energy: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    x0: float,
    y0: float,
    ref_x: float,
    ref_y: float,
    gamma: float,
    p_floor: float,
):
    i, j = wp.tid()
    if i >= nx or j >= ny or fluid[i, j] <= 0.5:
        return
    p = _pressure_from_conserved(density[i, j], momentum_x[i, j], momentum_y[i, j], total_energy[i, j], gamma, p_floor)
    if not _is_finite_scalar(p):
        return
    solid_east = i + 1 < nx and fluid[i + 1, j] <= 0.5
    solid_west = i > 0 and fluid[i - 1, j] <= 0.5
    solid_north = j + 1 < ny and fluid[i, j + 1] <= 0.5
    solid_south = j > 0 and fluid[i, j - 1] <= 0.5
    if solid_east:
        force_x = p * dy
        force_y = 0.0
        x_face = x0 + float(i + 1) * dx
        y_face = y0 + (float(j) + 0.5) * dy
        wp.atomic_add(out, 0, force_x)
        wp.atomic_add(out, 1, force_y)
        wp.atomic_add(out, 2, (x_face - ref_x) * force_y - (y_face - ref_y) * force_x)
    if solid_west:
        force_x = -p * dy
        force_y = 0.0
        x_face = x0 + float(i) * dx
        y_face = y0 + (float(j) + 0.5) * dy
        wp.atomic_add(out, 0, force_x)
        wp.atomic_add(out, 1, force_y)
        wp.atomic_add(out, 2, (x_face - ref_x) * force_y - (y_face - ref_y) * force_x)
    if solid_north:
        force_x = 0.0
        force_y = p * dx
        x_face = x0 + (float(i) + 0.5) * dx
        y_face = y0 + float(j + 1) * dy
        wp.atomic_add(out, 0, force_x)
        wp.atomic_add(out, 1, force_y)
        wp.atomic_add(out, 2, (x_face - ref_x) * force_y - (y_face - ref_y) * force_x)
    if solid_south:
        force_x = 0.0
        force_y = -p * dx
        x_face = x0 + (float(i) + 0.5) * dx
        y_face = y0 + float(j) * dy
        wp.atomic_add(out, 0, force_x)
        wp.atomic_add(out, 1, force_y)
        wp.atomic_add(out, 2, (x_face - ref_x) * force_y - (y_face - ref_y) * force_x)


def _resolve_device(device: Optional[str]) -> object:
    wp.init()
    requested = device or "cpu"
    getter = getattr(wp, "get_device", None)
    if getter is None:
        return requested
    try:
        return getter(requested)
    except Exception:
        if requested != "cpu":
            return getter("cpu")
        raise


def _fill_buffer(buffer: object, value: float, count: int, device: object) -> None:
    wp.launch(_fill_scalar_buffer_kernel, dim=count, inputs=(value, buffer, count), device=device)


def _copy_field(src: object, dst: object, shape: tuple[int, int], device: object) -> None:
    wp.launch(_copy_field_kernel, dim=shape, inputs=(src, dst, shape[0], shape[1]), device=device)


def _make_scratch(grid: GridSpec, device: object) -> _DensityBasedScratch:
    nx, ny = grid.shape
    return _DensityBasedScratch(
        rho_next=wp.zeros((nx, ny), dtype=wp.float32, device=device),
        rhou_next=wp.zeros((nx, ny), dtype=wp.float32, device=device),
        rhov_next=wp.zeros((nx, ny), dtype=wp.float32, device=device),
        rhoe_next=wp.zeros((nx, ny), dtype=wp.float32, device=device),
        flux_rho_x=wp.zeros((nx + 1, ny), dtype=wp.float32, device=device),
        flux_rhou_x=wp.zeros((nx + 1, ny), dtype=wp.float32, device=device),
        flux_rhov_x=wp.zeros((nx + 1, ny), dtype=wp.float32, device=device),
        flux_rhoe_x=wp.zeros((nx + 1, ny), dtype=wp.float32, device=device),
        flux_rho_y=wp.zeros((nx, ny + 1), dtype=wp.float32, device=device),
        flux_rhou_y=wp.zeros((nx, ny + 1), dtype=wp.float32, device=device),
        flux_rhov_y=wp.zeros((nx, ny + 1), dtype=wp.float32, device=device),
        flux_rhoe_y=wp.zeros((nx, ny + 1), dtype=wp.float32, device=device),
        rho_prev=wp.zeros((nx, ny), dtype=wp.float32, device=device),
        rhou_prev=wp.zeros((nx, ny), dtype=wp.float32, device=device),
        rhov_prev=wp.zeros((nx, ny), dtype=wp.float32, device=device),
        rhoe_prev=wp.zeros((nx, ny), dtype=wp.float32, device=device),
        scalar_buffer=wp.zeros((1,), dtype=wp.float32, device=device),
        pair_buffer=wp.zeros((2,), dtype=wp.float32, device=device),
        triple_buffer=wp.zeros((3,), dtype=wp.float32, device=device),
    )


def initialize_density_based_state_warp(
    grid: GridSpec,
    solid_levelset: np.ndarray,
    *,
    freestream: FreestreamCondition,
    config: DensityBasedSolverConfig = DensityBasedSolverConfig(),
    device: Optional[str] = None,
) -> DensityBasedState:
    solid_sdf = np.asarray(solid_levelset, dtype=np.float64)
    if solid_sdf.shape != grid.shape:
        raise ValueError(f"solid_levelset has shape {solid_sdf.shape}, expected {grid.shape}.")
    h = min(grid.dx, grid.dy)
    fluid_fraction = _smooth_heaviside(solid_sdf, 0.75 * h)
    fluid = fluid_fraction > 0.05
    aperture_x = _face_aperture_x(fluid_fraction, fluid)
    aperture_y = _face_aperture_y(fluid_fraction, fluid)

    u_inf, v_inf, _ = _freestream_velocity(freestream, config)
    rho_inf = freestream.static_pressure / (config.gas_constant * freestream.static_temperature)
    rhou_inf = rho_inf * u_inf
    rhov_inf = rho_inf * v_inf
    rhoe_inf = _energy_from_primitive(rho_inf, u_inf, v_inf, freestream.static_pressure, config.gamma)

    rho = np.full(grid.shape, rho_inf, dtype=np.float32)
    rhou = np.full(grid.shape, rhou_inf, dtype=np.float32)
    rhov = np.full(grid.shape, rhov_inf, dtype=np.float32)
    rhoe = np.full(grid.shape, rhoe_inf, dtype=np.float32)

    resolved_device = _resolve_device(device)
    return DensityBasedState(
        density=wp.array(rho, dtype=wp.float32, device=resolved_device),
        momentum_x=wp.array(rhou, dtype=wp.float32, device=resolved_device),
        momentum_y=wp.array(rhov, dtype=wp.float32, device=resolved_device),
        total_energy=wp.array(rhoe, dtype=wp.float32, device=resolved_device),
        fluid=wp.array(fluid.astype(np.float32), dtype=wp.float32, device=resolved_device),
        fluid_fraction=wp.array(fluid_fraction.astype(np.float32), dtype=wp.float32, device=resolved_device),
        aperture_x=wp.array(aperture_x.astype(np.float32), dtype=wp.float32, device=resolved_device),
        aperture_y=wp.array(aperture_y.astype(np.float32), dtype=wp.float32, device=resolved_device),
        solid_levelset=wp.array(solid_sdf.astype(np.float32), dtype=wp.float32, device=resolved_device),
        dt=0.0,
    )


def run_density_based_solver_warp(
    grid: GridSpec,
    solid_levelset: np.ndarray,
    *,
    chord: float,
    freestream: FreestreamCondition = FreestreamCondition(mach=1.5),
    config: DensityBasedSolverConfig = DensityBasedSolverConfig(),
    device: Optional[str] = None,
) -> tuple[DensityBasedState, list[DensityBasedSnapshot]]:
    if grid.is_3d:
        raise NotImplementedError("The experimental density-based Warp solver currently supports 2D grids only.")

    state = initialize_density_based_state_warp(grid, solid_levelset, freestream=freestream, config=config, device=device)
    device_obj = state.density.device
    scratch = _make_scratch(grid, device_obj)

    u_inf, v_inf, _ = _freestream_velocity(freestream, config)
    rho_inf = freestream.static_pressure / (config.gas_constant * freestream.static_temperature)
    rhou_inf = rho_inf * u_inf
    rhov_inf = rho_inf * v_inf
    rhoe_inf = _energy_from_primitive(rho_inf, u_inf, v_inf, freestream.static_pressure, config.gamma)
    pressure_ref = freestream.static_pressure
    ref_x, ref_y = _reference_point(chord, grid.x0 + 0.25 * grid.extent[0], grid.y0 + 0.5 * grid.extent[1])
    q_inf = max(_dynamic_pressure(freestream, config), 1.0e-6)
    alpha = math.radians(freestream.angle_of_attack_deg)
    drag_dir = (math.cos(alpha), math.sin(alpha))
    lift_dir = (-math.sin(alpha), math.cos(alpha))

    wp.launch(
        _apply_outer_boundaries_kernel,
        dim=grid.shape,
        inputs=(
            state.density,
            state.momentum_x,
            state.momentum_y,
            state.total_energy,
            state.fluid,
            grid.nx,
            grid.ny,
            rho_inf,
            rhou_inf,
            rhov_inf,
            rhoe_inf,
        ),
        device=device_obj,
    )

    history: list[DensityBasedSnapshot] = []
    residual_reference: Optional[tuple[float, float, float]] = None
    previous_forces: Optional[AerodynamicCoefficients] = None
    cut_cell_dt_scale = math.sqrt(max(config.min_fluid_fraction, 1.0e-3))
    velocity_limit = config.max_velocity_factor * max(math.hypot(u_inf, v_inf), math.sqrt(config.gamma * config.gas_constant * freestream.static_temperature))
    velocity_limit_sq = velocity_limit * velocity_limit

    for step in range(1, config.pseudo_steps + 1):
        _copy_field(state.density, scratch.rho_prev, grid.shape, device_obj)
        _copy_field(state.momentum_x, scratch.rhou_prev, grid.shape, device_obj)
        _copy_field(state.momentum_y, scratch.rhov_prev, grid.shape, device_obj)
        _copy_field(state.total_energy, scratch.rhoe_prev, grid.shape, device_obj)

        _fill_buffer(scratch.scalar_buffer, 0.0, 1, device_obj)
        wp.launch(
            _max_wave_speed_kernel,
            dim=grid.shape,
            inputs=(
                state.density,
                state.momentum_x,
                state.momentum_y,
                state.total_energy,
                state.fluid,
                scratch.scalar_buffer,
                grid.nx,
                grid.ny,
                config.gamma,
                config.min_pressure,
            ),
            device=device_obj,
        )
        max_speed = float(scratch.scalar_buffer.numpy()[0])
        h = min(grid.dx, grid.dy)
        state.dt = config.dt_safety * config.cfl * cut_cell_dt_scale * h / max(max_speed, 1.0)

        wp.launch(
            _face_flux_x_kernel,
            dim=(grid.nx + 1, grid.ny),
            inputs=(
                state.density,
                state.momentum_x,
                state.momentum_y,
                state.total_energy,
                state.fluid,
                state.fluid_fraction,
                state.aperture_x,
                scratch.flux_rho_x,
                scratch.flux_rhou_x,
                scratch.flux_rhov_x,
                scratch.flux_rhoe_x,
                grid.nx,
                grid.ny,
                config.gamma,
                config.min_density,
                config.min_pressure,
                rho_inf,
                rhou_inf,
                rhov_inf,
                rhoe_inf,
            ),
            device=device_obj,
        )
        wp.launch(
            _face_flux_y_kernel,
            dim=(grid.nx, grid.ny + 1),
            inputs=(
                state.density,
                state.momentum_x,
                state.momentum_y,
                state.total_energy,
                state.fluid,
                state.fluid_fraction,
                state.aperture_y,
                scratch.flux_rho_y,
                scratch.flux_rhou_y,
                scratch.flux_rhov_y,
                scratch.flux_rhoe_y,
                grid.nx,
                grid.ny,
                config.gamma,
                config.min_density,
                config.min_pressure,
                rho_inf,
                rhou_inf,
                rhov_inf,
                rhoe_inf,
            ),
            device=device_obj,
        )
        wp.launch(
            _update_conserved_kernel,
            dim=grid.shape,
            inputs=(
                state.density,
                state.momentum_x,
                state.momentum_y,
                state.total_energy,
                state.fluid,
                state.fluid_fraction,
                scratch.flux_rho_x,
                scratch.flux_rhou_x,
                scratch.flux_rhov_x,
                scratch.flux_rhoe_x,
                scratch.flux_rho_y,
                scratch.flux_rhou_y,
                scratch.flux_rhov_y,
                scratch.flux_rhoe_y,
                scratch.rho_next,
                scratch.rhou_next,
                scratch.rhov_next,
                scratch.rhoe_next,
                grid.nx,
                grid.ny,
                grid.dx,
                grid.dy,
                state.dt,
                config.gamma,
                config.min_density,
                config.min_pressure,
                config.min_fluid_fraction,
                rho_inf,
                rhou_inf,
                rhov_inf,
                rhoe_inf,
            ),
            device=device_obj,
        )
        wp.launch(
            _swap_next_kernel,
            dim=grid.shape,
            inputs=(
                scratch.rho_next,
                scratch.rhou_next,
                scratch.rhov_next,
                scratch.rhoe_next,
                state.density,
                state.momentum_x,
                state.momentum_y,
                state.total_energy,
                grid.nx,
                grid.ny,
            ),
            device=device_obj,
        )
        wp.launch(
            _apply_outer_boundaries_kernel,
            dim=grid.shape,
            inputs=(
                state.density,
                state.momentum_x,
                state.momentum_y,
                state.total_energy,
                state.fluid,
                grid.nx,
                grid.ny,
                rho_inf,
                rhou_inf,
                rhov_inf,
                rhoe_inf,
            ),
            device=device_obj,
        )
        wp.launch(
            _sanitize_conserved_kernel,
            dim=grid.shape,
            inputs=(
                state.density,
                state.momentum_x,
                state.momentum_y,
                state.total_energy,
                scratch.rho_prev,
                scratch.rhou_prev,
                scratch.rhov_prev,
                scratch.rhoe_prev,
                state.fluid,
                grid.nx,
                grid.ny,
                config.gamma,
                config.min_density,
                config.min_pressure,
                velocity_limit_sq,
                rho_inf,
                rhou_inf,
                rhov_inf,
                rhoe_inf,
            ),
            device=device_obj,
        )

        _fill_buffer(scratch.scalar_buffer, 0.0, 1, device_obj)
        wp.launch(
            _density_residual_kernel,
            dim=grid.shape,
            inputs=(state.density, scratch.rho_prev, state.fluid, scratch.scalar_buffer, grid.nx, grid.ny),
            device=device_obj,
        )
        density_residual = float(scratch.scalar_buffer.numpy()[0])

        _fill_buffer(scratch.scalar_buffer, 0.0, 1, device_obj)
        wp.launch(
            _momentum_residual_kernel,
            dim=grid.shape,
            inputs=(
                state.momentum_x,
                scratch.rhou_prev,
                state.momentum_y,
                scratch.rhov_prev,
                state.fluid,
                scratch.scalar_buffer,
                grid.nx,
                grid.ny,
            ),
            device=device_obj,
        )
        momentum_residual = float(scratch.scalar_buffer.numpy()[0])

        _fill_buffer(scratch.scalar_buffer, 0.0, 1, device_obj)
        wp.launch(
            _energy_residual_kernel,
            dim=grid.shape,
            inputs=(state.total_energy, scratch.rhoe_prev, state.fluid, scratch.scalar_buffer, grid.nx, grid.ny),
            device=device_obj,
        )
        energy_residual = float(scratch.scalar_buffer.numpy()[0])

        _fill_buffer(scratch.triple_buffer, 0.0, 3, device_obj)
        wp.launch(
            _aero_reduce_kernel,
            dim=grid.shape,
            inputs=(
                state.density,
                state.momentum_x,
                state.momentum_y,
                state.total_energy,
                state.fluid,
                scratch.triple_buffer,
                grid.nx,
                grid.ny,
                grid.dx,
                grid.dy,
                grid.x0,
                grid.y0,
                ref_x,
                ref_y,
                config.gamma,
                config.min_pressure,
            ),
            device=device_obj,
        )
        fx, fy, moment = np.asarray(scratch.triple_buffer.numpy(), dtype=np.float64)
        aero = AerodynamicCoefficients(
            drag=(fx * drag_dir[0] + fy * drag_dir[1]) / (q_inf * chord),
            lift=(fx * lift_dir[0] + fy * lift_dir[1]) / (q_inf * chord),
            moment=moment / (q_inf * chord * chord),
        )

        if residual_reference is None:
            residual_reference = (
                max(density_residual, 1.0e-12),
                max(momentum_residual, 1.0e-12),
                max(energy_residual, 1.0e-12),
            )
        if previous_forces is None:
            force_delta = float(max(abs(aero.drag), abs(aero.lift), abs(aero.moment)))
        else:
            force_delta = float(
                max(
                    abs(aero.drag - previous_forces.drag),
                    abs(aero.lift - previous_forces.lift),
                    abs(aero.moment - previous_forces.moment),
                )
            )
        previous_forces = aero

        snapshot = DensityBasedSnapshot(
            step=step,
            dt=state.dt,
            density_residual=density_residual,
            momentum_residual=momentum_residual,
            energy_residual=energy_residual,
            normalized_density=density_residual / residual_reference[0],
            normalized_momentum=momentum_residual / residual_reference[1],
            normalized_energy=energy_residual / residual_reference[2],
            drag_coefficient=aero.drag,
            lift_coefficient=aero.lift,
            moment_coefficient=aero.moment,
            force_delta=force_delta,
        )
        history.append(snapshot)

        recent = history[-config.convergence_window :]
        recent_residual = max(
            max(item.normalized_density, item.normalized_momentum, item.normalized_energy) for item in recent
        )
        recent_force_delta = max(item.force_delta for item in recent)
        if (
            len(recent) >= config.convergence_window
            and recent_residual < config.convergence_tolerance
            and recent_force_delta < config.force_coefficient_tolerance
        ):
            break

    return state, history


__all__ = [
    "DensityBasedSolverConfig",
    "DensityBasedSnapshot",
    "DensityBasedState",
    "initialize_density_based_state_warp",
    "run_density_based_solver_warp",
]
