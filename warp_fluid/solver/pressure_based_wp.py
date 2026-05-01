from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import warp as wp

from ..core.grid import GridSpec
from .pressure_based_np import (
    AerodynamicCoefficients,
    FreestreamCondition,
    PressureBasedSolverConfig,
    PressureBasedState,
    ResidualSnapshot,
    _dynamic_pressure,
    _freestream_velocity,
    _reference_point,
    initialize_pressure_based_state,
)


@dataclass
class PressureBasedWarpState:
    density: object
    u: object
    v: object
    pressure: object
    temperature: object
    turbulent_kinetic_energy: object
    specific_dissipation: object
    turbulent_viscosity: object
    wall_distance: object
    fluid: object
    fluid_fraction: object
    aperture_x: object
    aperture_y: object
    solid_levelset: object
    laminar_viscosity: float
    dt: float = 0.0

    def to_numpy_state(self) -> PressureBasedState:
        return PressureBasedState(
            density=np.asarray(self.density.numpy(), dtype=np.float64),
            u=np.asarray(self.u.numpy(), dtype=np.float64),
            v=np.asarray(self.v.numpy(), dtype=np.float64),
            pressure=np.asarray(self.pressure.numpy(), dtype=np.float64),
            temperature=np.asarray(self.temperature.numpy(), dtype=np.float64),
            turbulent_kinetic_energy=np.asarray(self.turbulent_kinetic_energy.numpy(), dtype=np.float64),
            specific_dissipation=np.asarray(self.specific_dissipation.numpy(), dtype=np.float64),
            turbulent_viscosity=np.asarray(self.turbulent_viscosity.numpy(), dtype=np.float64),
            wall_distance=np.asarray(self.wall_distance.numpy(), dtype=np.float64),
            fluid=np.asarray(self.fluid.numpy(), dtype=np.float32) > 0.5,
            fluid_fraction=np.asarray(self.fluid_fraction.numpy(), dtype=np.float64),
            aperture_x=np.asarray(self.aperture_x.numpy(), dtype=np.float64),
            aperture_y=np.asarray(self.aperture_y.numpy(), dtype=np.float64),
            solid_levelset=np.asarray(self.solid_levelset.numpy(), dtype=np.float64),
            laminar_viscosity=float(self.laminar_viscosity),
            dt=float(self.dt),
        )


@dataclass
class _PressureBasedWarpScratch:
    pressure_extended: object
    density_extended: object
    temperature_extended: object
    scalar_tmp: object
    mu_eff: object
    alpha_eff: object
    gamma_k: object
    gamma_w: object
    mu_t: object
    sigma_k: object
    sigma_w: object
    alpha: object
    beta: object
    f1: object
    strain: object
    temperature_next: object
    u_prev: object
    v_prev: object
    t_prev: object
    k_prev: object
    omega_prev: object
    u_star: object
    v_star: object
    rho_face_x: object
    rho_face_y: object
    mass_flux_x: object
    mass_flux_y: object
    divergence_star: object
    beta_x: object
    beta_y: object
    pcorr: object
    scalar_buffer: object
    pair_buffer: object
    triple_buffer: object


StepCallback = Callable[[int, "PressureBasedWarpState", ResidualSnapshot], None]


@wp.kernel
def _fill_scalar_buffer_kernel(value: float, out: wp.array(dtype=wp.float32), count: int):
    index = wp.tid()
    if index < count:
        out[index] = value


@wp.kernel
def _fill_cell_kernel(value: float, out: wp.array2d(dtype=wp.float32), nx: int, ny: int):
    i, j = wp.tid()
    if i < nx and j < ny:
        out[i, j] = value


@wp.func
def _fluid_active(fluid: wp.array2d(dtype=wp.float32), i: int, j: int) -> float:
    return 1.0 if fluid[i, j] > 0.5 else 0.0


@wp.func
def _sample_clamped(field: wp.array2d(dtype=wp.float32), i: int, j: int, nx: int, ny: int) -> float:
    ii = wp.max(0, wp.min(i, nx - 1))
    jj = wp.max(0, wp.min(j, ny - 1))
    return field[ii, jj]


@wp.func
def _gradient_x(field: wp.array2d(dtype=wp.float32), i: int, j: int, dx: float, nx: int, ny: int) -> float:
    if i <= 0:
        return (_sample_clamped(field, 1, j, nx, ny) - field[0, j]) / dx
    if i >= nx - 1:
        return (field[nx - 1, j] - _sample_clamped(field, nx - 2, j, nx, ny)) / dx
    return (_sample_clamped(field, i + 1, j, nx, ny) - _sample_clamped(field, i - 1, j, nx, ny)) / (2.0 * dx)


@wp.func
def _gradient_y(field: wp.array2d(dtype=wp.float32), i: int, j: int, dy: float, nx: int, ny: int) -> float:
    if j <= 0:
        return (_sample_clamped(field, i, 1, nx, ny) - field[i, 0]) / dy
    if j >= ny - 1:
        return (field[i, ny - 1] - _sample_clamped(field, i, ny - 2, nx, ny)) / dy
    return (_sample_clamped(field, i, j + 1, nx, ny) - _sample_clamped(field, i, j - 1, nx, ny)) / (2.0 * dy)


@wp.func
def _face_average_x(field: wp.array2d(dtype=wp.float32), fi: int, j: int, nx: int, ny: int) -> float:
    if fi <= 0:
        return field[0, j]
    if fi >= nx:
        return field[nx - 1, j]
    return 0.5 * (field[fi - 1, j] + field[fi, j])


@wp.func
def _face_average_y(field: wp.array2d(dtype=wp.float32), i: int, fj: int, nx: int, ny: int) -> float:
    if fj <= 0:
        return field[i, 0]
    if fj >= ny:
        return field[i, ny - 1]
    return 0.5 * (field[i, fj - 1] + field[i, fj])


@wp.func
def _mask_x(fluid: wp.array2d(dtype=wp.float32), fi: int, j: int, nx: int, ny: int) -> float:
    if fi <= 0:
        return _fluid_active(fluid, 0, j)
    if fi >= nx:
        return _fluid_active(fluid, nx - 1, j)
    return 1.0 if fluid[fi - 1, j] > 0.5 and fluid[fi, j] > 0.5 else 0.0


@wp.func
def _mask_y(fluid: wp.array2d(dtype=wp.float32), i: int, fj: int, nx: int, ny: int) -> float:
    if fj <= 0:
        return _fluid_active(fluid, i, 0)
    if fj >= ny:
        return _fluid_active(fluid, i, ny - 1)
    return 1.0 if fluid[i, fj - 1] > 0.5 and fluid[i, fj] > 0.5 else 0.0


@wp.func
def _cell_fraction(fluid_fraction: wp.array2d(dtype=wp.float32), fluid: wp.array2d(dtype=wp.float32), i: int, j: int) -> float:
    if fluid[i, j] <= 0.5:
        return 0.0
    return wp.max(fluid_fraction[i, j], 0.05)


@wp.func
def _second_order_upwind_x(
    field: wp.array2d(dtype=wp.float32),
    face_velocity: float,
    fi: int,
    j: int,
    dx: float,
    nx: int,
    ny: int,
) -> float:
    if fi <= 0:
        return field[0, j]
    if fi >= nx:
        return field[nx - 1, j]
    left = field[fi - 1, j] + 0.5 * dx * _gradient_x(field, fi - 1, j, dx, nx, ny)
    right = field[fi, j] - 0.5 * dx * _gradient_x(field, fi, j, dx, nx, ny)
    return left if face_velocity >= 0.0 else right


@wp.func
def _second_order_upwind_y(
    field: wp.array2d(dtype=wp.float32),
    face_velocity: float,
    i: int,
    fj: int,
    dy: float,
    nx: int,
    ny: int,
) -> float:
    if fj <= 0:
        return field[i, 0]
    if fj >= ny:
        return field[i, ny - 1]
    bottom = field[i, fj - 1] + 0.5 * dy * _gradient_y(field, i, fj - 1, dy, nx, ny)
    top = field[i, fj] - 0.5 * dy * _gradient_y(field, i, fj, dy, nx, ny)
    return bottom if face_velocity >= 0.0 else top


@wp.func
def _convection(
    field: wp.array2d(dtype=wp.float32),
    density: wp.array2d(dtype=wp.float32),
    u: wp.array2d(dtype=wp.float32),
    v: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    fluid_fraction: wp.array2d(dtype=wp.float32),
    aperture_x: wp.array2d(dtype=wp.float32),
    aperture_y: wp.array2d(dtype=wp.float32),
    i: int,
    j: int,
    dx: float,
    dy: float,
    nx: int,
    ny: int,
) -> float:
    if fluid[i, j] <= 0.5:
        return 0.0
    volfrac = _cell_fraction(fluid_fraction, fluid, i, j)
    ap_l = aperture_x[i, j]
    ap_r = aperture_x[i + 1, j]
    ap_b = aperture_y[i, j]
    ap_t = aperture_y[i, j + 1]

    u_l = _face_average_x(u, i, j, nx, ny)
    u_r = _face_average_x(u, i + 1, j, nx, ny)
    v_b = _face_average_y(v, i, j, nx, ny)
    v_t = _face_average_y(v, i, j + 1, nx, ny)

    rho_l = _face_average_x(density, i, j, nx, ny)
    rho_r = _face_average_x(density, i + 1, j, nx, ny)
    rho_b = _face_average_y(density, i, j, nx, ny)
    rho_t = _face_average_y(density, i, j + 1, nx, ny)

    phi_l = _second_order_upwind_x(field, u_l, i, j, dx, nx, ny)
    phi_r = _second_order_upwind_x(field, u_r, i + 1, j, dx, nx, ny)
    phi_b = _second_order_upwind_y(field, v_b, i, j, dy, nx, ny)
    phi_t = _second_order_upwind_y(field, v_t, i, j + 1, dy, nx, ny)

    flux_l = ap_l * rho_l * u_l * phi_l
    flux_r = ap_r * rho_r * u_r * phi_r
    flux_b = ap_b * rho_b * v_b * phi_b
    flux_t = ap_t * rho_t * v_t * phi_t
    return ((flux_r - flux_l) / dx + (flux_t - flux_b) / dy) / wp.max(volfrac, 1.0e-12)


@wp.func
def _diffusion(
    field: wp.array2d(dtype=wp.float32),
    diffusivity: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    fluid_fraction: wp.array2d(dtype=wp.float32),
    aperture_x: wp.array2d(dtype=wp.float32),
    aperture_y: wp.array2d(dtype=wp.float32),
    i: int,
    j: int,
    dx: float,
    dy: float,
    nx: int,
    ny: int,
) -> float:
    if fluid[i, j] <= 0.5:
        return 0.0
    volfrac = _cell_fraction(fluid_fraction, fluid, i, j)
    ap_l = aperture_x[i, j]
    ap_r = aperture_x[i + 1, j]
    ap_b = aperture_y[i, j]
    ap_t = aperture_y[i, j + 1]

    gamma_l = _face_average_x(diffusivity, i, j, nx, ny)
    gamma_r = _face_average_x(diffusivity, i + 1, j, nx, ny)
    gamma_b = _face_average_y(diffusivity, i, j, nx, ny)
    gamma_t = _face_average_y(diffusivity, i, j + 1, nx, ny)

    grad_l = 0.0 if i <= 0 else (field[i, j] - field[i - 1, j]) / dx
    grad_r = 0.0 if i >= nx - 1 else (field[i + 1, j] - field[i, j]) / dx
    grad_b = 0.0 if j <= 0 else (field[i, j] - field[i, j - 1]) / dy
    grad_t = 0.0 if j >= ny - 1 else (field[i, j + 1] - field[i, j]) / dy

    flux_l = ap_l * gamma_l * grad_l
    flux_r = ap_r * gamma_r * grad_r
    flux_b = ap_b * gamma_b * grad_b
    flux_t = ap_t * gamma_t * grad_t
    return ((flux_r - flux_l) / dx + (flux_t - flux_b) / dy) / wp.max(volfrac, 1.0e-12)


@wp.func
def _diffusion_zero_wall(
    field: wp.array2d(dtype=wp.float32),
    diffusivity: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    fluid_fraction: wp.array2d(dtype=wp.float32),
    aperture_x: wp.array2d(dtype=wp.float32),
    aperture_y: wp.array2d(dtype=wp.float32),
    i: int,
    j: int,
    dx: float,
    dy: float,
    nx: int,
    ny: int,
) -> float:
    if fluid[i, j] <= 0.5:
        return 0.0

    volfrac = _cell_fraction(fluid_fraction, fluid, i, j)
    c = field[i, j]

    if i > 0 and fluid[i - 1, j] > 0.5:
        gamma_l = _face_average_x(diffusivity, i, j, nx, ny)
        grad_l_open = (c - field[i - 1, j]) / dx
        grad_l_wall = (c - 0.0) / (0.5 * dx)
    elif i > 0:
        gamma_l = diffusivity[i, j]
        grad_l_open = 0.0
        grad_l_wall = (c - 0.0) / (0.5 * dx)
    else:
        gamma_l = diffusivity[i, j]
        grad_l_open = 0.0
        grad_l_wall = 0.0

    if i + 1 < nx and fluid[i + 1, j] > 0.5:
        gamma_r = _face_average_x(diffusivity, i + 1, j, nx, ny)
        grad_r_open = (field[i + 1, j] - c) / dx
        grad_r_wall = (0.0 - c) / (0.5 * dx)
    elif i + 1 < nx:
        gamma_r = diffusivity[i, j]
        grad_r_open = 0.0
        grad_r_wall = (0.0 - c) / (0.5 * dx)
    else:
        gamma_r = diffusivity[i, j]
        grad_r_open = 0.0
        grad_r_wall = 0.0

    if j > 0 and fluid[i, j - 1] > 0.5:
        gamma_b = _face_average_y(diffusivity, i, j, nx, ny)
        grad_b_open = (c - field[i, j - 1]) / dy
        grad_b_wall = (c - 0.0) / (0.5 * dy)
    elif j > 0:
        gamma_b = diffusivity[i, j]
        grad_b_open = 0.0
        grad_b_wall = (c - 0.0) / (0.5 * dy)
    else:
        gamma_b = diffusivity[i, j]
        grad_b_open = 0.0
        grad_b_wall = 0.0

    if j + 1 < ny and fluid[i, j + 1] > 0.5:
        gamma_t = _face_average_y(diffusivity, i, j + 1, nx, ny)
        grad_t_open = (field[i, j + 1] - c) / dy
        grad_t_wall = (0.0 - c) / (0.5 * dy)
    elif j + 1 < ny:
        gamma_t = diffusivity[i, j]
        grad_t_open = 0.0
        grad_t_wall = (0.0 - c) / (0.5 * dy)
    else:
        gamma_t = diffusivity[i, j]
        grad_t_open = 0.0
        grad_t_wall = 0.0

    ap_l = aperture_x[i, j]
    ap_r = aperture_x[i + 1, j]
    ap_b = aperture_y[i, j]
    ap_t = aperture_y[i, j + 1]
    wall_l = 1.0 - ap_l
    wall_r = 1.0 - ap_r
    wall_b = 1.0 - ap_b
    wall_t = 1.0 - ap_t

    flux_l = ap_l * gamma_l * grad_l_open + wall_l * gamma_l * grad_l_wall
    flux_r = ap_r * gamma_r * grad_r_open + wall_r * gamma_r * grad_r_wall
    flux_b = ap_b * gamma_b * grad_b_open + wall_b * gamma_b * grad_b_wall
    flux_t = ap_t * gamma_t * grad_t_open + wall_t * gamma_t * grad_t_wall
    return ((flux_r - flux_l) / dx + (flux_t - flux_b) / dy) / wp.max(volfrac, 1.0e-12)


@wp.func
def _velocity_divergence(
    u: wp.array2d(dtype=wp.float32),
    v: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    fluid_fraction: wp.array2d(dtype=wp.float32),
    aperture_x: wp.array2d(dtype=wp.float32),
    aperture_y: wp.array2d(dtype=wp.float32),
    i: int,
    j: int,
    dx: float,
    dy: float,
    nx: int,
    ny: int,
) -> float:
    if fluid[i, j] <= 0.5:
        return 0.0
    volfrac = _cell_fraction(fluid_fraction, fluid, i, j)
    u_l = aperture_x[i, j] * _face_average_x(u, i, j, nx, ny)
    u_r = aperture_x[i + 1, j] * _face_average_x(u, i + 1, j, nx, ny)
    v_b = aperture_y[i, j] * _face_average_y(v, i, j, nx, ny)
    v_t = aperture_y[i, j + 1] * _face_average_y(v, i, j + 1, nx, ny)
    return ((u_r - u_l) / dx + (v_t - v_b) / dy) / wp.max(volfrac, 1.0e-12)


@wp.kernel
def _apply_outer_boundaries_kernel(
    density: wp.array2d(dtype=wp.float32),
    u: wp.array2d(dtype=wp.float32),
    v: wp.array2d(dtype=wp.float32),
    pressure: wp.array2d(dtype=wp.float32),
    temperature: wp.array2d(dtype=wp.float32),
    k: wp.array2d(dtype=wp.float32),
    omega: wp.array2d(dtype=wp.float32),
    mu_t: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    rho_inf: float,
    u_inf: float,
    v_inf: float,
    pressure_inf: float,
    pressure_out: float,
    temperature_inf: float,
    k_inf: float,
    omega_inf: float,
):
    i, j = wp.tid()
    if i >= nx or j >= ny:
        return
    if j == 0 or j == ny - 1:
        pressure[i, j] = pressure_inf
        temperature[i, j] = temperature_inf
        density[i, j] = rho_inf
        u[i, j] = u_inf
        v[i, j] = v_inf
        k[i, j] = k_inf
        omega[i, j] = omega_inf
        mu_t[i, j] = 0.0
        return
    if i == 0:
        pressure[i, j] = pressure_inf
        temperature[i, j] = temperature_inf
        density[i, j] = rho_inf
        u[i, j] = u_inf
        v[i, j] = v_inf
        k[i, j] = k_inf
        omega[i, j] = omega_inf
        mu_t[i, j] = 0.0
        return
    if i == nx - 1:
        pressure[i, j] = pressure_out
        temperature[i, j] = temperature[i - 1, j]
        density[i, j] = density[i - 1, j]
        u[i, j] = u[i - 1, j]
        v[i, j] = v[i - 1, j]
        k[i, j] = k[i - 1, j]
        omega[i, j] = omega[i - 1, j]
        mu_t[i, j] = mu_t[i - 1, j]


@wp.kernel
def _solid_enforce_kernel(
    density: wp.array2d(dtype=wp.float32),
    u: wp.array2d(dtype=wp.float32),
    v: wp.array2d(dtype=wp.float32),
    temperature: wp.array2d(dtype=wp.float32),
    k: wp.array2d(dtype=wp.float32),
    omega: wp.array2d(dtype=wp.float32),
    mu_t: wp.array2d(dtype=wp.float32),
    wall_distance: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    laminar_viscosity: float,
    wall_floor: float,
    min_density: float,
    min_k: float,
    omega_inf: float,
    wall_temperature: float,
    use_wall_temperature: float,
):
    i, j = wp.tid()
    if i >= nx or j >= ny:
        return
    if fluid[i, j] > 0.5:
        return
    rho = wp.max(density[i, j], min_density)
    wall = wp.max(wall_distance[i, j], wall_floor)
    wall_omega = 60.0 * laminar_viscosity / wp.max(rho * wall * wall, 1.0e-12)
    u[i, j] = 0.0
    v[i, j] = 0.0
    k[i, j] = min_k
    omega[i, j] = wp.max(wall_omega, omega_inf)
    mu_t[i, j] = 0.0
    if use_wall_temperature > 0.5:
        temperature[i, j] = wall_temperature


@wp.kernel
def _near_wall_velocity_damping_kernel(
    u: wp.array2d(dtype=wp.float32),
    v: wp.array2d(dtype=wp.float32),
    wall_distance: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    fluid_fraction: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    damping_band: float,
):
    i, j = wp.tid()
    if i >= nx or j >= ny or fluid[i, j] <= 0.5:
        return
    distance = wall_distance[i, j]
    if distance >= damping_band:
        return
    ratio = wp.max(distance / wp.max(damping_band, 1.0e-12), 0.0)
    # Strongly damp the first one-cell band so the immersed wall behaves closer to no-slip.
    fraction = wp.min(wp.max(fluid_fraction[i, j], 0.0), 1.0)
    factor = ratio * ratio * fraction * fraction
    u[i, j] = u[i, j] * factor
    v[i, j] = v[i, j] * factor


@wp.kernel
def _extend_from_fluid_kernel(
    src: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    dst: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
):
    i, j = wp.tid()
    if i >= nx or j >= ny:
        return
    if fluid[i, j] > 0.5:
        dst[i, j] = src[i, j]
        return
    accum = 0.0
    count = 0.0
    if i > 0 and fluid[i - 1, j] > 0.5:
        accum += src[i - 1, j]
        count += 1.0
    if i + 1 < nx and fluid[i + 1, j] > 0.5:
        accum += src[i + 1, j]
        count += 1.0
    if j > 0 and fluid[i, j - 1] > 0.5:
        accum += src[i, j - 1]
        count += 1.0
    if j + 1 < ny and fluid[i, j + 1] > 0.5:
        accum += src[i, j + 1]
        count += 1.0
    dst[i, j] = accum / count if count > 0.0 else src[i, j]


@wp.kernel
def _clip_state_kernel(
    density: wp.array2d(dtype=wp.float32),
    pressure: wp.array2d(dtype=wp.float32),
    temperature: wp.array2d(dtype=wp.float32),
    k: wp.array2d(dtype=wp.float32),
    omega: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    min_density: float,
    min_pressure: float,
    min_temperature: float,
    min_k: float,
    min_omega: float,
):
    i, j = wp.tid()
    if i < nx and j < ny:
        density[i, j] = wp.max(density[i, j], min_density)
        pressure[i, j] = wp.max(pressure[i, j], min_pressure)
        temperature[i, j] = wp.max(temperature[i, j], min_temperature)
        k[i, j] = wp.max(k[i, j], min_k)
        omega[i, j] = wp.max(omega[i, j], min_omega)


@wp.kernel
def _wave_speed_max_kernel(
    u: wp.array2d(dtype=wp.float32),
    v: wp.array2d(dtype=wp.float32),
    temperature: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
    nx: int,
    ny: int,
    gamma: float,
    gas_constant: float,
):
    i, j = wp.tid()
    if i >= nx or j >= ny or fluid[i, j] <= 0.5:
        return
    sound = wp.sqrt(wp.max(gamma * gas_constant * temperature[i, j], 1.0e-12))
    speed = wp.abs(u[i, j]) + wp.abs(v[i, j]) + sound
    if not wp.isnan(speed) and not wp.isinf(speed):
        wp.atomic_max(out, 0, speed)


@wp.kernel
def _effective_diffusivity_max_kernel(
    density: wp.array2d(dtype=wp.float32),
    mu_t: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
    nx: int,
    ny: int,
    laminar_viscosity: float,
    min_density: float,
):
    i, j = wp.tid()
    if i >= nx or j >= ny or fluid[i, j] <= 0.5:
        return
    rho = wp.max(density[i, j], min_density)
    nu_eff = (laminar_viscosity + mu_t[i, j]) / rho
    if not wp.isnan(nu_eff) and not wp.isinf(nu_eff):
        wp.atomic_max(out, 0, nu_eff)


@wp.kernel
def _omega_max_kernel(
    omega: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
    nx: int,
    ny: int,
):
    i, j = wp.tid()
    if i >= nx or j >= ny or fluid[i, j] <= 0.5:
        return
    value = wp.abs(omega[i, j])
    if not wp.isnan(value) and not wp.isinf(value):
        wp.atomic_max(out, 0, value)


@wp.kernel
def _compute_alpha_eff_kernel(
    density: wp.array2d(dtype=wp.float32),
    mu_t: wp.array2d(dtype=wp.float32),
    alpha_eff: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    laminar_viscosity: float,
    min_density: float,
    prandtl: float,
    turbulent_prandtl: float,
):
    i, j = wp.tid()
    if i >= nx or j >= ny:
        return
    rho = wp.max(density[i, j], min_density)
    alpha_eff[i, j] = laminar_viscosity / (rho * prandtl) + mu_t[i, j] / (rho * turbulent_prandtl)


@wp.kernel
def _temperature_update_kernel(
    density: wp.array2d(dtype=wp.float32),
    u: wp.array2d(dtype=wp.float32),
    v: wp.array2d(dtype=wp.float32),
    temperature: wp.array2d(dtype=wp.float32),
    alpha_eff: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    fluid_fraction: wp.array2d(dtype=wp.float32),
    aperture_x: wp.array2d(dtype=wp.float32),
    aperture_y: wp.array2d(dtype=wp.float32),
    out: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    dt: float,
    min_density: float,
    min_temperature: float,
    temperature_relaxation: float,
    gamma: float,
    gas_constant: float,
):
    i, j = wp.tid()
    if i >= nx or j >= ny:
        return
    if fluid[i, j] <= 0.5:
        out[i, j] = temperature[i, j]
        return
    cp = gamma * gas_constant / (gamma - 1.0)
    rho = wp.max(density[i, j], min_density)
    convection = _convection(temperature, density, u, v, fluid, fluid_fraction, aperture_x, aperture_y, i, j, dx, dy, nx, ny)
    diffusion = _diffusion(temperature, alpha_eff, fluid, fluid_fraction, aperture_x, aperture_y, i, j, dx, dy, nx, ny)
    divergence = _velocity_divergence(u, v, fluid, fluid_fraction, aperture_x, aperture_y, i, j, dx, dy, nx, ny)
    rhs = -convection / rho + diffusion - (gamma - 1.0) * temperature[i, j] * divergence
    updated = temperature[i, j] + dt * rhs / cp
    relaxed = (1.0 - temperature_relaxation) * temperature[i, j] + temperature_relaxation * updated
    out[i, j] = wp.max(relaxed, min_temperature)


@wp.kernel
def _recompute_density_kernel(
    density: wp.array2d(dtype=wp.float32),
    pressure: wp.array2d(dtype=wp.float32),
    temperature: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    gas_constant: float,
    min_density: float,
):
    i, j = wp.tid()
    if i < nx and j < ny and fluid[i, j] > 0.5:
        density[i, j] = wp.max(pressure[i, j] / wp.max(gas_constant * temperature[i, j], 1.0e-12), min_density)


@wp.kernel
def _mu_eff_kernel(mu_t: wp.array2d(dtype=wp.float32), out: wp.array2d(dtype=wp.float32), nx: int, ny: int, laminar: float):
    i, j = wp.tid()
    if i < nx and j < ny:
        out[i, j] = laminar + mu_t[i, j]


@wp.kernel
def _limit_mu_t_kernel(
    mu_t: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    laminar: float,
    max_ratio: float,
):
    i, j = wp.tid()
    if i >= nx or j >= ny:
        return
    if fluid[i, j] <= 0.5:
        mu_t[i, j] = 0.0
        return
    upper = max_ratio * laminar
    value = mu_t[i, j]
    if wp.isnan(value) or wp.isinf(value):
        value = 0.0
    mu_t[i, j] = wp.min(wp.max(value, 0.0), upper)


@wp.kernel
def _compute_turbulence_aux_kernel(
    density: wp.array2d(dtype=wp.float32),
    k: wp.array2d(dtype=wp.float32),
    omega: wp.array2d(dtype=wp.float32),
    u: wp.array2d(dtype=wp.float32),
    v: wp.array2d(dtype=wp.float32),
    wall_distance: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    mu_t: wp.array2d(dtype=wp.float32),
    sigma_k: wp.array2d(dtype=wp.float32),
    sigma_w: wp.array2d(dtype=wp.float32),
    alpha: wp.array2d(dtype=wp.float32),
    beta: wp.array2d(dtype=wp.float32),
    f1: wp.array2d(dtype=wp.float32),
    strain: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    laminar_viscosity: float,
    wall_floor: float,
):
    i, j = wp.tid()
    if i >= nx or j >= ny:
        return
    if fluid[i, j] <= 0.5:
        mu_t[i, j] = 0.0
        sigma_k[i, j] = 0.85
        sigma_w[i, j] = 0.5
        alpha[i, j] = 5.0 / 9.0
        beta[i, j] = 0.075
        f1[i, j] = 1.0
        strain[i, j] = 0.0
        return
    beta_star = 0.09
    sigma_w2 = 0.856
    rho = wp.max(density[i, j], 1.0e-12)
    omega_safe = wp.max(omega[i, j], 1.0e-12)
    k_safe = wp.max(k[i, j], 1.0e-12)
    wall = wp.max(wall_distance[i, j], wall_floor)
    nu = laminar_viscosity / rho
    grad_kx = _gradient_x(k, i, j, dx, nx, ny)
    grad_ky = _gradient_y(k, i, j, dy, nx, ny)
    grad_wx = _gradient_x(omega, i, j, dx, nx, ny)
    grad_wy = _gradient_y(omega, i, j, dy, nx, ny)
    grad_dot = grad_kx * grad_wx + grad_ky * grad_wy
    cd_kw = wp.max(2.0 * rho * sigma_w2 * grad_dot / omega_safe, 1.0e-20)
    term_a = wp.sqrt(k_safe) / wp.max(beta_star * omega_safe * wall, 1.0e-12)
    term_b = 500.0 * nu / wp.max(wall * wall * omega_safe, 1.0e-12)
    term_c = 4.0 * rho * sigma_w2 * k_safe / wp.max(cd_kw * wall * wall, 1.0e-12)
    arg1 = wp.min(wp.max(term_a, term_b), term_c)
    f1_value = wp.tanh(arg1 * arg1 * arg1 * arg1)
    arg2 = wp.max(2.0 * wp.sqrt(k_safe) / wp.max(beta_star * omega_safe * wall, 1.0e-12), term_b)
    f2 = wp.tanh(arg2 * arg2)
    sigma_k_value = f1_value * 0.85 + (1.0 - f1_value) * 1.0
    sigma_w_value = f1_value * 0.5 + (1.0 - f1_value) * 0.856
    alpha_value = f1_value * (5.0 / 9.0) + (1.0 - f1_value) * 0.44
    beta_value = f1_value * 0.075 + (1.0 - f1_value) * 0.0828
    du_dx = _gradient_x(u, i, j, dx, nx, ny)
    du_dy = _gradient_y(u, i, j, dy, nx, ny)
    dv_dx = _gradient_x(v, i, j, dx, nx, ny)
    dv_dy = _gradient_y(v, i, j, dy, nx, ny)
    strain_value = wp.sqrt(
        wp.max(2.0 * du_dx * du_dx + 2.0 * dv_dy * dv_dy + (du_dy + dv_dx) * (du_dy + dv_dx), 0.0)
    )
    a1 = 0.31
    limiter = wp.max(a1 * omega_safe, strain_value * f2)
    mu_t_value = rho * a1 * k_safe / wp.max(limiter, 1.0e-12)
    mu_t[i, j] = wp.max(mu_t_value, 0.0)
    sigma_k[i, j] = sigma_k_value
    sigma_w[i, j] = sigma_w_value
    alpha[i, j] = alpha_value
    beta[i, j] = beta_value
    f1[i, j] = f1_value
    strain[i, j] = strain_value


@wp.kernel
def _gamma_from_sigma_kernel(
    sigma: wp.array2d(dtype=wp.float32),
    mu_t: wp.array2d(dtype=wp.float32),
    out: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    laminar: float,
):
    i, j = wp.tid()
    if i < nx and j < ny:
        out[i, j] = laminar + sigma[i, j] * mu_t[i, j]


@wp.kernel
def _simple_predict_kernel(
    density: wp.array2d(dtype=wp.float32),
    u: wp.array2d(dtype=wp.float32),
    v: wp.array2d(dtype=wp.float32),
    pressure_extended: wp.array2d(dtype=wp.float32),
    mu_eff: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    fluid_fraction: wp.array2d(dtype=wp.float32),
    aperture_x: wp.array2d(dtype=wp.float32),
    aperture_y: wp.array2d(dtype=wp.float32),
    u_star: wp.array2d(dtype=wp.float32),
    v_star: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    dt: float,
    velocity_relaxation: float,
    min_density: float,
):
    i, j = wp.tid()
    if i >= nx or j >= ny:
        return
    if fluid[i, j] <= 0.5:
        u_star[i, j] = 0.0
        v_star[i, j] = 0.0
        return
    rho = wp.max(density[i, j], min_density)
    dpdx = _gradient_x(pressure_extended, i, j, dx, nx, ny)
    dpdy = _gradient_y(pressure_extended, i, j, dy, nx, ny)
    conv_u = _convection(u, density, u, v, fluid, fluid_fraction, aperture_x, aperture_y, i, j, dx, dy, nx, ny)
    conv_v = _convection(v, density, u, v, fluid, fluid_fraction, aperture_x, aperture_y, i, j, dx, dy, nx, ny)
    diff_u = _diffusion_zero_wall(u, mu_eff, fluid, fluid_fraction, aperture_x, aperture_y, i, j, dx, dy, nx, ny)
    diff_v = _diffusion_zero_wall(v, mu_eff, fluid, fluid_fraction, aperture_x, aperture_y, i, j, dx, dy, nx, ny)
    rhs_u = (-conv_u + diff_u - dpdx) / rho
    rhs_v = (-conv_v + diff_v - dpdy) / rho
    u_star[i, j] = u[i, j] + velocity_relaxation * dt * rhs_u
    v_star[i, j] = v[i, j] + velocity_relaxation * dt * rhs_v


@wp.kernel
def _mass_flux_x_kernel(
    density: wp.array2d(dtype=wp.float32),
    u: wp.array2d(dtype=wp.float32),
    aperture_x: wp.array2d(dtype=wp.float32),
    rho_face_x: wp.array2d(dtype=wp.float32),
    mass_flux_x: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    dy: float,
):
    i, j = wp.tid()
    if i > nx or j >= ny:
        return
    aperture = aperture_x[i, j]
    rho_face = _face_average_x(density, i, j, nx, ny)
    u_face = _face_average_x(u, i, j, nx, ny)
    rho_face_x[i, j] = rho_face
    mass_flux_x[i, j] = aperture * rho_face * u_face * dy


@wp.kernel
def _mass_flux_y_kernel(
    density: wp.array2d(dtype=wp.float32),
    v: wp.array2d(dtype=wp.float32),
    aperture_y: wp.array2d(dtype=wp.float32),
    rho_face_y: wp.array2d(dtype=wp.float32),
    mass_flux_y: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    dx: float,
):
    i, j = wp.tid()
    if i >= nx or j > ny:
        return
    aperture = aperture_y[i, j]
    rho_face = _face_average_y(density, i, j, nx, ny)
    v_face = _face_average_y(v, i, j, nx, ny)
    rho_face_y[i, j] = rho_face
    mass_flux_y[i, j] = aperture * rho_face * v_face * dx


@wp.kernel
def _mass_divergence_kernel(
    mass_flux_x: wp.array2d(dtype=wp.float32),
    mass_flux_y: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    fluid_fraction: wp.array2d(dtype=wp.float32),
    divergence: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    cell_volume: float,
):
    i, j = wp.tid()
    if i >= nx or j >= ny:
        return
    if fluid[i, j] <= 0.5:
        divergence[i, j] = 0.0
        return
    volume = wp.max(_cell_fraction(fluid_fraction, fluid, i, j) * cell_volume, 1.0e-12)
    divergence[i, j] = ((mass_flux_x[i + 1, j] - mass_flux_x[i, j]) + (mass_flux_y[i, j + 1] - mass_flux_y[i, j])) / volume


@wp.kernel
def _beta_x_kernel(
    rho_face_x: wp.array2d(dtype=wp.float32),
    aperture_x: wp.array2d(dtype=wp.float32),
    beta_x: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    dt: float,
    dx: float,
):
    i, j = wp.tid()
    if i <= nx and j < ny:
        beta_x[i, j] = dt * aperture_x[i, j] / wp.max(rho_face_x[i, j] * dx * dx, 1.0e-12)


@wp.kernel
def _beta_y_kernel(
    rho_face_y: wp.array2d(dtype=wp.float32),
    aperture_y: wp.array2d(dtype=wp.float32),
    beta_y: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    dt: float,
    dy: float,
):
    i, j = wp.tid()
    if i < nx and j <= ny:
        beta_y[i, j] = dt * aperture_y[i, j] / wp.max(rho_face_y[i, j] * dy * dy, 1.0e-12)


@wp.kernel
def _rbgs_pressure_kernel(
    pcorr: wp.array2d(dtype=wp.float32),
    divergence_star: wp.array2d(dtype=wp.float32),
    beta_x: wp.array2d(dtype=wp.float32),
    beta_y: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    fluid_fraction: wp.array2d(dtype=wp.float32),
    max_change: wp.array(dtype=wp.float32),
    nx: int,
    ny: int,
    color: int,
    sor_omega: float,
):
    i, j = wp.tid()
    if i <= 0 or i >= nx - 1 or j <= 0 or j >= ny - 1:
        return
    parity = (i + j) & 1
    target_parity = 1 - color
    if parity != target_parity:
        return
    if fluid[i, j] <= 0.5:
        pcorr[i, j] = 0.0
        return
    volfrac = _cell_fraction(fluid_fraction, fluid, i, j)
    ae = beta_x[i + 1, j] / volfrac if fluid[i + 1, j] > 0.5 else 0.0
    aw = beta_x[i, j] / volfrac if fluid[i - 1, j] > 0.5 else 0.0
    an = beta_y[i, j + 1] / volfrac if fluid[i, j + 1] > 0.5 else 0.0
    ass = beta_y[i, j] / volfrac if fluid[i, j - 1] > 0.5 else 0.0
    ap = ae + aw + an + ass
    if ap <= 1.0e-12:
        pcorr[i, j] = 0.0
        return
    rhs = divergence_star[i, j]
    target = (ae * pcorr[i + 1, j] + aw * pcorr[i - 1, j] + an * pcorr[i, j + 1] + ass * pcorr[i, j - 1] + rhs) / ap
    old = pcorr[i, j]
    updated = (1.0 - sor_omega) * old + sor_omega * target
    pcorr[i, j] = updated
    delta = wp.abs(updated - old)
    if not wp.isnan(delta) and not wp.isinf(delta):
        wp.atomic_max(max_change, 0, delta)


@wp.kernel
def _pressure_residual_kernel(
    pcorr: wp.array2d(dtype=wp.float32),
    divergence_star: wp.array2d(dtype=wp.float32),
    beta_x: wp.array2d(dtype=wp.float32),
    beta_y: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    fluid_fraction: wp.array2d(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
    nx: int,
    ny: int,
):
    i, j = wp.tid()
    if i >= nx or j >= ny or fluid[i, j] <= 0.5:
        return
    volfrac = _cell_fraction(fluid_fraction, fluid, i, j)
    ae = beta_x[i + 1, j] / volfrac if i + 1 < nx and fluid[i + 1, j] > 0.5 else 0.0
    aw = beta_x[i, j] / volfrac if i > 0 and fluid[i - 1, j] > 0.5 else 0.0
    an = beta_y[i, j + 1] / volfrac if j + 1 < ny and fluid[i, j + 1] > 0.5 else 0.0
    ass = beta_y[i, j] / volfrac if j > 0 and fluid[i, j - 1] > 0.5 else 0.0
    operator = ae * (pcorr[i + 1, j] - pcorr[i, j]) if i + 1 < nx else 0.0
    operator -= aw * (pcorr[i, j] - pcorr[i - 1, j]) if i > 0 else 0.0
    operator += an * (pcorr[i, j + 1] - pcorr[i, j]) if j + 1 < ny else 0.0
    operator -= ass * (pcorr[i, j] - pcorr[i, j - 1]) if j > 0 else 0.0
    value = wp.abs(divergence_star[i, j] - operator)
    if not wp.isnan(value) and not wp.isinf(value):
        wp.atomic_max(out, 0, value)


@wp.kernel
def _correct_mass_flux_x_kernel(
    mass_flux_x: wp.array2d(dtype=wp.float32),
    rho_face_x: wp.array2d(dtype=wp.float32),
    aperture_x: wp.array2d(dtype=wp.float32),
    pcorr: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    dt: float,
    dx: float,
    dy: float,
):
    i, j = wp.tid()
    if i <= 0 or i >= nx or j >= ny:
        return
    if fluid[i - 1, j] <= 0.5 or fluid[i, j] <= 0.5:
        mass_flux_x[i, j] = 0.0
        return
    rho = wp.max(rho_face_x[i, j], 1.0e-12)
    mass_flux_x[i, j] = mass_flux_x[i, j] - aperture_x[i, j] * dt * dy * (pcorr[i, j] - pcorr[i - 1, j]) / rho / dx


@wp.kernel
def _correct_mass_flux_y_kernel(
    mass_flux_y: wp.array2d(dtype=wp.float32),
    rho_face_y: wp.array2d(dtype=wp.float32),
    aperture_y: wp.array2d(dtype=wp.float32),
    pcorr: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    dt: float,
    dx: float,
    dy: float,
):
    i, j = wp.tid()
    if i >= nx or j <= 0 or j >= ny:
        return
    if fluid[i, j - 1] <= 0.5 or fluid[i, j] <= 0.5:
        mass_flux_y[i, j] = 0.0
        return
    rho = wp.max(rho_face_y[i, j], 1.0e-12)
    mass_flux_y[i, j] = mass_flux_y[i, j] - aperture_y[i, j] * dt * dx * (pcorr[i, j] - pcorr[i, j - 1]) / rho / dy


@wp.kernel
def _limit_pcorr_kernel(
    pcorr: wp.array2d(dtype=wp.float32),
    pressure: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    pressure_fraction: float,
    pressure_scale: float,
):
    i, j = wp.tid()
    if i >= nx or j >= ny or fluid[i, j] <= 0.5:
        return
    limit = wp.max(pressure_fraction * wp.abs(pressure[i, j]), pressure_scale)
    value = pcorr[i, j]
    if wp.isnan(value) or wp.isinf(value):
        value = 0.0
    pcorr[i, j] = wp.min(wp.max(value, -limit), limit)


@wp.kernel
def _cell_velocity_from_fluxes_kernel(
    mass_flux_x: wp.array2d(dtype=wp.float32),
    mass_flux_y: wp.array2d(dtype=wp.float32),
    rho_face_x: wp.array2d(dtype=wp.float32),
    rho_face_y: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    aperture_x: wp.array2d(dtype=wp.float32),
    aperture_y: wp.array2d(dtype=wp.float32),
    u: wp.array2d(dtype=wp.float32),
    v: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    dx: float,
    dy: float,
):
    i, j = wp.tid()
    if i >= nx or j >= ny:
        return
    if fluid[i, j] <= 0.5:
        u[i, j] = 0.0
        v[i, j] = 0.0
        return
    ap_l = aperture_x[i, j]
    ap_r = aperture_x[i + 1, j]
    ap_b = aperture_y[i, j]
    ap_t = aperture_y[i, j + 1]
    u_face_l = mass_flux_x[i, j] / wp.max(rho_face_x[i, j] * dy * ap_l, 1.0e-12) if ap_l > 1.0e-6 else 0.0
    u_face_r = mass_flux_x[i + 1, j] / wp.max(rho_face_x[i + 1, j] * dy * ap_r, 1.0e-12) if ap_r > 1.0e-6 else 0.0
    v_face_b = mass_flux_y[i, j] / wp.max(rho_face_y[i, j] * dx * ap_b, 1.0e-12) if ap_b > 1.0e-6 else 0.0
    v_face_t = mass_flux_y[i, j + 1] / wp.max(rho_face_y[i, j + 1] * dx * ap_t, 1.0e-12) if ap_t > 1.0e-6 else 0.0
    u[i, j] = (ap_l * u_face_l + ap_r * u_face_r) / wp.max(ap_l + ap_r, 1.0e-12)
    v[i, j] = (ap_b * v_face_b + ap_t * v_face_t) / wp.max(ap_b + ap_t, 1.0e-12)


@wp.kernel
def _limit_velocity_kernel(
    u: wp.array2d(dtype=wp.float32),
    v: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    u_ref: float,
    v_ref: float,
    speed_limit: float,
):
    i, j = wp.tid()
    if i >= nx or j >= ny or fluid[i, j] <= 0.5:
        return
    u_val = u[i, j]
    v_val = v[i, j]
    if wp.isnan(u_val) or wp.isinf(u_val) or wp.isnan(v_val) or wp.isinf(v_val):
        u[i, j] = u_ref
        v[i, j] = v_ref
        return
    speed = wp.sqrt(wp.max(u_val * u_val + v_val * v_val, 0.0))
    if speed > speed_limit:
        scale = speed_limit / wp.max(speed, 1.0e-12)
        u[i, j] = u_val * scale
        v[i, j] = v_val * scale


@wp.kernel
def _update_pressure_density_kernel(
    pressure: wp.array2d(dtype=wp.float32),
    density: wp.array2d(dtype=wp.float32),
    temperature: wp.array2d(dtype=wp.float32),
    pcorr: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    pressure_relaxation: float,
    gas_constant: float,
    min_pressure: float,
    min_density: float,
):
    i, j = wp.tid()
    if i >= nx or j >= ny or fluid[i, j] <= 0.5:
        return
    p = wp.max(pressure[i, j] + pressure_relaxation * pcorr[i, j], min_pressure)
    pressure[i, j] = p
    density[i, j] = wp.max(p / wp.max(gas_constant * temperature[i, j], 1.0e-12), min_density)


@wp.kernel
def _stabilize_thermo_kernel(
    density: wp.array2d(dtype=wp.float32),
    pressure: wp.array2d(dtype=wp.float32),
    temperature: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    rho_ref: float,
    pressure_ref: float,
    temperature_ref: float,
    rho_min_ratio: float,
    rho_max_ratio: float,
    pressure_min_ratio: float,
    pressure_max_ratio: float,
    temperature_min_ratio: float,
    temperature_max_ratio: float,
):
    i, j = wp.tid()
    if i >= nx or j >= ny or fluid[i, j] <= 0.5:
        return
    rho_min = rho_min_ratio * rho_ref
    rho_max = rho_max_ratio * rho_ref
    pressure_min = pressure_min_ratio * pressure_ref
    pressure_max = pressure_max_ratio * pressure_ref
    temperature_min = temperature_min_ratio * temperature_ref
    temperature_max = temperature_max_ratio * temperature_ref

    rho = density[i, j]
    p = pressure[i, j]
    t = temperature[i, j]
    if wp.isnan(rho) or wp.isinf(rho):
        rho = rho_ref
    if wp.isnan(p) or wp.isinf(p):
        p = pressure_ref
    if wp.isnan(t) or wp.isinf(t):
        t = temperature_ref
    density[i, j] = wp.min(wp.max(rho, rho_min), rho_max)
    pressure[i, j] = wp.min(wp.max(p, pressure_min), pressure_max)
    temperature[i, j] = wp.min(wp.max(t, temperature_min), temperature_max)


@wp.kernel
def _turbulence_update_kernel(
    density: wp.array2d(dtype=wp.float32),
    u: wp.array2d(dtype=wp.float32),
    v: wp.array2d(dtype=wp.float32),
    k: wp.array2d(dtype=wp.float32),
    omega: wp.array2d(dtype=wp.float32),
    mu_t: wp.array2d(dtype=wp.float32),
    gamma_k: wp.array2d(dtype=wp.float32),
    gamma_w: wp.array2d(dtype=wp.float32),
    alpha: wp.array2d(dtype=wp.float32),
    beta: wp.array2d(dtype=wp.float32),
    f1: wp.array2d(dtype=wp.float32),
    strain: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    fluid_fraction: wp.array2d(dtype=wp.float32),
    aperture_x: wp.array2d(dtype=wp.float32),
    aperture_y: wp.array2d(dtype=wp.float32),
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    dt: float,
    min_density: float,
    min_k: float,
    min_omega: float,
    turbulence_relaxation: float,
    k_inf: float,
    omega_inf: float,
):
    i, j = wp.tid()
    if i >= nx or j >= ny:
        return
    if fluid[i, j] <= 0.5:
        k[i, j] = min_k
        omega[i, j] = wp.max(omega_inf, min_omega)
        return
    rho = wp.max(density[i, j], min_density)
    k_safe = wp.max(k[i, j], min_k)
    omega_safe = wp.max(omega[i, j], min_omega)
    production = wp.min(mu_t[i, j] * strain[i, j] * strain[i, j], 10.0 * 0.09 * rho * k_safe * omega_safe)
    grad_kx = _gradient_x(k, i, j, dx, nx, ny)
    grad_ky = _gradient_y(k, i, j, dy, nx, ny)
    grad_wx = _gradient_x(omega, i, j, dx, nx, ny)
    grad_wy = _gradient_y(omega, i, j, dy, nx, ny)
    cross_diff = 2.0 * (1.0 - f1[i, j]) * 0.856 * (grad_kx * grad_wx + grad_ky * grad_wy) / wp.max(omega_safe, min_omega)
    k_diff = _diffusion(k, gamma_k, fluid, fluid_fraction, aperture_x, aperture_y, i, j, dx, dy, nx, ny)
    w_diff = _diffusion(omega, gamma_w, fluid, fluid_fraction, aperture_x, aperture_y, i, j, dx, dy, nx, ny)
    k_conv = _convection(k, density, u, v, fluid, fluid_fraction, aperture_x, aperture_y, i, j, dx, dy, nx, ny)
    w_conv = _convection(omega, density, u, v, fluid, fluid_fraction, aperture_x, aperture_y, i, j, dx, dy, nx, ny)
    k_rhs = -k_conv / rho + k_diff / rho + production / rho - 0.09 * k_safe * omega_safe
    w_rhs = -w_conv / rho + w_diff / rho + alpha[i, j] * strain[i, j] * strain[i, j] - beta[i, j] * omega_safe * omega_safe + cross_diff
    k_new = (1.0 - turbulence_relaxation) * k_safe + turbulence_relaxation * (k_safe + dt * k_rhs)
    w_new = (1.0 - turbulence_relaxation) * omega_safe + turbulence_relaxation * (omega_safe + dt * w_rhs)
    k_limit = wp.max(4.0 * k_safe, 10.0 * k_inf)
    omega_limit = wp.max(4.0 * omega_safe, 10.0 * omega_inf)
    k[i, j] = wp.min(wp.max(k_new, min_k), k_limit)
    omega[i, j] = wp.min(wp.max(w_new, min_omega), omega_limit)


@wp.kernel
def _mass_residual_kernel(
    mass_flux_x: wp.array2d(dtype=wp.float32),
    mass_flux_y: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    fluid_fraction: wp.array2d(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
    nx: int,
    ny: int,
    cell_volume: float,
):
    i, j = wp.tid()
    if i >= nx or j >= ny or fluid[i, j] <= 0.5:
        return
    volume = wp.max(_cell_fraction(fluid_fraction, fluid, i, j) * cell_volume, 1.0e-12)
    value = wp.abs(((mass_flux_x[i + 1, j] - mass_flux_x[i, j]) + (mass_flux_y[i, j + 1] - mass_flux_y[i, j])) / volume)
    if not wp.isnan(value) and not wp.isinf(value):
        wp.atomic_max(out, 0, value)


@wp.kernel
def _max_abs_diff_kernel(
    a: wp.array2d(dtype=wp.float32),
    b: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
    nx: int,
    ny: int,
):
    i, j = wp.tid()
    if i >= nx or j >= ny or fluid[i, j] <= 0.5:
        return
    value = wp.abs(a[i, j] - b[i, j])
    if not wp.isnan(value) and not wp.isinf(value):
        wp.atomic_max(out, 0, value)


@wp.kernel
def _pair_max_abs_diff_kernel(
    a0: wp.array2d(dtype=wp.float32),
    b0: wp.array2d(dtype=wp.float32),
    a1: wp.array2d(dtype=wp.float32),
    b1: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
    nx: int,
    ny: int,
):
    i, j = wp.tid()
    if i >= nx or j >= ny or fluid[i, j] <= 0.5:
        return
    value0 = wp.abs(a0[i, j] - b0[i, j])
    value1 = wp.abs(a1[i, j] - b1[i, j])
    if not wp.isnan(value0) and not wp.isinf(value0):
        wp.atomic_max(out, 0, value0)
    if not wp.isnan(value1) and not wp.isinf(value1):
        wp.atomic_max(out, 1, value1)


@wp.kernel
def _aero_reduce_kernel(
    pressure: wp.array2d(dtype=wp.float32),
    u: wp.array2d(dtype=wp.float32),
    v: wp.array2d(dtype=wp.float32),
    mu_t: wp.array2d(dtype=wp.float32),
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
    laminar_viscosity: float,
):
    i, j = wp.tid()
    if i >= nx or j >= ny or fluid[i, j] <= 0.5:
        return
    solid_east = i + 1 < nx and fluid[i + 1, j] <= 0.5
    solid_west = i > 0 and fluid[i - 1, j] <= 0.5
    solid_north = j + 1 < ny and fluid[i, j + 1] <= 0.5
    solid_south = j > 0 and fluid[i, j - 1] <= 0.5
    mu = laminar_viscosity + mu_t[i, j]
    if solid_east:
        force_x = pressure[i, j] * dy
        force_y = -mu * v[i, j] / (0.5 * dx) * dy
        x_face = x0 + float(i + 1) * dx
        y_face = y0 + (float(j) + 0.5) * dy
        wp.atomic_add(out, 0, force_x)
        wp.atomic_add(out, 1, force_y)
        wp.atomic_add(out, 2, (x_face - ref_x) * force_y - (y_face - ref_y) * force_x)
    if solid_west:
        force_x = -pressure[i, j] * dy
        force_y = -mu * v[i, j] / (0.5 * dx) * dy
        x_face = x0 + float(i) * dx
        y_face = y0 + (float(j) + 0.5) * dy
        wp.atomic_add(out, 0, force_x)
        wp.atomic_add(out, 1, force_y)
        wp.atomic_add(out, 2, (x_face - ref_x) * force_y - (y_face - ref_y) * force_x)
    if solid_north:
        force_x = -mu * u[i, j] / (0.5 * dy) * dx
        force_y = pressure[i, j] * dx
        x_face = x0 + (float(i) + 0.5) * dx
        y_face = y0 + float(j + 1) * dy
        wp.atomic_add(out, 0, force_x)
        wp.atomic_add(out, 1, force_y)
        wp.atomic_add(out, 2, (x_face - ref_x) * force_y - (y_face - ref_y) * force_x)
    if solid_south:
        force_x = -mu * u[i, j] / (0.5 * dy) * dx
        force_y = -pressure[i, j] * dx
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


def _make_scratch(grid: GridSpec, device: object) -> _PressureBasedWarpScratch:
    nx, ny = grid.shape
    return _PressureBasedWarpScratch(
        pressure_extended=wp.zeros((nx, ny), dtype=wp.float32, device=device),
        density_extended=wp.zeros((nx, ny), dtype=wp.float32, device=device),
        temperature_extended=wp.zeros((nx, ny), dtype=wp.float32, device=device),
        scalar_tmp=wp.zeros((nx, ny), dtype=wp.float32, device=device),
        mu_eff=wp.zeros((nx, ny), dtype=wp.float32, device=device),
        alpha_eff=wp.zeros((nx, ny), dtype=wp.float32, device=device),
        gamma_k=wp.zeros((nx, ny), dtype=wp.float32, device=device),
        gamma_w=wp.zeros((nx, ny), dtype=wp.float32, device=device),
        mu_t=wp.zeros((nx, ny), dtype=wp.float32, device=device),
        sigma_k=wp.zeros((nx, ny), dtype=wp.float32, device=device),
        sigma_w=wp.zeros((nx, ny), dtype=wp.float32, device=device),
        alpha=wp.zeros((nx, ny), dtype=wp.float32, device=device),
        beta=wp.zeros((nx, ny), dtype=wp.float32, device=device),
        f1=wp.zeros((nx, ny), dtype=wp.float32, device=device),
        strain=wp.zeros((nx, ny), dtype=wp.float32, device=device),
        temperature_next=wp.zeros((nx, ny), dtype=wp.float32, device=device),
        u_prev=wp.zeros((nx, ny), dtype=wp.float32, device=device),
        v_prev=wp.zeros((nx, ny), dtype=wp.float32, device=device),
        t_prev=wp.zeros((nx, ny), dtype=wp.float32, device=device),
        k_prev=wp.zeros((nx, ny), dtype=wp.float32, device=device),
        omega_prev=wp.zeros((nx, ny), dtype=wp.float32, device=device),
        u_star=wp.zeros((nx, ny), dtype=wp.float32, device=device),
        v_star=wp.zeros((nx, ny), dtype=wp.float32, device=device),
        rho_face_x=wp.zeros((nx + 1, ny), dtype=wp.float32, device=device),
        rho_face_y=wp.zeros((nx, ny + 1), dtype=wp.float32, device=device),
        mass_flux_x=wp.zeros((nx + 1, ny), dtype=wp.float32, device=device),
        mass_flux_y=wp.zeros((nx, ny + 1), dtype=wp.float32, device=device),
        divergence_star=wp.zeros((nx, ny), dtype=wp.float32, device=device),
        beta_x=wp.zeros((nx + 1, ny), dtype=wp.float32, device=device),
        beta_y=wp.zeros((nx, ny + 1), dtype=wp.float32, device=device),
        pcorr=wp.zeros((nx, ny), dtype=wp.float32, device=device),
        scalar_buffer=wp.zeros((1,), dtype=wp.float32, device=device),
        pair_buffer=wp.zeros((2,), dtype=wp.float32, device=device),
        triple_buffer=wp.zeros((3,), dtype=wp.float32, device=device),
    )


def _fill_buffer(buffer: object, value: float, count: int, device: object) -> None:
    wp.launch(_fill_scalar_buffer_kernel, dim=count, inputs=(value, buffer, count), device=device)


def _fill_cells(buffer: object, value: float, shape: tuple[int, int], device: object) -> None:
    wp.launch(_fill_cell_kernel, dim=shape, inputs=(value, buffer, shape[0], shape[1]), device=device)


def _extend_scalar_device(src: object, fluid: object, dst: object, tmp: object, shape: tuple[int, int], device: object, passes: int = 4) -> None:
    wp.copy(dst, src)
    current = dst
    other = tmp
    for _ in range(passes):
        wp.launch(_extend_from_fluid_kernel, dim=shape, inputs=(current, fluid, other, shape[0], shape[1]), device=device)
        current, other = other, current
    if current is not dst:
        wp.copy(dst, current)


def initialize_pressure_based_state_warp(
    grid: GridSpec,
    solid_levelset: np.ndarray,
    *,
    chord: float,
    freestream: FreestreamCondition = FreestreamCondition(),
    config: PressureBasedSolverConfig = PressureBasedSolverConfig(),
    device: Optional[str] = None,
) -> PressureBasedWarpState:
    host_state = initialize_pressure_based_state(
        grid,
        solid_levelset,
        chord=chord,
        freestream=freestream,
        config=config,
    )
    resolved_device = _resolve_device(device)
    return PressureBasedWarpState(
        density=wp.array(host_state.density.astype(np.float32), dtype=wp.float32, device=resolved_device),
        u=wp.array(host_state.u.astype(np.float32), dtype=wp.float32, device=resolved_device),
        v=wp.array(host_state.v.astype(np.float32), dtype=wp.float32, device=resolved_device),
        pressure=wp.array(host_state.pressure.astype(np.float32), dtype=wp.float32, device=resolved_device),
        temperature=wp.array(host_state.temperature.astype(np.float32), dtype=wp.float32, device=resolved_device),
        turbulent_kinetic_energy=wp.array(
            host_state.turbulent_kinetic_energy.astype(np.float32), dtype=wp.float32, device=resolved_device
        ),
        specific_dissipation=wp.array(
            host_state.specific_dissipation.astype(np.float32), dtype=wp.float32, device=resolved_device
        ),
        turbulent_viscosity=wp.array(
            host_state.turbulent_viscosity.astype(np.float32), dtype=wp.float32, device=resolved_device
        ),
        wall_distance=wp.array(host_state.wall_distance.astype(np.float32), dtype=wp.float32, device=resolved_device),
        fluid=wp.array(host_state.fluid.astype(np.float32), dtype=wp.float32, device=resolved_device),
        fluid_fraction=wp.array(host_state.fluid_fraction.astype(np.float32), dtype=wp.float32, device=resolved_device),
        aperture_x=wp.array(host_state.aperture_x.astype(np.float32), dtype=wp.float32, device=resolved_device),
        aperture_y=wp.array(host_state.aperture_y.astype(np.float32), dtype=wp.float32, device=resolved_device),
        solid_levelset=wp.array(host_state.solid_levelset.astype(np.float32), dtype=wp.float32, device=resolved_device),
        laminar_viscosity=float(host_state.laminar_viscosity),
        dt=float(host_state.dt),
    )


def _state_to_device(state: PressureBasedState, device: object) -> PressureBasedWarpState:
    return PressureBasedWarpState(
        density=wp.array(state.density.astype(np.float32), dtype=wp.float32, device=device),
        u=wp.array(state.u.astype(np.float32), dtype=wp.float32, device=device),
        v=wp.array(state.v.astype(np.float32), dtype=wp.float32, device=device),
        pressure=wp.array(state.pressure.astype(np.float32), dtype=wp.float32, device=device),
        temperature=wp.array(state.temperature.astype(np.float32), dtype=wp.float32, device=device),
        turbulent_kinetic_energy=wp.array(
            state.turbulent_kinetic_energy.astype(np.float32), dtype=wp.float32, device=device
        ),
        specific_dissipation=wp.array(state.specific_dissipation.astype(np.float32), dtype=wp.float32, device=device),
        turbulent_viscosity=wp.array(state.turbulent_viscosity.astype(np.float32), dtype=wp.float32, device=device),
        wall_distance=wp.array(state.wall_distance.astype(np.float32), dtype=wp.float32, device=device),
        fluid=wp.array(state.fluid.astype(np.float32), dtype=wp.float32, device=device),
        fluid_fraction=wp.array(state.fluid_fraction.astype(np.float32), dtype=wp.float32, device=device),
        aperture_x=wp.array(state.aperture_x.astype(np.float32), dtype=wp.float32, device=device),
        aperture_y=wp.array(state.aperture_y.astype(np.float32), dtype=wp.float32, device=device),
        solid_levelset=wp.array(state.solid_levelset.astype(np.float32), dtype=wp.float32, device=device),
        laminar_viscosity=float(state.laminar_viscosity),
        dt=float(state.dt),
    )


def _apply_boundaries_device(
    state: PressureBasedWarpState,
    scratch: _PressureBasedWarpScratch,
    grid: GridSpec,
    freestream: FreestreamCondition,
    config: PressureBasedSolverConfig,
    *,
    u_inf: float,
    v_inf: float,
    rho_inf: float,
    k_inf: float,
    omega_inf: float,
    wall_floor: float,
) -> None:
    device = state.density.device
    nx, ny = grid.shape
    pressure_out = config.outlet_static_pressure or freestream.static_pressure
    wp.launch(
        _apply_outer_boundaries_kernel,
        dim=grid.shape,
        inputs=(
            state.density,
            state.u,
            state.v,
            state.pressure,
            state.temperature,
            state.turbulent_kinetic_energy,
            state.specific_dissipation,
            state.turbulent_viscosity,
            nx,
            ny,
            rho_inf,
            u_inf,
            v_inf,
            freestream.static_pressure,
            pressure_out,
            freestream.static_temperature,
            k_inf,
            omega_inf,
        ),
        device=device,
    )
    wall_temperature = float(config.wall_temperature) if config.wall_temperature is not None else 0.0
    use_wall_temperature = 1.0 if config.wall_temperature is not None else 0.0
    wp.launch(
        _solid_enforce_kernel,
        dim=grid.shape,
        inputs=(
            state.density,
            state.u,
            state.v,
            state.temperature,
            state.turbulent_kinetic_energy,
            state.specific_dissipation,
            state.turbulent_viscosity,
            state.wall_distance,
            state.fluid,
            nx,
            ny,
            state.laminar_viscosity,
            wall_floor,
            config.min_density,
            config.min_turbulent_kinetic_energy,
            omega_inf,
            wall_temperature,
            use_wall_temperature,
        ),
        device=device,
    )
    wp.launch(
        _near_wall_velocity_damping_kernel,
        dim=grid.shape,
        inputs=(
            state.u,
            state.v,
            state.wall_distance,
            state.fluid,
            state.fluid_fraction,
            nx,
            ny,
            1.5 * max(grid.dx, grid.dy),
        ),
        device=device,
    )
    _extend_scalar_device(state.pressure, state.fluid, scratch.pressure_extended, scratch.scalar_tmp, grid.shape, device)
    wp.copy(state.pressure, scratch.pressure_extended)
    _extend_scalar_device(state.density, state.fluid, scratch.density_extended, scratch.scalar_tmp, grid.shape, device)
    wp.copy(state.density, scratch.density_extended)
    if config.wall_temperature is None:
        _extend_scalar_device(
            state.temperature,
            state.fluid,
            scratch.temperature_extended,
            scratch.scalar_tmp,
            grid.shape,
            device,
        )
        wp.copy(state.temperature, scratch.temperature_extended)
    wp.launch(
        _clip_state_kernel,
        dim=grid.shape,
        inputs=(
            state.density,
            state.pressure,
            state.temperature,
            state.turbulent_kinetic_energy,
            state.specific_dissipation,
            nx,
            ny,
            config.min_density,
            config.min_pressure,
            config.min_temperature,
            config.min_turbulent_kinetic_energy,
            config.min_specific_dissipation,
        ),
        device=device,
    )


def _pressure_correction_sor_device(
    state: PressureBasedWarpState,
    scratch: _PressureBasedWarpScratch,
    grid: GridSpec,
    config: PressureBasedSolverConfig,
) -> float:
    device = state.density.device
    nx, ny = grid.shape
    _fill_cells(scratch.pcorr, 0.0, grid.shape, device)
    wp.launch(
        _beta_x_kernel,
        dim=(nx + 1, ny),
        inputs=(scratch.rho_face_x, state.aperture_x, scratch.beta_x, nx, ny, state.dt, grid.dx),
        device=device,
    )
    wp.launch(
        _beta_y_kernel,
        dim=(nx, ny + 1),
        inputs=(scratch.rho_face_y, state.aperture_y, scratch.beta_y, nx, ny, state.dt, grid.dy),
        device=device,
    )
    pressure_residual = 0.0
    for _ in range(config.pressure_correction_iterations):
        _fill_buffer(scratch.scalar_buffer, 0.0, 1, device)
        wp.launch(
            _rbgs_pressure_kernel,
            dim=grid.shape,
            inputs=(
                scratch.pcorr,
                scratch.divergence_star,
                scratch.beta_x,
                scratch.beta_y,
                state.fluid,
                state.fluid_fraction,
                scratch.scalar_buffer,
                nx,
                ny,
                0,
                config.pressure_sor_omega,
            ),
            device=device,
        )
        wp.launch(
            _rbgs_pressure_kernel,
            dim=grid.shape,
            inputs=(
                scratch.pcorr,
                scratch.divergence_star,
                scratch.beta_x,
                scratch.beta_y,
                state.fluid,
                state.fluid_fraction,
                scratch.scalar_buffer,
                nx,
                ny,
                1,
                config.pressure_sor_omega,
            ),
            device=device,
        )
        max_change = float(scratch.scalar_buffer.numpy()[0])
        _fill_buffer(scratch.scalar_buffer, 0.0, 1, device)
        wp.launch(
            _pressure_residual_kernel,
            dim=grid.shape,
            inputs=(
                scratch.pcorr,
                scratch.divergence_star,
                scratch.beta_x,
                scratch.beta_y,
                state.fluid,
                state.fluid_fraction,
                scratch.scalar_buffer,
                nx,
                ny,
            ),
            device=device,
        )
        pressure_residual = float(scratch.scalar_buffer.numpy()[0])
        if pressure_residual < config.pressure_correction_tolerance or max_change < 1.0e-8:
            break
    return pressure_residual


def run_pressure_based_solver_warp(
    grid: GridSpec,
    solid_levelset: np.ndarray,
    *,
    chord: float,
    freestream: FreestreamCondition = FreestreamCondition(),
    config: PressureBasedSolverConfig = PressureBasedSolverConfig(),
    device: Optional[str] = None,
    initial_state: Optional[PressureBasedWarpState | PressureBasedState] = None,
    step_offset: int = 0,
    stop_on_convergence: bool = True,
    step_callback: Optional[StepCallback] = None,
) -> tuple[PressureBasedWarpState, list[ResidualSnapshot]]:
    if grid.is_3d:
        raise NotImplementedError("The experimental pressure-based Warp solver currently supports 2D grids only.")
    if config.pressure_linear_solver != "SOR":
        raise NotImplementedError("The Warp pressure-based solver currently supports only the SOR pressure solver.")

    if initial_state is None:
        state = initialize_pressure_based_state_warp(
            grid,
            solid_levelset,
            chord=chord,
            freestream=freestream,
            config=config,
            device=device,
        )
    elif isinstance(initial_state, PressureBasedWarpState):
        state = initial_state
    else:
        state = _state_to_device(initial_state, _resolve_device(device))
    device_obj = state.density.device
    scratch = _make_scratch(grid, device_obj)
    host_state = state.to_numpy_state()

    u_inf, v_inf, _ = _freestream_velocity(freestream, config)
    speed_inf = math.sqrt(u_inf * u_inf + v_inf * v_inf)
    rho_inf = freestream.static_pressure / (config.gas_constant * freestream.static_temperature)
    k_inf = max(1.5 * (speed_inf * freestream.turbulence_intensity) ** 2, config.min_turbulent_kinetic_energy)
    omega_inf = max(
        rho_inf * k_inf / (state.laminar_viscosity * freestream.turbulent_viscosity_ratio),
        config.min_specific_dissipation,
    )
    wall_floor = max(float(np.min(host_state.wall_distance[host_state.fluid])) if np.any(host_state.fluid) else 0.0, 1.0e-6)
    solid = ~host_state.fluid
    solid_i, solid_j = np.nonzero(solid)
    if solid_i.size:
        leading_edge_x = float(np.min(grid.x0 + (solid_i.astype(np.float64) + 0.5) * grid.dx))
        leading_edge_y = float(np.mean(grid.y0 + (solid_j.astype(np.float64) + 0.5) * grid.dy))
    else:
        leading_edge_x = grid.x0
        leading_edge_y = grid.y0 + 0.5 * grid.extent[1]
    ref_x, ref_y = _reference_point(chord, leading_edge_x, leading_edge_y)
    q_inf = max(_dynamic_pressure(freestream, config), 1.0e-6)
    alpha = math.radians(freestream.angle_of_attack_deg)
    drag_dir = (math.cos(alpha), math.sin(alpha))
    lift_dir = (-math.sin(alpha), math.cos(alpha))
    pressure_ref = freestream.static_pressure
    temperature_ref = freestream.static_temperature
    density_ref = rho_inf
    speed_limit = max(3.0 * speed_inf, 1.0)
    pcorr_pressure_fraction = 0.2
    pcorr_pressure_scale = 0.05 * pressure_ref
    rho_min_ratio = 0.2
    rho_max_ratio = 5.0
    pressure_min_ratio = 0.2
    pressure_max_ratio = 5.0
    temperature_min_ratio = 0.5
    temperature_max_ratio = 2.0
    mu_t_max_ratio = 100.0

    history: list[ResidualSnapshot] = []
    residual_reference: Optional[tuple[float, float, float, float]] = None
    previous_forces: Optional[AerodynamicCoefficients] = None

    for step in range(1, config.pseudo_steps + 1):
        wp.copy(scratch.u_prev, state.u)
        wp.copy(scratch.v_prev, state.v)
        wp.copy(scratch.t_prev, state.temperature)
        wp.copy(scratch.k_prev, state.turbulent_kinetic_energy)
        wp.copy(scratch.omega_prev, state.specific_dissipation)

        _fill_buffer(scratch.scalar_buffer, 0.0, 1, device_obj)
        wp.launch(
            _wave_speed_max_kernel,
            dim=grid.shape,
            inputs=(
                state.u,
                state.v,
                state.temperature,
                state.fluid,
                scratch.scalar_buffer,
                grid.nx,
                grid.ny,
                config.gamma,
                config.gas_constant,
            ),
            device=device_obj,
        )
        max_speed = float(scratch.scalar_buffer.numpy()[0])
        _fill_buffer(scratch.scalar_buffer, 0.0, 1, device_obj)
        wp.launch(
            _effective_diffusivity_max_kernel,
            dim=grid.shape,
            inputs=(
                state.density,
                state.turbulent_viscosity,
                state.fluid,
                scratch.scalar_buffer,
                grid.nx,
                grid.ny,
                state.laminar_viscosity,
                config.min_density,
            ),
            device=device_obj,
        )
        max_nu_eff = float(scratch.scalar_buffer.numpy()[0])
        _fill_buffer(scratch.scalar_buffer, 0.0, 1, device_obj)
        wp.launch(
            _omega_max_kernel,
            dim=grid.shape,
            inputs=(
                state.specific_dissipation,
                state.fluid,
                scratch.scalar_buffer,
                grid.nx,
                grid.ny,
            ),
            device=device_obj,
        )
        max_omega = float(scratch.scalar_buffer.numpy()[0])
        h = min(grid.dx, grid.dy)
        dt_conv = h / max(max_speed, 1.0)
        dt_diff = 0.5 * h * h / max(max_nu_eff, 1.0e-12)
        dt_turb = 0.25 / max(max_omega, 1.0)
        state.dt = min(config.cfl * dt_conv, dt_diff, dt_turb)

        wp.launch(
            _compute_alpha_eff_kernel,
            dim=grid.shape,
            inputs=(
                state.density,
                state.turbulent_viscosity,
                scratch.alpha_eff,
                grid.nx,
                grid.ny,
                state.laminar_viscosity,
                config.min_density,
                config.prandtl,
                config.turbulent_prandtl,
            ),
            device=device_obj,
        )
        wp.launch(
            _temperature_update_kernel,
            dim=grid.shape,
            inputs=(
                state.density,
                state.u,
                state.v,
                state.temperature,
                scratch.alpha_eff,
                state.fluid,
                state.fluid_fraction,
                state.aperture_x,
                state.aperture_y,
                scratch.temperature_next,
                grid.nx,
                grid.ny,
                grid.dx,
                grid.dy,
                state.dt,
                config.min_density,
                config.min_temperature,
                config.temperature_relaxation,
                config.gamma,
                config.gas_constant,
            ),
            device=device_obj,
        )
        wp.copy(state.temperature, scratch.temperature_next)
        wp.launch(
            _recompute_density_kernel,
            dim=grid.shape,
            inputs=(
                state.density,
                state.pressure,
                state.temperature,
                state.fluid,
                grid.nx,
                grid.ny,
                config.gas_constant,
                config.min_density,
            ),
            device=device_obj,
        )
        _apply_boundaries_device(
            state,
            scratch,
            grid,
            freestream,
            config,
            u_inf=u_inf,
            v_inf=v_inf,
            rho_inf=rho_inf,
            k_inf=k_inf,
            omega_inf=omega_inf,
            wall_floor=wall_floor,
        )
        wp.launch(
            _compute_turbulence_aux_kernel,
            dim=grid.shape,
            inputs=(
                state.density,
                state.turbulent_kinetic_energy,
                state.specific_dissipation,
                state.u,
                state.v,
                state.wall_distance,
                state.fluid,
                scratch.mu_t,
                scratch.sigma_k,
                scratch.sigma_w,
                scratch.alpha,
                scratch.beta,
                scratch.f1,
                scratch.strain,
                grid.nx,
                grid.ny,
                grid.dx,
                grid.dy,
                state.laminar_viscosity,
                wall_floor,
            ),
            device=device_obj,
        )
        wp.copy(state.turbulent_viscosity, scratch.mu_t)
        wp.launch(
            _limit_mu_t_kernel,
            dim=grid.shape,
            inputs=(state.turbulent_viscosity, state.fluid, grid.nx, grid.ny, state.laminar_viscosity, mu_t_max_ratio),
            device=device_obj,
        )

        pressure_residual = 0.0
        for _ in range(config.simple_iterations):
            _extend_scalar_device(
                state.pressure,
                state.fluid,
                scratch.pressure_extended,
                scratch.scalar_tmp,
                grid.shape,
                device_obj,
            )
            wp.launch(
                _mu_eff_kernel,
                dim=grid.shape,
                inputs=(state.turbulent_viscosity, scratch.mu_eff, grid.nx, grid.ny, state.laminar_viscosity),
                device=device_obj,
            )
            wp.launch(
                _simple_predict_kernel,
                dim=grid.shape,
                inputs=(
                    state.density,
                    state.u,
                    state.v,
                    scratch.pressure_extended,
                    scratch.mu_eff,
                    state.fluid,
                    state.fluid_fraction,
                    state.aperture_x,
                    state.aperture_y,
                    scratch.u_star,
                    scratch.v_star,
                    grid.nx,
                    grid.ny,
                    grid.dx,
                    grid.dy,
                    state.dt,
                    config.velocity_relaxation,
                    config.min_density,
                ),
                device=device_obj,
            )
            wp.launch(
                _mass_flux_x_kernel,
                dim=(grid.nx + 1, grid.ny),
                inputs=(
                    state.density,
                    scratch.u_star,
                    state.aperture_x,
                    scratch.rho_face_x,
                    scratch.mass_flux_x,
                    grid.nx,
                    grid.ny,
                    grid.dy,
                ),
                device=device_obj,
            )
            wp.launch(
                _mass_flux_y_kernel,
                dim=(grid.nx, grid.ny + 1),
                inputs=(
                    state.density,
                    scratch.v_star,
                    state.aperture_y,
                    scratch.rho_face_y,
                    scratch.mass_flux_y,
                    grid.nx,
                    grid.ny,
                    grid.dx,
                ),
                device=device_obj,
            )
            wp.launch(
                _mass_divergence_kernel,
                dim=grid.shape,
                inputs=(
                    scratch.mass_flux_x,
                    scratch.mass_flux_y,
                    state.fluid,
                    state.fluid_fraction,
                    scratch.divergence_star,
                    grid.nx,
                    grid.ny,
                    grid.dx * grid.dy,
                ),
                device=device_obj,
            )
            pressure_residual = _pressure_correction_sor_device(state, scratch, grid, config)
            wp.launch(
                _limit_pcorr_kernel,
                dim=grid.shape,
                inputs=(
                    scratch.pcorr,
                    state.pressure,
                    state.fluid,
                    grid.nx,
                    grid.ny,
                    pcorr_pressure_fraction,
                    pcorr_pressure_scale,
                ),
                device=device_obj,
            )
            wp.launch(
                _correct_mass_flux_x_kernel,
                dim=(grid.nx + 1, grid.ny),
                inputs=(
                    scratch.mass_flux_x,
                    scratch.rho_face_x,
                    state.aperture_x,
                    scratch.pcorr,
                    state.fluid,
                    grid.nx,
                    grid.ny,
                    state.dt,
                    grid.dx,
                    grid.dy,
                ),
                device=device_obj,
            )
            wp.launch(
                _correct_mass_flux_y_kernel,
                dim=(grid.nx, grid.ny + 1),
                inputs=(
                    scratch.mass_flux_y,
                    scratch.rho_face_y,
                    state.aperture_y,
                    scratch.pcorr,
                    state.fluid,
                    grid.nx,
                    grid.ny,
                    state.dt,
                    grid.dx,
                    grid.dy,
                ),
                device=device_obj,
            )
            wp.launch(
                _cell_velocity_from_fluxes_kernel,
                dim=grid.shape,
                inputs=(
                    scratch.mass_flux_x,
                    scratch.mass_flux_y,
                    scratch.rho_face_x,
                    scratch.rho_face_y,
                    state.fluid,
                    state.aperture_x,
                    state.aperture_y,
                    state.u,
                    state.v,
                    grid.nx,
                    grid.ny,
                    grid.dx,
                    grid.dy,
                ),
                device=device_obj,
            )
            wp.launch(
                _limit_velocity_kernel,
                dim=grid.shape,
                inputs=(state.u, state.v, state.fluid, grid.nx, grid.ny, u_inf, v_inf, speed_limit),
                device=device_obj,
            )
            wp.launch(
                _update_pressure_density_kernel,
                dim=grid.shape,
                inputs=(
                    state.pressure,
                    state.density,
                    state.temperature,
                    scratch.pcorr,
                    state.fluid,
                    grid.nx,
                    grid.ny,
                    config.pressure_relaxation,
                    config.gas_constant,
                    config.min_pressure,
                    config.min_density,
                ),
                device=device_obj,
            )
            wp.launch(
                _stabilize_thermo_kernel,
                dim=grid.shape,
                inputs=(
                    state.density,
                    state.pressure,
                    state.temperature,
                    state.fluid,
                    grid.nx,
                    grid.ny,
                    density_ref,
                    pressure_ref,
                    temperature_ref,
                    rho_min_ratio,
                    rho_max_ratio,
                    pressure_min_ratio,
                    pressure_max_ratio,
                    temperature_min_ratio,
                    temperature_max_ratio,
                ),
                device=device_obj,
            )
            _apply_boundaries_device(
                state,
                scratch,
                grid,
                freestream,
                config,
                u_inf=u_inf,
                v_inf=v_inf,
                rho_inf=rho_inf,
                k_inf=k_inf,
                omega_inf=omega_inf,
                wall_floor=wall_floor,
            )

        wp.launch(
            _compute_turbulence_aux_kernel,
            dim=grid.shape,
            inputs=(
                state.density,
                state.turbulent_kinetic_energy,
                state.specific_dissipation,
                state.u,
                state.v,
                state.wall_distance,
                state.fluid,
                scratch.mu_t,
                scratch.sigma_k,
                scratch.sigma_w,
                scratch.alpha,
                scratch.beta,
                scratch.f1,
                scratch.strain,
                grid.nx,
                grid.ny,
                grid.dx,
                grid.dy,
                state.laminar_viscosity,
                wall_floor,
            ),
            device=device_obj,
        )
        wp.launch(
            _gamma_from_sigma_kernel,
            dim=grid.shape,
            inputs=(scratch.sigma_k, scratch.mu_t, scratch.gamma_k, grid.nx, grid.ny, state.laminar_viscosity),
            device=device_obj,
        )
        wp.launch(
            _gamma_from_sigma_kernel,
            dim=grid.shape,
            inputs=(scratch.sigma_w, scratch.mu_t, scratch.gamma_w, grid.nx, grid.ny, state.laminar_viscosity),
            device=device_obj,
        )
        wp.launch(
            _turbulence_update_kernel,
            dim=grid.shape,
            inputs=(
                state.density,
                state.u,
                state.v,
                state.turbulent_kinetic_energy,
                state.specific_dissipation,
                scratch.mu_t,
                scratch.gamma_k,
                scratch.gamma_w,
                scratch.alpha,
                scratch.beta,
                scratch.f1,
                scratch.strain,
                state.fluid,
                state.fluid_fraction,
                state.aperture_x,
                state.aperture_y,
                grid.nx,
                grid.ny,
                grid.dx,
                grid.dy,
                state.dt,
                config.min_density,
                config.min_turbulent_kinetic_energy,
                config.min_specific_dissipation,
                config.turbulence_relaxation,
                k_inf,
                omega_inf,
            ),
            device=device_obj,
        )
        wp.launch(
            _compute_turbulence_aux_kernel,
            dim=grid.shape,
            inputs=(
                state.density,
                state.turbulent_kinetic_energy,
                state.specific_dissipation,
                state.u,
                state.v,
                state.wall_distance,
                state.fluid,
                scratch.mu_t,
                scratch.sigma_k,
                scratch.sigma_w,
                scratch.alpha,
                scratch.beta,
                scratch.f1,
                scratch.strain,
                grid.nx,
                grid.ny,
                grid.dx,
                grid.dy,
                state.laminar_viscosity,
                wall_floor,
            ),
            device=device_obj,
        )
        wp.copy(state.turbulent_viscosity, scratch.mu_t)
        wp.launch(
            _limit_mu_t_kernel,
            dim=grid.shape,
            inputs=(state.turbulent_viscosity, state.fluid, grid.nx, grid.ny, state.laminar_viscosity, mu_t_max_ratio),
            device=device_obj,
        )
        wp.launch(
            _recompute_density_kernel,
            dim=grid.shape,
            inputs=(
                state.density,
                state.pressure,
                state.temperature,
                state.fluid,
                grid.nx,
                grid.ny,
                config.gas_constant,
                config.min_density,
            ),
            device=device_obj,
        )
        wp.launch(
            _stabilize_thermo_kernel,
            dim=grid.shape,
            inputs=(
                state.density,
                state.pressure,
                state.temperature,
                state.fluid,
                grid.nx,
                grid.ny,
                density_ref,
                pressure_ref,
                temperature_ref,
                rho_min_ratio,
                rho_max_ratio,
                pressure_min_ratio,
                pressure_max_ratio,
                temperature_min_ratio,
                temperature_max_ratio,
            ),
            device=device_obj,
        )
        wp.launch(
            _limit_velocity_kernel,
            dim=grid.shape,
            inputs=(state.u, state.v, state.fluid, grid.nx, grid.ny, u_inf, v_inf, speed_limit),
            device=device_obj,
        )
        _apply_boundaries_device(
            state,
            scratch,
            grid,
            freestream,
            config,
            u_inf=u_inf,
            v_inf=v_inf,
            rho_inf=rho_inf,
            k_inf=k_inf,
            omega_inf=omega_inf,
            wall_floor=wall_floor,
        )

        wp.launch(
            _mass_flux_x_kernel,
            dim=(grid.nx + 1, grid.ny),
            inputs=(
                state.density,
                state.u,
                state.aperture_x,
                scratch.rho_face_x,
                scratch.mass_flux_x,
                grid.nx,
                grid.ny,
                grid.dy,
            ),
            device=device_obj,
        )
        wp.launch(
            _mass_flux_y_kernel,
            dim=(grid.nx, grid.ny + 1),
            inputs=(
                state.density,
                state.v,
                state.aperture_y,
                scratch.rho_face_y,
                scratch.mass_flux_y,
                grid.nx,
                grid.ny,
                grid.dx,
            ),
            device=device_obj,
        )

        _fill_buffer(scratch.scalar_buffer, 0.0, 1, device_obj)
        wp.launch(
            _mass_residual_kernel,
            dim=grid.shape,
            inputs=(
                scratch.mass_flux_x,
                scratch.mass_flux_y,
                state.fluid,
                state.fluid_fraction,
                scratch.scalar_buffer,
                grid.nx,
                grid.ny,
                grid.dx * grid.dy,
            ),
            device=device_obj,
        )
        mass_residual = float(scratch.scalar_buffer.numpy()[0])

        _fill_buffer(scratch.pair_buffer, 0.0, 2, device_obj)
        wp.launch(
            _pair_max_abs_diff_kernel,
            dim=grid.shape,
            inputs=(
                state.u,
                scratch.u_prev,
                state.v,
                scratch.v_prev,
                state.fluid,
                scratch.pair_buffer,
                grid.nx,
                grid.ny,
            ),
            device=device_obj,
        )
        momentum_pair = np.asarray(scratch.pair_buffer.numpy(), dtype=np.float64)
        momentum_residual = float(max(momentum_pair[0], momentum_pair[1]))

        _fill_buffer(scratch.scalar_buffer, 0.0, 1, device_obj)
        wp.launch(
            _max_abs_diff_kernel,
            dim=grid.shape,
            inputs=(state.temperature, scratch.t_prev, state.fluid, scratch.scalar_buffer, grid.nx, grid.ny),
            device=device_obj,
        )
        energy_residual = float(scratch.scalar_buffer.numpy()[0])

        _fill_buffer(scratch.pair_buffer, 0.0, 2, device_obj)
        wp.launch(
            _pair_max_abs_diff_kernel,
            dim=grid.shape,
            inputs=(
                state.turbulent_kinetic_energy,
                scratch.k_prev,
                state.specific_dissipation,
                scratch.omega_prev,
                state.fluid,
                scratch.pair_buffer,
                grid.nx,
                grid.ny,
            ),
            device=device_obj,
        )
        turbulence_pair = np.asarray(scratch.pair_buffer.numpy(), dtype=np.float64)
        turbulence_residual = float(max(turbulence_pair[0], turbulence_pair[1]))

        _fill_buffer(scratch.triple_buffer, 0.0, 3, device_obj)
        wp.launch(
            _aero_reduce_kernel,
            dim=grid.shape,
            inputs=(
                state.pressure,
                state.u,
                state.v,
                state.turbulent_viscosity,
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
                state.laminar_viscosity,
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
                max(mass_residual, 1.0e-12),
                max(momentum_residual, 1.0e-12),
                max(energy_residual, 1.0e-12),
                max(turbulence_residual, 1.0e-12),
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
        norm_mass = mass_residual / residual_reference[0]
        norm_momentum = momentum_residual / residual_reference[1]
        norm_energy = energy_residual / residual_reference[2]
        norm_turbulence = turbulence_residual / residual_reference[3]
        snapshot = ResidualSnapshot(
            step=step_offset + step,
            dt=state.dt,
            mass=mass_residual,
            momentum=momentum_residual,
            energy=energy_residual,
            turbulence=turbulence_residual,
            pressure_correction=pressure_residual,
            normalized_mass=norm_mass,
            normalized_momentum=norm_momentum,
            normalized_energy=norm_energy,
            normalized_turbulence=norm_turbulence,
            drag_coefficient=aero.drag,
            lift_coefficient=aero.lift,
            moment_coefficient=aero.moment,
            force_delta=force_delta,
        )
        history.append(snapshot)
        if step_callback is not None:
            step_callback(snapshot.step, state, snapshot)
        recent = history[-config.convergence_window :]
        recent_force_delta = max(item.force_delta for item in recent)
        recent_residual = max(
            max(item.normalized_mass, item.normalized_momentum, item.normalized_energy, item.normalized_turbulence)
            for item in recent
        )
        recent_pressure = max(item.pressure_correction for item in recent)
        if (
            stop_on_convergence
            and
            len(recent) >= config.convergence_window
            and recent_residual < config.convergence_tolerance
            and recent_force_delta < config.force_coefficient_tolerance
            and recent_pressure < config.pressure_correction_tolerance
        ):
            break

    return state, history


__all__ = [
    "PressureBasedWarpState",
    "initialize_pressure_based_state_warp",
    "run_pressure_based_solver_warp",
]
