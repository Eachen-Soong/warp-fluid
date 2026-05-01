from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core.field import CenteredField
from ..core.grid import GridSpec


SUPPORTED_PRESSURE_COUPLING = {"SIMPLE"}
SUPPORTED_TURBULENCE_MODELS = {"sst_k_omega"}
SUPPORTED_CONVECTION_SCHEMES = {"second_order_upwind"}
SUPPORTED_PRESSURE_LINEAR_SOLVERS = {"PCG", "SOR"}


@dataclass(frozen=True)
class FreestreamCondition:
    mach: float = 0.3
    static_pressure: float = 101325.0
    static_temperature: float = 288.15
    angle_of_attack_deg: float = 2.0
    reynolds_number: float = 1.0e6
    turbulence_intensity: float = 0.01
    turbulent_viscosity_ratio: float = 10.0

    def __post_init__(self) -> None:
        if self.mach <= 0.0:
            raise ValueError("mach must be positive.")
        if self.static_pressure <= 0.0:
            raise ValueError("static_pressure must be positive.")
        if self.static_temperature <= 0.0:
            raise ValueError("static_temperature must be positive.")
        if self.reynolds_number <= 0.0:
            raise ValueError("reynolds_number must be positive.")
        if self.turbulence_intensity < 0.0:
            raise ValueError("turbulence_intensity must be non-negative.")
        if self.turbulent_viscosity_ratio <= 0.0:
            raise ValueError("turbulent_viscosity_ratio must be positive.")


@dataclass(frozen=True)
class PressureBasedSolverConfig:
    pseudo_steps: int = 250
    cfl: float = 1.2
    simple_iterations: int = 3
    pressure_correction_iterations: int = 40
    velocity_relaxation: float = 0.6
    pressure_relaxation: float = 0.25
    temperature_relaxation: float = 0.5
    turbulence_relaxation: float = 0.35
    gamma: float = 1.4
    gas_constant: float = 287.05
    prandtl: float = 0.72
    turbulent_prandtl: float = 0.9
    pressure_velocity_coupling: str = "SIMPLE"
    turbulence_model: str = "sst_k_omega"
    convection_scheme: str = "second_order_upwind"
    pressure_linear_solver: str = "SOR"
    pressure_sor_omega: float = 1.6
    pressure_correction_tolerance: float = 1.0e-5
    pressure_pcg_max_iterations: int = 200
    outlet_static_pressure: Optional[float] = None
    wall_temperature: Optional[float] = None
    min_density: float = 1.0e-4
    min_pressure: float = 1.0e2
    min_temperature: float = 50.0
    min_turbulent_kinetic_energy: float = 1.0e-8
    min_specific_dissipation: float = 1.0e-6
    convergence_tolerance: float = 5.0e-4
    force_coefficient_tolerance: float = 1.0e-4
    convergence_window: int = 5

    def __post_init__(self) -> None:
        if self.pseudo_steps < 1:
            raise ValueError("pseudo_steps must be >= 1.")
        if self.cfl <= 0.0:
            raise ValueError("cfl must be positive.")
        if self.simple_iterations < 1:
            raise ValueError("simple_iterations must be >= 1.")
        if self.pressure_correction_iterations < 1:
            raise ValueError("pressure_correction_iterations must be >= 1.")
        if not 0.0 < self.velocity_relaxation <= 1.0:
            raise ValueError("velocity_relaxation must be in (0, 1].")
        if not 0.0 < self.pressure_relaxation <= 1.0:
            raise ValueError("pressure_relaxation must be in (0, 1].")
        if not 0.0 < self.temperature_relaxation <= 1.0:
            raise ValueError("temperature_relaxation must be in (0, 1].")
        if not 0.0 < self.turbulence_relaxation <= 1.0:
            raise ValueError("turbulence_relaxation must be in (0, 1].")
        if self.gamma <= 1.0:
            raise ValueError("gamma must be > 1.")
        if self.gas_constant <= 0.0:
            raise ValueError("gas_constant must be positive.")
        if self.prandtl <= 0.0 or self.turbulent_prandtl <= 0.0:
            raise ValueError("Prandtl numbers must be positive.")
        if not 0.0 < self.pressure_sor_omega < 2.0:
            raise ValueError("pressure_sor_omega must be in (0, 2).")
        if self.pressure_correction_tolerance <= 0.0:
            raise ValueError("pressure_correction_tolerance must be positive.")
        if self.pressure_pcg_max_iterations < 1:
            raise ValueError("pressure_pcg_max_iterations must be >= 1.")
        if self.pressure_velocity_coupling not in SUPPORTED_PRESSURE_COUPLING:
            raise ValueError(f"Unsupported pressure_velocity_coupling '{self.pressure_velocity_coupling}'.")
        if self.turbulence_model not in SUPPORTED_TURBULENCE_MODELS:
            raise ValueError(f"Unsupported turbulence_model '{self.turbulence_model}'.")
        if self.convection_scheme not in SUPPORTED_CONVECTION_SCHEMES:
            raise ValueError(f"Unsupported convection_scheme '{self.convection_scheme}'.")
        if self.pressure_linear_solver not in SUPPORTED_PRESSURE_LINEAR_SOLVERS:
            raise ValueError(f"Unsupported pressure_linear_solver '{self.pressure_linear_solver}'.")
        if self.outlet_static_pressure is not None and self.outlet_static_pressure <= 0.0:
            raise ValueError("outlet_static_pressure must be positive when provided.")
        if self.wall_temperature is not None and self.wall_temperature <= 0.0:
            raise ValueError("wall_temperature must be positive when provided.")
        if self.min_density <= 0.0 or self.min_pressure <= 0.0 or self.min_temperature <= 0.0:
            raise ValueError("Minimum thermodynamic limits must be positive.")
        if self.min_turbulent_kinetic_energy <= 0.0 or self.min_specific_dissipation <= 0.0:
            raise ValueError("Minimum turbulence limits must be positive.")
        if self.convergence_tolerance <= 0.0:
            raise ValueError("convergence_tolerance must be positive.")
        if self.force_coefficient_tolerance <= 0.0:
            raise ValueError("force_coefficient_tolerance must be positive.")
        if self.convergence_window < 1:
            raise ValueError("convergence_window must be >= 1.")


@dataclass
class PressureBasedState:
    density: np.ndarray
    u: np.ndarray
    v: np.ndarray
    pressure: np.ndarray
    temperature: np.ndarray
    turbulent_kinetic_energy: np.ndarray
    specific_dissipation: np.ndarray
    turbulent_viscosity: np.ndarray
    wall_distance: np.ndarray
    fluid: np.ndarray
    fluid_fraction: np.ndarray
    aperture_x: np.ndarray
    aperture_y: np.ndarray
    solid_levelset: np.ndarray
    laminar_viscosity: float
    dt: float = 0.0

    def field_dict(self, grid: GridSpec, *, device: Optional[str] = None) -> dict[str, CenteredField]:
        fields = {
            "density": self.density,
            "u": self.u,
            "v": self.v,
            "pressure": self.pressure,
            "temperature": self.temperature,
            "k": self.turbulent_kinetic_energy,
            "omega": self.specific_dissipation,
            "mu_t": self.turbulent_viscosity,
        }
        return {
            name: CenteredField.from_numpy(grid, values.astype(np.float32, copy=False), device=device)
            for name, values in fields.items()
        }


@dataclass(frozen=True)
class ResidualSnapshot:
    step: int
    dt: float
    mass: float
    momentum: float
    energy: float
    turbulence: float
    pressure_correction: float
    normalized_mass: float
    normalized_momentum: float
    normalized_energy: float
    normalized_turbulence: float
    drag_coefficient: float
    lift_coefficient: float
    moment_coefficient: float
    force_delta: float


@dataclass(frozen=True)
class AerodynamicCoefficients:
    drag: float
    lift: float
    moment: float


def _clip_positive(values: np.ndarray, floor: float) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=np.float64), float(floor))


def _smooth_heaviside(signed_distance: np.ndarray, epsilon: float) -> np.ndarray:
    phi = np.asarray(signed_distance, dtype=np.float64)
    eps = max(float(epsilon), 1.0e-12)
    result = np.empty_like(phi)
    result[phi <= -eps] = 0.0
    result[phi >= eps] = 1.0
    band = np.abs(phi) < eps
    band_phi = phi[band]
    result[band] = 0.5 + 0.5 * band_phi / eps + 0.5 / math.pi * np.sin(math.pi * band_phi / eps)
    return np.clip(result, 0.0, 1.0)


def _face_aperture_x(fluid_fraction: np.ndarray, active: np.ndarray) -> np.ndarray:
    face = np.zeros((fluid_fraction.shape[0] + 1, fluid_fraction.shape[1]), dtype=np.float64)
    face[0, :] = np.where(active[0, :], fluid_fraction[0, :], 0.0)
    face[-1, :] = np.where(active[-1, :], fluid_fraction[-1, :], 0.0)
    interior = active[:-1, :] & active[1:, :]
    face[1:-1, :] = np.where(interior, 0.5 * (fluid_fraction[:-1, :] + fluid_fraction[1:, :]), 0.0)
    return np.clip(face, 0.0, 1.0)


def _face_aperture_y(fluid_fraction: np.ndarray, active: np.ndarray) -> np.ndarray:
    face = np.zeros((fluid_fraction.shape[0], fluid_fraction.shape[1] + 1), dtype=np.float64)
    face[:, 0] = np.where(active[:, 0], fluid_fraction[:, 0], 0.0)
    face[:, -1] = np.where(active[:, -1], fluid_fraction[:, -1], 0.0)
    interior = active[:, :-1] & active[:, 1:]
    face[:, 1:-1] = np.where(interior, 0.5 * (fluid_fraction[:, :-1] + fluid_fraction[:, 1:]), 0.0)
    return np.clip(face, 0.0, 1.0)


def _freestream_velocity(freestream: FreestreamCondition, config: PressureBasedSolverConfig) -> tuple[float, float, float]:
    sound_speed = math.sqrt(config.gamma * config.gas_constant * freestream.static_temperature)
    speed = freestream.mach * sound_speed
    angle = math.radians(freestream.angle_of_attack_deg)
    return speed * math.cos(angle), speed * math.sin(angle), sound_speed


def _boundary_mask_x(fluid: np.ndarray) -> np.ndarray:
    face = np.zeros((fluid.shape[0] + 1, fluid.shape[1]), dtype=np.float64)
    face[0, :] = fluid[0, :]
    face[-1, :] = fluid[-1, :]
    face[1:-1, :] = fluid[:-1, :] & fluid[1:, :]
    return face


def _boundary_mask_y(fluid: np.ndarray) -> np.ndarray:
    face = np.zeros((fluid.shape[0], fluid.shape[1] + 1), dtype=np.float64)
    face[:, 0] = fluid[:, 0]
    face[:, -1] = fluid[:, -1]
    face[:, 1:-1] = fluid[:, :-1] & fluid[:, 1:]
    return face


def _fill_outer_boundary(field: np.ndarray, value: float) -> None:
    field[0, :] = value
    field[-1, :] = value
    field[:, 0] = value
    field[:, -1] = value


def _copy_east_boundary_from_neighbor(field: np.ndarray) -> None:
    field[-1, :] = field[-2, :]


def _reference_point(chord: float, leading_edge_x: float, leading_edge_y: float) -> tuple[float, float]:
    return leading_edge_x + 0.25 * chord, leading_edge_y


def _dynamic_pressure(freestream: FreestreamCondition, config: PressureBasedSolverConfig) -> float:
    u_inf, v_inf, _ = _freestream_velocity(freestream, config)
    rho_inf = freestream.static_pressure / (config.gas_constant * freestream.static_temperature)
    return 0.5 * rho_inf * (u_inf * u_inf + v_inf * v_inf)


def _extend_scalar_into_solids(field: np.ndarray, fluid: np.ndarray, passes: int = 4) -> np.ndarray:
    filled = field.astype(np.float64, copy=True)
    if np.all(fluid):
        return filled
    for _ in range(passes):
        accum = np.zeros_like(filled)
        count = np.zeros_like(filled)
        accum[1:, :] += filled[:-1, :] * fluid[:-1, :]
        count[1:, :] += fluid[:-1, :]
        accum[:-1, :] += filled[1:, :] * fluid[1:, :]
        count[:-1, :] += fluid[1:, :]
        accum[:, 1:] += filled[:, :-1] * fluid[:, :-1]
        count[:, 1:] += fluid[:, :-1]
        accum[:, :-1] += filled[:, 1:] * fluid[:, 1:]
        count[:, :-1] += fluid[:, 1:]
        update = np.where(count > 0.0, accum / np.maximum(count, 1.0), filled)
        filled = np.where(fluid, filled, update)
    return filled


def _gradient_x(field: np.ndarray, dx: float) -> np.ndarray:
    grad = np.empty_like(field, dtype=np.float64)
    grad[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2.0 * dx)
    grad[0, :] = (field[1, :] - field[0, :]) / dx
    grad[-1, :] = (field[-1, :] - field[-2, :]) / dx
    return grad


def _gradient_y(field: np.ndarray, dy: float) -> np.ndarray:
    grad = np.empty_like(field, dtype=np.float64)
    grad[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2.0 * dy)
    grad[:, 0] = (field[:, 1] - field[:, 0]) / dy
    grad[:, -1] = (field[:, -1] - field[:, -2]) / dy
    return grad


def _face_average_x(field: np.ndarray) -> np.ndarray:
    face = np.empty((field.shape[0] + 1, field.shape[1]), dtype=np.float64)
    face[0, :] = field[0, :]
    face[-1, :] = field[-1, :]
    face[1:-1, :] = 0.5 * (field[:-1, :] + field[1:, :])
    return face


def _face_average_y(field: np.ndarray) -> np.ndarray:
    face = np.empty((field.shape[0], field.shape[1] + 1), dtype=np.float64)
    face[:, 0] = field[:, 0]
    face[:, -1] = field[:, -1]
    face[:, 1:-1] = 0.5 * (field[:, :-1] + field[:, 1:])
    return face


def _cell_to_face_velocities(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u_face = _face_average_x(u)
    v_face = _face_average_y(v)
    return u_face, v_face


def _mass_fluxes_from_velocity(
    density: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    fluid: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    face_x_mask = _boundary_mask_x(fluid)
    face_y_mask = _boundary_mask_y(fluid)
    rho_face_x = _face_average_x(density) * face_x_mask
    rho_face_y = _face_average_y(density) * face_y_mask
    u_face, v_face = _cell_to_face_velocities(u, v)
    mass_flux_x = rho_face_x * u_face * dy
    mass_flux_y = rho_face_y * v_face * dx
    return mass_flux_x, mass_flux_y, rho_face_x, rho_face_y


def _mass_divergence_from_fluxes(mass_flux_x: np.ndarray, mass_flux_y: np.ndarray, dx: float, dy: float) -> np.ndarray:
    cell_volume = dx * dy
    return ((mass_flux_x[1:, :] - mass_flux_x[:-1, :]) + (mass_flux_y[:, 1:] - mass_flux_y[:, :-1])) / cell_volume


def _face_velocities_from_mass_fluxes(
    mass_flux_x: np.ndarray,
    mass_flux_y: np.ndarray,
    rho_face_x: np.ndarray,
    rho_face_y: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    u_face = np.zeros_like(mass_flux_x)
    v_face = np.zeros_like(mass_flux_y)
    np.divide(mass_flux_x, rho_face_x * dy, out=u_face, where=np.abs(rho_face_x) > 1.0e-12)
    np.divide(mass_flux_y, rho_face_y * dx, out=v_face, where=np.abs(rho_face_y) > 1.0e-12)
    return u_face, v_face


def _cell_velocities_from_faces(u_face: np.ndarray, v_face: np.ndarray, fluid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = 0.5 * (u_face[:-1, :] + u_face[1:, :])
    v = 0.5 * (v_face[:, :-1] + v_face[:, 1:])
    return np.where(fluid, u, 0.0), np.where(fluid, v, 0.0)


def _second_order_upwind_x(field: np.ndarray, face_velocity: np.ndarray, dx: float) -> np.ndarray:
    gradx = _gradient_x(field, dx)
    face = np.empty((field.shape[0] + 1, field.shape[1]), dtype=np.float64)
    face[0, :] = field[0, :]
    face[-1, :] = field[-1, :]
    left = field[:-1, :] + 0.5 * dx * gradx[:-1, :]
    right = field[1:, :] - 0.5 * dx * gradx[1:, :]
    face[1:-1, :] = np.where(face_velocity[1:-1, :] >= 0.0, left, right)
    return face


def _second_order_upwind_y(field: np.ndarray, face_velocity: np.ndarray, dy: float) -> np.ndarray:
    grady = _gradient_y(field, dy)
    face = np.empty((field.shape[0], field.shape[1] + 1), dtype=np.float64)
    face[:, 0] = field[:, 0]
    face[:, -1] = field[:, -1]
    bottom = field[:, :-1] + 0.5 * dy * grady[:, :-1]
    top = field[:, 1:] - 0.5 * dy * grady[:, 1:]
    face[:, 1:-1] = np.where(face_velocity[:, 1:-1] >= 0.0, bottom, top)
    return face


def _convection(field: np.ndarray, density: np.ndarray, u: np.ndarray, v: np.ndarray, fluid: np.ndarray, dx: float, dy: float) -> np.ndarray:
    face_x_mask = _boundary_mask_x(fluid)
    face_y_mask = _boundary_mask_y(fluid)
    u_face = _face_average_x(u) * face_x_mask
    v_face = _face_average_y(v) * face_y_mask
    rho_face_x = _face_average_x(density) * face_x_mask
    rho_face_y = _face_average_y(density) * face_y_mask
    phi_x = _second_order_upwind_x(field, u_face, dx) * face_x_mask
    phi_y = _second_order_upwind_y(field, v_face, dy) * face_y_mask
    flux_x = rho_face_x * u_face * phi_x
    flux_y = rho_face_y * v_face * phi_y
    return (flux_x[1:, :] - flux_x[:-1, :]) / dx + (flux_y[:, 1:] - flux_y[:, :-1]) / dy


def _diffusion(field: np.ndarray, diffusivity: np.ndarray, fluid: np.ndarray, dx: float, dy: float) -> np.ndarray:
    face_x_mask = _boundary_mask_x(fluid)
    face_y_mask = _boundary_mask_y(fluid)
    gamma_x = _face_average_x(diffusivity) * face_x_mask
    gamma_y = _face_average_y(diffusivity) * face_y_mask
    grad_x = np.zeros((field.shape[0] + 1, field.shape[1]), dtype=np.float64)
    grad_y = np.zeros((field.shape[0], field.shape[1] + 1), dtype=np.float64)
    grad_x[1:-1, :] = (field[1:, :] - field[:-1, :]) / dx
    grad_y[:, 1:-1] = (field[:, 1:] - field[:, :-1]) / dy
    flux_x = gamma_x * grad_x
    flux_y = gamma_y * grad_y
    return (flux_x[1:, :] - flux_x[:-1, :]) / dx + (flux_y[:, 1:] - flux_y[:, :-1]) / dy


def _velocity_divergence(u: np.ndarray, v: np.ndarray, fluid: np.ndarray, dx: float, dy: float) -> np.ndarray:
    face_x_mask = _boundary_mask_x(fluid)
    face_y_mask = _boundary_mask_y(fluid)
    flux_x = _face_average_x(u) * face_x_mask
    flux_y = _face_average_y(v) * face_y_mask
    return (flux_x[1:, :] - flux_x[:-1, :]) / dx + (flux_y[:, 1:] - flux_y[:, :-1]) / dy


def _mass_fluxes(density: np.ndarray, u: np.ndarray, v: np.ndarray, fluid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    face_x_mask = _boundary_mask_x(fluid)
    face_y_mask = _boundary_mask_y(fluid)
    flux_x = _face_average_x(density) * _face_average_x(u) * face_x_mask
    flux_y = _face_average_y(density) * _face_average_y(v) * face_y_mask
    return flux_x, flux_y


def _mass_divergence(density: np.ndarray, u: np.ndarray, v: np.ndarray, fluid: np.ndarray, dx: float, dy: float) -> np.ndarray:
    flux_x, flux_y = _mass_fluxes(density, u, v, fluid)
    return (flux_x[1:, :] - flux_x[:-1, :]) / dx + (flux_y[:, 1:] - flux_y[:, :-1]) / dy


def _compute_time_step(
    density: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    temperature: np.ndarray,
    fluid: np.ndarray,
    grid: GridSpec,
    config: PressureBasedSolverConfig,
) -> float:
    sound = np.sqrt(config.gamma * config.gas_constant * temperature)
    wave_speed = np.abs(u) + np.abs(v) + sound
    active = wave_speed[fluid]
    max_speed = float(np.max(active)) if active.size else 1.0
    return config.cfl * min(grid.dx, grid.dy) / max(max_speed, 1.0)


def _sst_blending(
    density: np.ndarray,
    k: np.ndarray,
    omega: np.ndarray,
    laminar_viscosity: float,
    wall_distance: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    beta_star = 0.09
    sigma_w2 = 0.856
    rho = _clip_positive(density, 1.0e-12)
    omega_safe = _clip_positive(omega, 1.0e-12)
    k_safe = _clip_positive(k, 1.0e-12)
    wall = _clip_positive(wall_distance, 0.25 * min(dx, dy))
    nu = laminar_viscosity / rho
    grad_kx = _gradient_x(k_safe, dx)
    grad_ky = _gradient_y(k_safe, dy)
    grad_wx = _gradient_x(omega_safe, dx)
    grad_wy = _gradient_y(omega_safe, dy)
    grad_dot = grad_kx * grad_wx + grad_ky * grad_wy
    cd_kw = np.maximum(2.0 * rho * sigma_w2 * grad_dot / omega_safe, 1.0e-20)
    term_a = np.sqrt(k_safe) / (beta_star * omega_safe * wall)
    term_b = 500.0 * nu / (wall * wall * omega_safe)
    term_c = 4.0 * rho * sigma_w2 * k_safe / (cd_kw * wall * wall)
    arg1 = np.minimum(np.maximum(term_a, term_b), term_c)
    f1 = np.tanh(arg1**4)
    arg2 = np.maximum(2.0 * np.sqrt(k_safe) / (beta_star * omega_safe * wall), term_b)
    f2 = np.tanh(arg2**2)
    sigma_k = f1 * 0.85 + (1.0 - f1) * 1.0
    sigma_w = f1 * 0.5 + (1.0 - f1) * 0.856
    alpha = f1 * (5.0 / 9.0) + (1.0 - f1) * 0.44
    beta = f1 * 0.075 + (1.0 - f1) * 0.0828
    return f1, f2, sigma_k, sigma_w, alpha, beta


def _compute_turbulent_viscosity(
    density: np.ndarray,
    k: np.ndarray,
    omega: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    fluid: np.ndarray,
    dx: float,
    dy: float,
    wall_distance: np.ndarray,
    laminar_viscosity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f1, f2, sigma_k, sigma_w, alpha, beta = _sst_blending(
        density,
        k,
        omega,
        laminar_viscosity,
        wall_distance,
        dx,
        dy,
    )
    du_dx = _gradient_x(u, dx)
    du_dy = _gradient_y(u, dy)
    dv_dx = _gradient_x(v, dx)
    dv_dy = _gradient_y(v, dy)
    strain = np.sqrt(
        np.maximum(
            2.0 * du_dx * du_dx + 2.0 * dv_dy * dv_dy + (du_dy + dv_dx) * (du_dy + dv_dx),
            0.0,
        )
    )
    a1 = 0.31
    omega_safe = _clip_positive(omega, 1.0e-12)
    k_safe = _clip_positive(k, 1.0e-12)
    limiter = np.maximum(a1 * omega_safe, strain * f2)
    mu_t = density * a1 * k_safe / np.maximum(limiter, 1.0e-12)
    mu_t = np.where(fluid, np.maximum(mu_t, 0.0), 0.0)
    return mu_t, sigma_k, sigma_w, alpha, beta, strain


def _compute_aerodynamic_coefficients(
    state: PressureBasedState,
    grid: GridSpec,
    freestream: FreestreamCondition,
    config: PressureBasedSolverConfig,
    *,
    chord: float,
) -> AerodynamicCoefficients:
    solid = ~state.fluid
    mu_eff = state.laminar_viscosity + state.turbulent_viscosity
    pressure = state.pressure
    q_inf = max(_dynamic_pressure(freestream, config), 1.0e-6)
    alpha = math.radians(freestream.angle_of_attack_deg)
    drag_dir = np.array([math.cos(alpha), math.sin(alpha)], dtype=np.float64)
    lift_dir = np.array([-math.sin(alpha), math.cos(alpha)], dtype=np.float64)
    solid_i, solid_j = np.nonzero(solid)
    if solid_i.size == 0:
        return AerodynamicCoefficients(drag=0.0, lift=0.0, moment=0.0)
    leading_edge_x = float(np.min(grid.x0 + (solid_i.astype(np.float64) + 0.5) * grid.dx))
    leading_edge_y = float(np.mean(grid.y0 + (solid_j.astype(np.float64) + 0.5) * grid.dy))
    ref_x, ref_y = _reference_point(chord, leading_edge_x, leading_edge_y)

    fx = 0.0
    fy = 0.0
    moment = 0.0

    def accumulate(force_x: np.ndarray, force_y: np.ndarray, x_face: np.ndarray, y_face: np.ndarray) -> None:
        nonlocal fx, fy, moment
        if force_x.size == 0:
            return
        fx += float(np.sum(force_x))
        fy += float(np.sum(force_y))
        moment += float(np.sum((x_face - ref_x) * force_y - (y_face - ref_y) * force_x))

    east = state.fluid[:-1, :] & solid[1:, :]
    if np.any(east):
        p = pressure[:-1, :][east]
        mu = mu_eff[:-1, :][east]
        tangential = state.v[:-1, :][east]
        shear = -mu * tangential / (0.5 * grid.dx)
        force_x = p * grid.dy * np.ones_like(p)
        force_y = shear * grid.dy
        x_face = np.full_like(p, grid.x0) + (np.nonzero(east)[0] + 1.0) * grid.dx
        y_face = grid.y0 + (np.nonzero(east)[1] + 0.5) * grid.dy
        accumulate(force_x, force_y, x_face, y_face)

    west = state.fluid[1:, :] & solid[:-1, :]
    if np.any(west):
        p = pressure[1:, :][west]
        mu = mu_eff[1:, :][west]
        tangential = state.v[1:, :][west]
        shear = -mu * tangential / (0.5 * grid.dx)
        force_x = -p * grid.dy * np.ones_like(p)
        force_y = shear * grid.dy
        x_face = np.full_like(p, grid.x0) + np.nonzero(west)[0] * grid.dx
        y_face = grid.y0 + (np.nonzero(west)[1] + 0.5) * grid.dy
        accumulate(force_x, force_y, x_face, y_face)

    north = state.fluid[:, :-1] & solid[:, 1:]
    if np.any(north):
        p = pressure[:, :-1][north]
        mu = mu_eff[:, :-1][north]
        tangential = state.u[:, :-1][north]
        shear = -mu * tangential / (0.5 * grid.dy)
        force_x = shear * grid.dx
        force_y = p * grid.dx * np.ones_like(p)
        x_face = grid.x0 + (np.nonzero(north)[0] + 0.5) * grid.dx
        y_face = np.full_like(p, grid.y0) + (np.nonzero(north)[1] + 1.0) * grid.dy
        accumulate(force_x, force_y, x_face, y_face)

    south = state.fluid[:, 1:] & solid[:, :-1]
    if np.any(south):
        p = pressure[:, 1:][south]
        mu = mu_eff[:, 1:][south]
        tangential = state.u[:, 1:][south]
        shear = -mu * tangential / (0.5 * grid.dy)
        force_x = shear * grid.dx
        force_y = -p * grid.dx * np.ones_like(p)
        x_face = grid.x0 + (np.nonzero(south)[0] + 0.5) * grid.dx
        y_face = np.full_like(p, grid.y0) + np.nonzero(south)[1] * grid.dy
        accumulate(force_x, force_y, x_face, y_face)

    drag = (fx * drag_dir[0] + fy * drag_dir[1]) / (q_inf * chord)
    lift = (fx * lift_dir[0] + fy * lift_dir[1]) / (q_inf * chord)
    moment_coeff = moment / (q_inf * chord * chord)
    return AerodynamicCoefficients(drag=drag, lift=lift, moment=moment_coeff)


def _apply_boundaries(
    state: PressureBasedState,
    freestream: FreestreamCondition,
    config: PressureBasedSolverConfig,
    *,
    u_inf: float,
    v_inf: float,
    k_inf: float,
    omega_inf: float,
) -> None:
    rho_inf = freestream.static_pressure / (config.gas_constant * freestream.static_temperature)

    state.pressure[0, :] = freestream.static_pressure
    state.temperature[0, :] = freestream.static_temperature
    state.density[0, :] = rho_inf
    state.u[0, :] = u_inf
    state.v[0, :] = v_inf
    state.turbulent_kinetic_energy[0, :] = k_inf
    state.specific_dissipation[0, :] = omega_inf
    state.turbulent_viscosity[0, :] = 0.0

    outlet_pressure = config.outlet_static_pressure or freestream.static_pressure
    state.pressure[-1, :] = outlet_pressure
    _copy_east_boundary_from_neighbor(state.temperature)
    _copy_east_boundary_from_neighbor(state.density)
    _copy_east_boundary_from_neighbor(state.u)
    _copy_east_boundary_from_neighbor(state.v)
    _copy_east_boundary_from_neighbor(state.turbulent_kinetic_energy)
    _copy_east_boundary_from_neighbor(state.specific_dissipation)
    _copy_east_boundary_from_neighbor(state.turbulent_viscosity)

    state.pressure[:, 0] = freestream.static_pressure
    state.pressure[:, -1] = freestream.static_pressure
    state.temperature[:, 0] = freestream.static_temperature
    state.temperature[:, -1] = freestream.static_temperature
    state.density[:, 0] = rho_inf
    state.density[:, -1] = rho_inf
    state.u[:, 0] = u_inf
    state.u[:, -1] = u_inf
    state.v[:, 0] = v_inf
    state.v[:, -1] = v_inf
    state.turbulent_kinetic_energy[:, 0] = k_inf
    state.turbulent_kinetic_energy[:, -1] = k_inf
    state.specific_dissipation[:, 0] = omega_inf
    state.specific_dissipation[:, -1] = omega_inf
    state.turbulent_viscosity[:, 0] = 0.0
    state.turbulent_viscosity[:, -1] = 0.0

    wall_floor = max(float(np.min(state.wall_distance[state.fluid])) if np.any(state.fluid) else 0.0, 1.0e-6)
    wall_omega = 60.0 * state.laminar_viscosity / (
        np.maximum(state.density, config.min_density) * np.maximum(state.wall_distance * state.wall_distance, wall_floor * wall_floor)
    )
    wall_omega = np.maximum(wall_omega, omega_inf)
    solid = ~state.fluid
    state.u[solid] = 0.0
    state.v[solid] = 0.0
    state.turbulent_kinetic_energy[solid] = config.min_turbulent_kinetic_energy
    state.specific_dissipation[solid] = wall_omega[solid]
    state.turbulent_viscosity[solid] = 0.0
    state.pressure = _extend_scalar_into_solids(state.pressure, state.fluid)
    if config.wall_temperature is None:
        state.temperature = _extend_scalar_into_solids(state.temperature, state.fluid)
    else:
        state.temperature[solid] = config.wall_temperature
    state.density = _extend_scalar_into_solids(state.density, state.fluid)
    state.pressure = _clip_positive(state.pressure, config.min_pressure)
    state.temperature = _clip_positive(state.temperature, config.min_temperature)
    state.density = _clip_positive(state.density, config.min_density)
    state.turbulent_kinetic_energy = _clip_positive(
        state.turbulent_kinetic_energy,
        config.min_turbulent_kinetic_energy,
    )
    state.specific_dissipation = _clip_positive(
        state.specific_dissipation,
        config.min_specific_dissipation,
    )


def initialize_pressure_based_state(
    grid: GridSpec,
    solid_levelset: np.ndarray,
    *,
    chord: float,
    freestream: FreestreamCondition = FreestreamCondition(),
    config: PressureBasedSolverConfig = PressureBasedSolverConfig(),
) -> PressureBasedState:
    if grid.is_3d:
        raise NotImplementedError("The experimental pressure-based solver currently supports 2D grids only.")
    solid_sdf = np.asarray(solid_levelset, dtype=np.float64)
    if solid_sdf.shape != grid.shape:
        raise ValueError(f"solid_levelset has shape {solid_sdf.shape}, expected {grid.shape}.")
    h = min(grid.dx, grid.dy)
    fluid_fraction = _smooth_heaviside(solid_sdf, 0.75 * h)
    fluid = fluid_fraction > 0.05
    if not np.any(fluid):
        raise ValueError("The supplied levelset leaves no fluid cells.")
    aperture_x = _face_aperture_x(fluid_fraction, fluid)
    aperture_y = _face_aperture_y(fluid_fraction, fluid)
    u_inf, v_inf, _ = _freestream_velocity(freestream, config)
    rho_inf = freestream.static_pressure / (config.gas_constant * freestream.static_temperature)
    laminar_viscosity = rho_inf * math.sqrt(u_inf * u_inf + v_inf * v_inf) * chord / freestream.reynolds_number
    k_inf = max(1.5 * (math.sqrt(u_inf * u_inf + v_inf * v_inf) * freestream.turbulence_intensity) ** 2, config.min_turbulent_kinetic_energy)
    omega_inf = max(
        rho_inf * k_inf / (laminar_viscosity * freestream.turbulent_viscosity_ratio),
        config.min_specific_dissipation,
    )
    state = PressureBasedState(
        density=np.full(grid.shape, rho_inf, dtype=np.float64),
        u=np.full(grid.shape, u_inf, dtype=np.float64),
        v=np.full(grid.shape, v_inf, dtype=np.float64),
        pressure=np.full(grid.shape, freestream.static_pressure, dtype=np.float64),
        temperature=np.full(grid.shape, freestream.static_temperature, dtype=np.float64),
        turbulent_kinetic_energy=np.full(grid.shape, k_inf, dtype=np.float64),
        specific_dissipation=np.full(grid.shape, omega_inf, dtype=np.float64),
        turbulent_viscosity=np.zeros(grid.shape, dtype=np.float64),
        wall_distance=np.maximum(np.abs(solid_sdf), 1.0e-6 * h),
        fluid=fluid,
        fluid_fraction=fluid_fraction,
        aperture_x=aperture_x,
        aperture_y=aperture_y,
        solid_levelset=solid_sdf,
        laminar_viscosity=laminar_viscosity,
    )
    _apply_boundaries(state, freestream, config, u_inf=u_inf, v_inf=v_inf, k_inf=k_inf, omega_inf=omega_inf)
    return state


def _pressure_operator(
    field: np.ndarray,
    beta_x: np.ndarray,
    beta_y: np.ndarray,
    fluid: np.ndarray,
) -> np.ndarray:
    result = np.zeros_like(field, dtype=np.float64)
    nx, ny = field.shape
    result[1:-1, 1:-1] = (
        beta_x[2:nx, 1:-1] * (field[2:, 1:-1] - field[1:-1, 1:-1])
        - beta_x[1 : nx - 1, 1:-1] * (field[1:-1, 1:-1] - field[:-2, 1:-1])
        + beta_y[1:-1, 2:ny] * (field[1:-1, 2:] - field[1:-1, 1:-1])
        - beta_y[1:-1, 1 : ny - 1] * (field[1:-1, 1:-1] - field[1:-1, :-2])
    )
    return np.where(fluid, result, 0.0)


def _pressure_diagonal(
    beta_x: np.ndarray,
    beta_y: np.ndarray,
    fluid: np.ndarray,
) -> np.ndarray:
    diagonal = np.zeros(fluid.shape, dtype=np.float64)
    nx, ny = fluid.shape
    diagonal[1:-1, 1:-1] = (
        beta_x[2:nx, 1:-1] * fluid[2:, 1:-1]
        + beta_x[1 : nx - 1, 1:-1] * fluid[:-2, 1:-1]
        + beta_y[1:-1, 2:ny] * fluid[1:-1, 2:]
        + beta_y[1:-1, 1 : ny - 1] * fluid[1:-1, :-2]
    )
    return np.where(fluid, np.maximum(diagonal, 1.0e-12), 1.0)


def _pressure_correction_pcg(
    rhs: np.ndarray,
    beta_x: np.ndarray,
    beta_y: np.ndarray,
    fluid: np.ndarray,
    diagonal: np.ndarray,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, float]:
    x = np.zeros_like(rhs, dtype=np.float64)
    r = np.where(fluid, rhs - _pressure_operator(x, beta_x, beta_y, fluid), 0.0)
    residual = float(np.max(np.abs(r[fluid]))) if np.any(fluid) else 0.0
    if residual < tolerance:
        return x, residual
    z = np.where(fluid, r / diagonal, 0.0)
    p = z.copy()
    rz_old = float(np.sum(r[fluid] * z[fluid]))
    for _ in range(max_iterations):
        ap = _pressure_operator(p, beta_x, beta_y, fluid)
        denom = float(np.sum(p[fluid] * ap[fluid]))
        if abs(denom) <= 1.0e-20:
            break
        alpha = rz_old / denom
        x = np.where(fluid, x + alpha * p, 0.0)
        r = np.where(fluid, r - alpha * ap, 0.0)
        residual = float(np.max(np.abs(r[fluid]))) if np.any(fluid) else 0.0
        if residual < tolerance:
            break
        z = np.where(fluid, r / diagonal, 0.0)
        rz_new = float(np.sum(r[fluid] * z[fluid]))
        if abs(rz_old) <= 1.0e-20:
            break
        beta = rz_new / rz_old
        p = np.where(fluid, z + beta * p, 0.0)
        rz_old = rz_new
    return x, residual


def _pressure_correction_sor(
    divergence_star: np.ndarray,
    beta_x: np.ndarray,
    beta_y: np.ndarray,
    fluid: np.ndarray,
    iterations: int,
    sor_omega: float,
    tolerance: float,
) -> tuple[np.ndarray, float]:
    pcorr = np.zeros_like(divergence_star, dtype=np.float64)
    pressure_residual = 0.0
    for _ in range(iterations):
        max_change = 0.0
        for color in (0, 1):
            for i in range(1, divergence_star.shape[0] - 1):
                j_start = 1 + ((i + color) % 2)
                for j in range(j_start, divergence_star.shape[1] - 1, 2):
                    if not fluid[i, j]:
                        pcorr[i, j] = 0.0
                        continue
                    ae = beta_x[i + 1, j] if fluid[i + 1, j] else 0.0
                    aw = beta_x[i, j] if fluid[i - 1, j] else 0.0
                    an = beta_y[i, j + 1] if fluid[i, j + 1] else 0.0
                    ass = beta_y[i, j] if fluid[i, j - 1] else 0.0
                    ap = ae + aw + an + ass
                    if ap <= 1.0e-12:
                        pcorr[i, j] = 0.0
                        continue
                    rhs = divergence_star[i, j]
                    sor_target = (ae * pcorr[i + 1, j] + aw * pcorr[i - 1, j] + an * pcorr[i, j + 1] + ass * pcorr[i, j - 1] + rhs) / ap
                    updated = (1.0 - sor_omega) * pcorr[i, j] + sor_omega * sor_target
                    max_change = max(max_change, abs(updated - pcorr[i, j]))
                    pcorr[i, j] = updated
        residual = np.where(fluid, divergence_star - _pressure_operator(pcorr, beta_x, beta_y, fluid), 0.0)
        pressure_residual = float(np.max(np.abs(residual[fluid]))) if np.any(fluid) else 0.0
        if pressure_residual < tolerance or max_change < 1.0e-8:
            break
    return pcorr, pressure_residual


def _pressure_correction(
    divergence_star: np.ndarray,
    density: np.ndarray,
    fluid: np.ndarray,
    dt: float,
    dx: float,
    dy: float,
    config: PressureBasedSolverConfig,
) -> tuple[np.ndarray, float]:
    rho_face_x = _face_average_x(np.maximum(density, 1.0e-12))
    rho_face_y = _face_average_y(np.maximum(density, 1.0e-12))
    beta_x = dt / np.maximum(rho_face_x * dx * dx, 1.0e-12)
    beta_y = dt / np.maximum(rho_face_y * dy * dy, 1.0e-12)
    if config.pressure_linear_solver == "SOR":
        return _pressure_correction_sor(
            divergence_star,
            beta_x,
            beta_y,
            fluid,
            config.pressure_correction_iterations,
            config.pressure_sor_omega,
            config.pressure_correction_tolerance,
        )
    diagonal = _pressure_diagonal(beta_x, beta_y, fluid)
    return _pressure_correction_pcg(
        divergence_star,
        beta_x,
        beta_y,
        fluid,
        diagonal,
        config.pressure_correction_tolerance,
        config.pressure_pcg_max_iterations,
    )


def _update_temperature(
    state: PressureBasedState,
    grid: GridSpec,
    config: PressureBasedSolverConfig,
) -> np.ndarray:
    cp = config.gamma * config.gas_constant / (config.gamma - 1.0)
    mu_eff = state.laminar_viscosity + state.turbulent_viscosity
    alpha_eff = state.laminar_viscosity / (np.maximum(state.density, config.min_density) * config.prandtl) + state.turbulent_viscosity / (
        np.maximum(state.density, config.min_density) * config.turbulent_prandtl
    )
    convection = _convection(state.temperature, state.density, state.u, state.v, state.fluid, grid.dx, grid.dy)
    diffusion = _diffusion(state.temperature, np.maximum(alpha_eff, 0.0), state.fluid, grid.dx, grid.dy)
    divergence = _velocity_divergence(state.u, state.v, state.fluid, grid.dx, grid.dy)
    rhs = -convection / np.maximum(state.density, config.min_density) + diffusion - (config.gamma - 1.0) * state.temperature * divergence
    updated = state.temperature + state.dt * rhs / cp
    relaxed = (1.0 - config.temperature_relaxation) * state.temperature + config.temperature_relaxation * updated
    return _clip_positive(relaxed, config.min_temperature)


def _simple_update(
    state: PressureBasedState,
    grid: GridSpec,
    freestream: FreestreamCondition,
    config: PressureBasedSolverConfig,
    *,
    u_inf: float,
    v_inf: float,
    k_inf: float,
    omega_inf: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mu_eff = state.laminar_viscosity + state.turbulent_viscosity
    rho = np.maximum(state.density, config.min_density)
    pressure_for_grad = _extend_scalar_into_solids(state.pressure, state.fluid)
    dpdx = _gradient_x(pressure_for_grad, grid.dx)
    dpdy = _gradient_y(pressure_for_grad, grid.dy)
    conv_u = _convection(state.u, rho, state.u, state.v, state.fluid, grid.dx, grid.dy)
    conv_v = _convection(state.v, rho, state.u, state.v, state.fluid, grid.dx, grid.dy)
    diff_u = _diffusion(state.u, mu_eff, state.fluid, grid.dx, grid.dy)
    diff_v = _diffusion(state.v, mu_eff, state.fluid, grid.dx, grid.dy)
    rhs_u = (-conv_u + diff_u - dpdx) / rho
    rhs_v = (-conv_v + diff_v - dpdy) / rho
    u_star = state.u + config.velocity_relaxation * state.dt * rhs_u
    v_star = state.v + config.velocity_relaxation * state.dt * rhs_v
    u_star = np.where(state.fluid, u_star, 0.0)
    v_star = np.where(state.fluid, v_star, 0.0)

    mass_flux_x_star, mass_flux_y_star, rho_face_x, rho_face_y = _mass_fluxes_from_velocity(
        rho,
        u_star,
        v_star,
        state.fluid,
        grid.dx,
        grid.dy,
    )
    divergence_star = _mass_divergence_from_fluxes(mass_flux_x_star, mass_flux_y_star, grid.dx, grid.dy)
    pcorr, pressure_residual = _pressure_correction(
        divergence_star,
        rho,
        state.fluid,
        state.dt,
        grid.dx,
        grid.dy,
        config,
    )
    mass_flux_x = mass_flux_x_star.copy()
    mass_flux_y = mass_flux_y_star.copy()
    mass_flux_x[1:-1, :] -= state.dt * grid.dy * (pcorr[1:, :] - pcorr[:-1, :]) / np.maximum(rho_face_x[1:-1, :], 1.0e-12) / grid.dx
    mass_flux_y[:, 1:-1] -= state.dt * grid.dx * (pcorr[:, 1:] - pcorr[:, :-1]) / np.maximum(rho_face_y[:, 1:-1], 1.0e-12) / grid.dy
    u_face, v_face = _face_velocities_from_mass_fluxes(mass_flux_x, mass_flux_y, rho_face_x, rho_face_y, grid.dx, grid.dy)
    u_new, v_new = _cell_velocities_from_faces(u_face, v_face, state.fluid)
    pressure_new = _clip_positive(state.pressure + config.pressure_relaxation * pcorr, config.min_pressure)
    state.pressure = np.where(state.fluid, pressure_new, state.pressure)
    state.u = np.where(state.fluid, u_new, 0.0)
    state.v = np.where(state.fluid, v_new, 0.0)
    state.density = _clip_positive(state.pressure / (config.gas_constant * state.temperature), config.min_density)
    _apply_boundaries(state, freestream, config, u_inf=u_inf, v_inf=v_inf, k_inf=k_inf, omega_inf=omega_inf)
    return divergence_star, u_star, v_star, pressure_residual


def _update_turbulence(
    state: PressureBasedState,
    grid: GridSpec,
    config: PressureBasedSolverConfig,
    freestream: FreestreamCondition,
    *,
    k_inf: float,
    omega_inf: float,
) -> None:
    mu_t, sigma_k, sigma_w, alpha, beta, strain = _compute_turbulent_viscosity(
        state.density,
        state.turbulent_kinetic_energy,
        state.specific_dissipation,
        state.u,
        state.v,
        state.fluid,
        grid.dx,
        grid.dy,
        state.wall_distance,
        state.laminar_viscosity,
    )
    rho = np.maximum(state.density, config.min_density)
    k = np.maximum(state.turbulent_kinetic_energy, config.min_turbulent_kinetic_energy)
    omega = np.maximum(state.specific_dissipation, config.min_specific_dissipation)
    production = np.minimum(mu_t * strain * strain, 10.0 * 0.09 * rho * k * omega)
    grad_kx = _gradient_x(k, grid.dx)
    grad_ky = _gradient_y(k, grid.dy)
    grad_wx = _gradient_x(omega, grid.dx)
    grad_wy = _gradient_y(omega, grid.dy)
    cross_diff = 2.0 * (1.0 - _sst_blending(rho, k, omega, state.laminar_viscosity, state.wall_distance, grid.dx, grid.dy)[0]) * 0.856 * (
        grad_kx * grad_wx + grad_ky * grad_wy
    ) / np.maximum(omega, config.min_specific_dissipation)
    k_diff = _diffusion(k, state.laminar_viscosity + sigma_k * mu_t, state.fluid, grid.dx, grid.dy)
    w_diff = _diffusion(omega, state.laminar_viscosity + sigma_w * mu_t, state.fluid, grid.dx, grid.dy)
    k_conv = _convection(k, rho, state.u, state.v, state.fluid, grid.dx, grid.dy)
    w_conv = _convection(omega, rho, state.u, state.v, state.fluid, grid.dx, grid.dy)
    k_rhs = -k_conv / rho + k_diff / rho + production / rho - 0.09 * k * omega
    w_rhs = -w_conv / rho + w_diff / rho + alpha * strain * strain - beta * omega * omega + cross_diff
    k_new = (1.0 - config.turbulence_relaxation) * k + config.turbulence_relaxation * (k + state.dt * k_rhs)
    w_new = (1.0 - config.turbulence_relaxation) * omega + config.turbulence_relaxation * (omega + state.dt * w_rhs)
    state.turbulent_kinetic_energy = np.where(
        state.fluid,
        _clip_positive(k_new, config.min_turbulent_kinetic_energy),
        config.min_turbulent_kinetic_energy,
    )
    state.specific_dissipation = np.where(
        state.fluid,
        _clip_positive(w_new, config.min_specific_dissipation),
        np.maximum(omega_inf, config.min_specific_dissipation),
    )
    state.turbulent_viscosity = mu_t


def run_pressure_based_solver(
    grid: GridSpec,
    solid_levelset: np.ndarray,
    *,
    chord: float,
    freestream: FreestreamCondition = FreestreamCondition(),
    config: PressureBasedSolverConfig = PressureBasedSolverConfig(),
) -> tuple[PressureBasedState, list[ResidualSnapshot]]:
    state = initialize_pressure_based_state(
        grid,
        solid_levelset,
        chord=chord,
        freestream=freestream,
        config=config,
    )
    u_inf, v_inf, _ = _freestream_velocity(freestream, config)
    speed_inf = math.sqrt(u_inf * u_inf + v_inf * v_inf)
    rho_inf = freestream.static_pressure / (config.gas_constant * freestream.static_temperature)
    k_inf = max(1.5 * (speed_inf * freestream.turbulence_intensity) ** 2, config.min_turbulent_kinetic_energy)
    omega_inf = max(
        rho_inf * k_inf / (state.laminar_viscosity * freestream.turbulent_viscosity_ratio),
        config.min_specific_dissipation,
    )

    history: list[ResidualSnapshot] = []
    residual_reference: Optional[tuple[float, float, float, float]] = None
    previous_forces: Optional[AerodynamicCoefficients] = None
    for step in range(1, config.pseudo_steps + 1):
        u_prev = state.u.copy()
        v_prev = state.v.copy()
        t_prev = state.temperature.copy()
        k_prev = state.turbulent_kinetic_energy.copy()
        omega_prev = state.specific_dissipation.copy()

        state.dt = _compute_time_step(state.density, state.u, state.v, state.temperature, state.fluid, grid, config)
        state.temperature = _update_temperature(state, grid, config)
        state.density = _clip_positive(state.pressure / (config.gas_constant * state.temperature), config.min_density)
        _apply_boundaries(state, freestream, config, u_inf=u_inf, v_inf=v_inf, k_inf=k_inf, omega_inf=omega_inf)
        state.turbulent_viscosity, _, _, _, _, _ = _compute_turbulent_viscosity(
            state.density,
            state.turbulent_kinetic_energy,
            state.specific_dissipation,
            state.u,
            state.v,
            state.fluid,
            grid.dx,
            grid.dy,
            state.wall_distance,
            state.laminar_viscosity,
        )

        divergence_star = np.zeros_like(state.pressure)
        pressure_residual = 0.0
        for _ in range(config.simple_iterations):
            divergence_star, _, _, pressure_residual = _simple_update(
                state,
                grid,
                freestream,
                config,
                u_inf=u_inf,
                v_inf=v_inf,
                k_inf=k_inf,
                omega_inf=omega_inf,
            )

        _update_turbulence(
            state,
            grid,
            config,
            freestream,
            k_inf=k_inf,
            omega_inf=omega_inf,
        )
        state.density = _clip_positive(state.pressure / (config.gas_constant * state.temperature), config.min_density)
        _apply_boundaries(state, freestream, config, u_inf=u_inf, v_inf=v_inf, k_inf=k_inf, omega_inf=omega_inf)

        mass_flux_x, mass_flux_y, _, _ = _mass_fluxes_from_velocity(
            state.density,
            state.u,
            state.v,
            state.fluid,
            grid.dx,
            grid.dy,
        )
        mass_residual = float(
            np.max(
                np.abs(
                    _mass_divergence_from_fluxes(mass_flux_x, mass_flux_y, grid.dx, grid.dy)[state.fluid]
                )
            )
        )
        momentum_residual = float(
            max(
                np.max(np.abs(state.u[state.fluid] - u_prev[state.fluid])),
                np.max(np.abs(state.v[state.fluid] - v_prev[state.fluid])),
            )
        )
        energy_residual = float(np.max(np.abs(state.temperature[state.fluid] - t_prev[state.fluid])))
        turbulence_residual = float(
            max(
                np.max(np.abs(state.turbulent_kinetic_energy[state.fluid] - k_prev[state.fluid])),
                np.max(np.abs(state.specific_dissipation[state.fluid] - omega_prev[state.fluid])),
            )
        )
        if residual_reference is None:
            residual_reference = (
                max(mass_residual, 1.0e-12),
                max(momentum_residual, 1.0e-12),
                max(energy_residual, 1.0e-12),
                max(turbulence_residual, 1.0e-12),
            )
        aero = _compute_aerodynamic_coefficients(state, grid, freestream, config, chord=chord)
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
        history.append(
            ResidualSnapshot(
                step=step,
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
        )
        recent = history[-config.convergence_window :]
        recent_force_delta = max(item.force_delta for item in recent)
        recent_residual = max(
            max(item.normalized_mass, item.normalized_momentum, item.normalized_energy, item.normalized_turbulence)
            for item in recent
        )
        recent_pressure = max(item.pressure_correction for item in recent)
        if (
            len(recent) >= config.convergence_window
            and recent_residual < config.convergence_tolerance
            and recent_force_delta < config.force_coefficient_tolerance
            and recent_pressure < config.pressure_correction_tolerance
        ):
            break

    return state, history


__all__ = [
    "AerodynamicCoefficients",
    "FreestreamCondition",
    "PressureBasedSolverConfig",
    "PressureBasedState",
    "ResidualSnapshot",
    "run_pressure_based_solver",
    "initialize_pressure_based_state",
]
