#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from warp_fluid.core import GridSpec
from warp_fluid.geom.levelset_grid import naca4_airfoil_levelset
from warp_fluid.solver.pressure_based import (
    FreestreamCondition,
    PressureBasedSolverConfig,
    run_pressure_based_solver,
)


DEFAULT_CONFIG_PATH = Path(__file__).with_name("configs") / "naca_pressure_based.yaml"


@dataclass(frozen=True)
class NACAPressureBasedConfig:
    nx: int = 256
    ny: int = 128
    lx: float = 12.0
    ly: float = 6.0
    airfoil_code: str = "2412"
    chord: float = 1.5
    leading_edge: tuple[float, float] = (3.0, 3.0)
    airfoil_angle_deg: float = 0.0
    mach: float = 0.3
    static_pressure: float = 101325.0
    static_temperature: float = 288.15
    angle_of_attack_deg: float = 4.0
    reynolds_number: float = 1.0e6
    turbulence_intensity: float = 0.01
    turbulent_viscosity_ratio: float = 10.0
    pseudo_steps: int = 250
    cfl: float = 1.2
    simple_iterations: int = 3
    pressure_correction_iterations: int = 40
    velocity_relaxation: float = 0.6
    pressure_relaxation: float = 0.25
    temperature_relaxation: float = 0.5
    turbulence_relaxation: float = 0.35
    pressure_velocity_coupling: str = "SIMPLE"
    turbulence_model: str = "sst_k_omega"
    convection_scheme: str = "second_order_upwind"
    pressure_linear_solver: str = "SOR"
    pressure_sor_omega: float = 1.6
    pressure_correction_tolerance: float = 1.0e-5
    pressure_pcg_max_iterations: int = 200
    outlet_static_pressure: Optional[float] = None
    wall_temperature: Optional[float] = None
    convergence_tolerance: float = 5.0e-4
    force_coefficient_tolerance: float = 1.0e-4
    convergence_window: int = 5
    output_npz: Optional[str] = "outputs/naca_pressure_based/final_state.npz"

    def __post_init__(self) -> None:
        if self.nx < 1 or self.ny < 1:
            raise ValueError("nx and ny must be >= 1.")
        if self.lx <= 0.0 or self.ly <= 0.0:
            raise ValueError("lx and ly must be positive.")
        if self.chord <= 0.0:
            raise ValueError("chord must be positive.")
        code = str(self.airfoil_code).strip()
        if len(code) != 4 or not code.isdigit():
            raise ValueError("airfoil_code must be a 4-digit NACA code such as '2412'.")


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level mapping in {path}.")
    return dict(data)


def _as_pair(name: str, values: Any) -> tuple[float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError(f"{name} must be a sequence of length 2.")
    return float(values[0]), float(values[1])


def config_from_mapping(data: Mapping[str, Any]) -> NACAPressureBasedConfig:
    values = dict(data)
    unknown = set(values) - {
        "nx",
        "ny",
        "lx",
        "ly",
        "airfoil_code",
        "chord",
        "leading_edge",
        "airfoil_angle_deg",
        "mach",
        "static_pressure",
        "static_temperature",
        "angle_of_attack_deg",
        "reynolds_number",
        "turbulence_intensity",
        "turbulent_viscosity_ratio",
        "pseudo_steps",
        "cfl",
        "simple_iterations",
        "pressure_correction_iterations",
        "velocity_relaxation",
        "pressure_relaxation",
        "temperature_relaxation",
        "turbulence_relaxation",
        "pressure_velocity_coupling",
        "turbulence_model",
        "convection_scheme",
        "pressure_linear_solver",
        "pressure_sor_omega",
        "pressure_correction_tolerance",
        "pressure_pcg_max_iterations",
        "outlet_static_pressure",
        "wall_temperature",
        "convergence_tolerance",
        "force_coefficient_tolerance",
        "convergence_window",
        "output_npz",
    }
    if unknown:
        raise ValueError(f"Unknown NACA pressure-based config keys: {sorted(unknown)}")
    return NACAPressureBasedConfig(
        nx=int(values.get("nx", NACAPressureBasedConfig.nx)),
        ny=int(values.get("ny", NACAPressureBasedConfig.ny)),
        lx=float(values.get("lx", NACAPressureBasedConfig.lx)),
        ly=float(values.get("ly", NACAPressureBasedConfig.ly)),
        airfoil_code=str(values.get("airfoil_code", NACAPressureBasedConfig.airfoil_code)),
        chord=float(values.get("chord", NACAPressureBasedConfig.chord)),
        leading_edge=_as_pair("leading_edge", values.get("leading_edge", NACAPressureBasedConfig.leading_edge)),
        airfoil_angle_deg=float(values.get("airfoil_angle_deg", NACAPressureBasedConfig.airfoil_angle_deg)),
        mach=float(values.get("mach", NACAPressureBasedConfig.mach)),
        static_pressure=float(values.get("static_pressure", NACAPressureBasedConfig.static_pressure)),
        static_temperature=float(values.get("static_temperature", NACAPressureBasedConfig.static_temperature)),
        angle_of_attack_deg=float(values.get("angle_of_attack_deg", NACAPressureBasedConfig.angle_of_attack_deg)),
        reynolds_number=float(values.get("reynolds_number", NACAPressureBasedConfig.reynolds_number)),
        turbulence_intensity=float(values.get("turbulence_intensity", NACAPressureBasedConfig.turbulence_intensity)),
        turbulent_viscosity_ratio=float(
            values.get("turbulent_viscosity_ratio", NACAPressureBasedConfig.turbulent_viscosity_ratio)
        ),
        pseudo_steps=int(values.get("pseudo_steps", NACAPressureBasedConfig.pseudo_steps)),
        cfl=float(values.get("cfl", NACAPressureBasedConfig.cfl)),
        simple_iterations=int(values.get("simple_iterations", NACAPressureBasedConfig.simple_iterations)),
        pressure_correction_iterations=int(
            values.get("pressure_correction_iterations", NACAPressureBasedConfig.pressure_correction_iterations)
        ),
        velocity_relaxation=float(values.get("velocity_relaxation", NACAPressureBasedConfig.velocity_relaxation)),
        pressure_relaxation=float(values.get("pressure_relaxation", NACAPressureBasedConfig.pressure_relaxation)),
        temperature_relaxation=float(
            values.get("temperature_relaxation", NACAPressureBasedConfig.temperature_relaxation)
        ),
        turbulence_relaxation=float(
            values.get("turbulence_relaxation", NACAPressureBasedConfig.turbulence_relaxation)
        ),
        pressure_velocity_coupling=str(
            values.get("pressure_velocity_coupling", NACAPressureBasedConfig.pressure_velocity_coupling)
        ),
        turbulence_model=str(values.get("turbulence_model", NACAPressureBasedConfig.turbulence_model)),
        convection_scheme=str(values.get("convection_scheme", NACAPressureBasedConfig.convection_scheme)),
        pressure_linear_solver=str(values.get("pressure_linear_solver", NACAPressureBasedConfig.pressure_linear_solver)),
        pressure_sor_omega=float(values.get("pressure_sor_omega", NACAPressureBasedConfig.pressure_sor_omega)),
        pressure_correction_tolerance=float(
            values.get("pressure_correction_tolerance", NACAPressureBasedConfig.pressure_correction_tolerance)
        ),
        pressure_pcg_max_iterations=int(
            values.get("pressure_pcg_max_iterations", NACAPressureBasedConfig.pressure_pcg_max_iterations)
        ),
        outlet_static_pressure=values.get("outlet_static_pressure", NACAPressureBasedConfig.outlet_static_pressure),
        wall_temperature=values.get("wall_temperature", NACAPressureBasedConfig.wall_temperature),
        convergence_tolerance=float(
            values.get("convergence_tolerance", NACAPressureBasedConfig.convergence_tolerance)
        ),
        force_coefficient_tolerance=float(
            values.get("force_coefficient_tolerance", NACAPressureBasedConfig.force_coefficient_tolerance)
        ),
        convergence_window=int(values.get("convergence_window", NACAPressureBasedConfig.convergence_window)),
        output_npz=values.get("output_npz", NACAPressureBasedConfig.output_npz),
    )


def load_config(path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> NACAPressureBasedConfig:
    config_path = Path(path).expanduser().resolve()
    return config_from_mapping(_load_yaml_mapping(config_path))


def run_naca_pressure_based(
    config: NACAPressureBasedConfig = NACAPressureBasedConfig(),
) -> tuple[object, list[object]]:
    grid = GridSpec.from_extent(config.nx, config.ny, config.lx, config.ly)
    obstacle_sdf = naca4_airfoil_levelset(
        grid,
        config.airfoil_code,
        chord=config.chord,
        leading_edge=config.leading_edge,
        angle=math.radians(config.airfoil_angle_deg),
        samples=256,
    )
    freestream = FreestreamCondition(
        mach=config.mach,
        static_pressure=config.static_pressure,
        static_temperature=config.static_temperature,
        angle_of_attack_deg=config.angle_of_attack_deg,
        reynolds_number=config.reynolds_number,
        turbulence_intensity=config.turbulence_intensity,
        turbulent_viscosity_ratio=config.turbulent_viscosity_ratio,
    )
    solver = PressureBasedSolverConfig(
        pseudo_steps=config.pseudo_steps,
        cfl=config.cfl,
        simple_iterations=config.simple_iterations,
        pressure_correction_iterations=config.pressure_correction_iterations,
        velocity_relaxation=config.velocity_relaxation,
        pressure_relaxation=config.pressure_relaxation,
        temperature_relaxation=config.temperature_relaxation,
        turbulence_relaxation=config.turbulence_relaxation,
        pressure_velocity_coupling=config.pressure_velocity_coupling,
        turbulence_model=config.turbulence_model,
        convection_scheme=config.convection_scheme,
        pressure_linear_solver=config.pressure_linear_solver,
        pressure_sor_omega=config.pressure_sor_omega,
        pressure_correction_tolerance=config.pressure_correction_tolerance,
        pressure_pcg_max_iterations=config.pressure_pcg_max_iterations,
        outlet_static_pressure=config.outlet_static_pressure,
        wall_temperature=config.wall_temperature,
        convergence_tolerance=config.convergence_tolerance,
        force_coefficient_tolerance=config.force_coefficient_tolerance,
        convergence_window=config.convergence_window,
    )
    state, history = run_pressure_based_solver(
        grid,
        obstacle_sdf,
        chord=config.chord,
        freestream=freestream,
        config=solver,
    )
    if config.output_npz:
        output_path = Path(config.output_npz).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sound = np.sqrt(solver.gamma * solver.gas_constant * state.temperature)
        speed = np.sqrt(state.u * state.u + state.v * state.v)
        rho_inf = freestream.static_pressure / (solver.gas_constant * freestream.static_temperature)
        u_inf = freestream.mach * math.sqrt(solver.gamma * solver.gas_constant * freestream.static_temperature)
        dynamic_pressure = 0.5 * rho_inf * u_inf * u_inf
        cp = (state.pressure - freestream.static_pressure) / max(dynamic_pressure, 1.0e-6)
        np.savez_compressed(
            output_path,
            density=state.density.astype(np.float32),
            u=state.u.astype(np.float32),
            v=state.v.astype(np.float32),
            pressure=state.pressure.astype(np.float32),
            temperature=state.temperature.astype(np.float32),
            k=state.turbulent_kinetic_energy.astype(np.float32),
            omega=state.specific_dissipation.astype(np.float32),
            mu_t=state.turbulent_viscosity.astype(np.float32),
            mach=(speed / np.maximum(sound, 1.0e-6)).astype(np.float32),
            cp=cp.astype(np.float32),
            fluid=state.fluid.astype(np.float32),
            residual_step=np.asarray([item.step for item in history], dtype=np.int32),
            residual_mass=np.asarray([item.mass for item in history], dtype=np.float32),
            residual_momentum=np.asarray([item.momentum for item in history], dtype=np.float32),
            residual_energy=np.asarray([item.energy for item in history], dtype=np.float32),
            residual_turbulence=np.asarray([item.turbulence for item in history], dtype=np.float32),
            residual_pressure_correction=np.asarray([item.pressure_correction for item in history], dtype=np.float32),
            normalized_mass=np.asarray([item.normalized_mass for item in history], dtype=np.float32),
            normalized_momentum=np.asarray([item.normalized_momentum for item in history], dtype=np.float32),
            normalized_energy=np.asarray([item.normalized_energy for item in history], dtype=np.float32),
            normalized_turbulence=np.asarray([item.normalized_turbulence for item in history], dtype=np.float32),
            drag_coefficient=np.asarray([item.drag_coefficient for item in history], dtype=np.float32),
            lift_coefficient=np.asarray([item.lift_coefficient for item in history], dtype=np.float32),
            moment_coefficient=np.asarray([item.moment_coefficient for item in history], dtype=np.float32),
            force_delta=np.asarray([item.force_delta for item in history], dtype=np.float32),
        )
    return state, history


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the experimental pressure-based compressible NACA example.")
    parser.add_argument("-c", "--config", default=str(DEFAULT_CONFIG_PATH), help="Path to the YAML configuration file.")
    parser.add_argument("--output-npz", default=None, help="Override the output npz path.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.output_npz is not None:
        config = NACAPressureBasedConfig(
            nx=config.nx,
            ny=config.ny,
            lx=config.lx,
            ly=config.ly,
            airfoil_code=config.airfoil_code,
            chord=config.chord,
            leading_edge=config.leading_edge,
            airfoil_angle_deg=config.airfoil_angle_deg,
            mach=config.mach,
            static_pressure=config.static_pressure,
            static_temperature=config.static_temperature,
            angle_of_attack_deg=config.angle_of_attack_deg,
            reynolds_number=config.reynolds_number,
            turbulence_intensity=config.turbulence_intensity,
            turbulent_viscosity_ratio=config.turbulent_viscosity_ratio,
            pseudo_steps=config.pseudo_steps,
            cfl=config.cfl,
            simple_iterations=config.simple_iterations,
            pressure_correction_iterations=config.pressure_correction_iterations,
            velocity_relaxation=config.velocity_relaxation,
            pressure_relaxation=config.pressure_relaxation,
            temperature_relaxation=config.temperature_relaxation,
            turbulence_relaxation=config.turbulence_relaxation,
            pressure_velocity_coupling=config.pressure_velocity_coupling,
            turbulence_model=config.turbulence_model,
            convection_scheme=config.convection_scheme,
            pressure_linear_solver=config.pressure_linear_solver,
            pressure_sor_omega=config.pressure_sor_omega,
            pressure_correction_tolerance=config.pressure_correction_tolerance,
            pressure_pcg_max_iterations=config.pressure_pcg_max_iterations,
            outlet_static_pressure=config.outlet_static_pressure,
            wall_temperature=config.wall_temperature,
            convergence_tolerance=config.convergence_tolerance,
            force_coefficient_tolerance=config.force_coefficient_tolerance,
            convergence_window=config.convergence_window,
            output_npz=args.output_npz,
        )
    state, history = run_naca_pressure_based(config)
    final = history[-1]
    print(
        "Experimental NACA pressure-based run finished: "
        f"steps={final.step}, dt={final.dt:.6e}, mass={final.mass:.6e}, "
        f"momentum={final.momentum:.6e}, energy={final.energy:.6e}, turbulence={final.turbulence:.6e}, "
        f"pcorr={final.pressure_correction:.6e}"
    )
    if config.output_npz:
        print(f"Field dump written to {Path(config.output_npz).expanduser()}")
    print(
        "Methods: "
        f"{config.pressure_velocity_coupling} + {config.turbulence_model} + {config.convection_scheme}"
    )
    print(
        "Ranges: "
        f"p=({float(np.min(state.pressure)):.2f}, {float(np.max(state.pressure)):.2f}), "
        f"rho=({float(np.min(state.density)):.6f}, {float(np.max(state.density)):.6f}), "
        f"T=({float(np.min(state.temperature)):.2f}, {float(np.max(state.temperature)):.2f})"
    )
    print(
        "Aerodynamics: "
        f"CD={final.drag_coefficient:.6f}, CL={final.lift_coefficient:.6f}, "
        f"CM={final.moment_coefficient:.6f}, dF={final.force_delta:.6e}"
    )
    print(
        "Normalized residuals: "
        f"mass={final.normalized_mass:.6e}, momentum={final.normalized_momentum:.6e}, "
        f"energy={final.normalized_energy:.6e}, turbulence={final.normalized_turbulence:.6e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
