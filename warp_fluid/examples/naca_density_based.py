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
from warp_fluid.solver.density_based_wp import (
    DensityBasedSolverConfig,
    run_density_based_solver_warp,
)
from warp_fluid.solver.pressure_based_np import FreestreamCondition


DEFAULT_CONFIG_PATH = Path(__file__).with_name("configs") / "naca_density_based.yaml"


@dataclass(frozen=True)
class NACADensityBasedConfig:
    nx: int = 512
    ny: int = 256
    lx: float = 12.0
    ly: float = 6.0
    airfoil_code: str = "2412"
    chord: float = 1.5
    leading_edge: tuple[float, float] = (3.0, 3.0)
    airfoil_angle_deg: float = 0.0
    mach: float = 1.5
    static_pressure: float = 101325.0
    static_temperature: float = 288.15
    angle_of_attack_deg: float = 4.0
    reynolds_number: float = 1.0e6
    turbulence_intensity: float = 0.0
    turbulent_viscosity_ratio: float = 1.0
    pseudo_steps: int = 4000
    cfl: float = 0.35
    dt_safety: float = 0.7
    min_fluid_fraction: float = 0.2
    max_velocity_factor: float = 6.0
    convergence_tolerance: float = 1.0e-4
    force_coefficient_tolerance: float = 1.0e-4
    convergence_window: int = 10
    device: Optional[str] = None
    output_npz: Optional[str] = "outputs/naca_density_based/final_state.npz"


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


def config_from_mapping(data: Mapping[str, Any]) -> NACADensityBasedConfig:
    values = dict(data)
    return NACADensityBasedConfig(
        nx=int(values.get("nx", NACADensityBasedConfig.nx)),
        ny=int(values.get("ny", NACADensityBasedConfig.ny)),
        lx=float(values.get("lx", NACADensityBasedConfig.lx)),
        ly=float(values.get("ly", NACADensityBasedConfig.ly)),
        airfoil_code=str(values.get("airfoil_code", NACADensityBasedConfig.airfoil_code)),
        chord=float(values.get("chord", NACADensityBasedConfig.chord)),
        leading_edge=_as_pair("leading_edge", values.get("leading_edge", NACADensityBasedConfig.leading_edge)),
        airfoil_angle_deg=float(values.get("airfoil_angle_deg", NACADensityBasedConfig.airfoil_angle_deg)),
        mach=float(values.get("mach", NACADensityBasedConfig.mach)),
        static_pressure=float(values.get("static_pressure", NACADensityBasedConfig.static_pressure)),
        static_temperature=float(values.get("static_temperature", NACADensityBasedConfig.static_temperature)),
        angle_of_attack_deg=float(values.get("angle_of_attack_deg", NACADensityBasedConfig.angle_of_attack_deg)),
        reynolds_number=float(values.get("reynolds_number", NACADensityBasedConfig.reynolds_number)),
        turbulence_intensity=float(values.get("turbulence_intensity", NACADensityBasedConfig.turbulence_intensity)),
        turbulent_viscosity_ratio=float(
            values.get("turbulent_viscosity_ratio", NACADensityBasedConfig.turbulent_viscosity_ratio)
        ),
        pseudo_steps=int(values.get("pseudo_steps", NACADensityBasedConfig.pseudo_steps)),
        cfl=float(values.get("cfl", NACADensityBasedConfig.cfl)),
        dt_safety=float(values.get("dt_safety", NACADensityBasedConfig.dt_safety)),
        min_fluid_fraction=float(values.get("min_fluid_fraction", NACADensityBasedConfig.min_fluid_fraction)),
        max_velocity_factor=float(values.get("max_velocity_factor", NACADensityBasedConfig.max_velocity_factor)),
        convergence_tolerance=float(values.get("convergence_tolerance", NACADensityBasedConfig.convergence_tolerance)),
        force_coefficient_tolerance=float(
            values.get("force_coefficient_tolerance", NACADensityBasedConfig.force_coefficient_tolerance)
        ),
        convergence_window=int(values.get("convergence_window", NACADensityBasedConfig.convergence_window)),
        device=values.get("device", NACADensityBasedConfig.device),
        output_npz=values.get("output_npz", NACADensityBasedConfig.output_npz),
    )


def load_config(path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> NACADensityBasedConfig:
    return config_from_mapping(_load_yaml_mapping(Path(path).expanduser().resolve()))


def _write_output(
    path: Union[str, Path],
    primitive: dict[str, np.ndarray],
    history,
    *,
    freestream: FreestreamCondition,
    gamma: float = 1.4,
    gas_constant: float = 287.05,
) -> None:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    density = primitive["density"]
    u = primitive["u"]
    v = primitive["v"]
    pressure = primitive["pressure"]
    temperature = primitive["temperature"]
    mach = primitive["mach"]
    fluid = primitive["fluid"]
    solid_levelset = primitive["solid_levelset"]
    rho_grad_x = np.zeros_like(density)
    rho_grad_y = np.zeros_like(density)
    rho_grad_x[1:-1, :] = 0.5 * (density[2:, :] - density[:-2, :])
    rho_grad_y[:, 1:-1] = 0.5 * (density[:, 2:] - density[:, :-2])
    density_gradient = np.sqrt(rho_grad_x * rho_grad_x + rho_grad_y * rho_grad_y)
    rho_inf = freestream.static_pressure / (gas_constant * freestream.static_temperature)
    a_inf = math.sqrt(gamma * gas_constant * freestream.static_temperature)
    u_inf = freestream.mach * a_inf
    q_inf = max(0.5 * rho_inf * u_inf * u_inf, 1.0e-6)
    cp = (pressure - freestream.static_pressure) / q_inf
    schlieren = np.log1p(12.0 * density_gradient)
    np.savez_compressed(
        output_path,
        density=density.astype(np.float32),
        u=u.astype(np.float32),
        v=v.astype(np.float32),
        pressure=pressure.astype(np.float32),
        temperature=temperature.astype(np.float32),
        mach=mach.astype(np.float32),
        cp=cp.astype(np.float32),
        density_gradient=density_gradient.astype(np.float32),
        schlieren=schlieren.astype(np.float32),
        fluid=fluid.astype(np.float32),
        fluid_fraction=primitive["fluid_fraction"].astype(np.float32),
        solid_levelset=solid_levelset.astype(np.float32),
        freestream_density=np.asarray(rho_inf, dtype=np.float32),
        freestream_pressure=np.asarray(freestream.static_pressure, dtype=np.float32),
        freestream_temperature=np.asarray(freestream.static_temperature, dtype=np.float32),
        freestream_mach=np.asarray(freestream.mach, dtype=np.float32),
        freestream_dynamic_pressure=np.asarray(q_inf, dtype=np.float32),
        residual_step=np.asarray([item.step for item in history], dtype=np.int32),
        residual_density=np.asarray([item.density_residual for item in history], dtype=np.float32),
        residual_momentum=np.asarray([item.momentum_residual for item in history], dtype=np.float32),
        residual_energy=np.asarray([item.energy_residual for item in history], dtype=np.float32),
        normalized_density=np.asarray([item.normalized_density for item in history], dtype=np.float32),
        normalized_momentum=np.asarray([item.normalized_momentum for item in history], dtype=np.float32),
        normalized_energy=np.asarray([item.normalized_energy for item in history], dtype=np.float32),
        drag_coefficient=np.asarray([item.drag_coefficient for item in history], dtype=np.float32),
        lift_coefficient=np.asarray([item.lift_coefficient for item in history], dtype=np.float32),
        moment_coefficient=np.asarray([item.moment_coefficient for item in history], dtype=np.float32),
        force_delta=np.asarray([item.force_delta for item in history], dtype=np.float32),
    )


def run_naca_density_based(config: NACADensityBasedConfig = NACADensityBasedConfig()):
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
    solver = DensityBasedSolverConfig(
        pseudo_steps=config.pseudo_steps,
        cfl=config.cfl,
        dt_safety=config.dt_safety,
        min_fluid_fraction=config.min_fluid_fraction,
        max_velocity_factor=config.max_velocity_factor,
        convergence_tolerance=config.convergence_tolerance,
        force_coefficient_tolerance=config.force_coefficient_tolerance,
        convergence_window=config.convergence_window,
    )
    state, history = run_density_based_solver_warp(
        grid,
        obstacle_sdf,
        chord=config.chord,
        freestream=freestream,
        config=solver,
        device=config.device,
    )
    primitive = state.primitive_fields(solver.gamma, solver.gas_constant)
    if config.output_npz:
        _write_output(
            config.output_npz,
            primitive,
            history,
            freestream=freestream,
            gamma=solver.gamma,
            gas_constant=solver.gas_constant,
        )
    return primitive, history


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the experimental density-based supersonic NACA example.")
    parser.add_argument("-c", "--config", default=str(DEFAULT_CONFIG_PATH), help="Path to the YAML configuration file.")
    parser.add_argument("--output-npz", default=None, help="Override the output npz path.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.output_npz is not None:
        config = NACADensityBasedConfig(**{**config.__dict__, "output_npz": args.output_npz})
    primitive, history = run_naca_density_based(config)
    final = history[-1]
    print(
        "Experimental density-based NACA run finished: "
        f"steps={final.step}, dt={final.dt:.6e}, "
        f"rho_res={final.density_residual:.6e}, mom_res={final.momentum_residual:.6e}, "
        f"E_res={final.energy_residual:.6e}, CD={final.drag_coefficient:.6f}, "
        f"CL={final.lift_coefficient:.6f}, CM={final.moment_coefficient:.6f}"
    )
    print(
        "Ranges: "
        f"Mach=({float(np.min(primitive['mach'])):.3f}, {float(np.max(primitive['mach'])):.3f}), "
        f"p=({float(np.min(primitive['pressure'])):.1f}, {float(np.max(primitive['pressure'])):.1f})"
    )
    if config.output_npz:
        print(f"Field dump written to {Path(config.output_npz).expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
