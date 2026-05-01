#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

from warp_fluid.examples.naca_pressure_based import (
    DEFAULT_CONFIG_PATH,
    build_naca_solver_inputs,
    load_config,
    write_naca_snapshot_npz,
)
from warp_fluid.solver.pressure_based_wp import run_pressure_based_solver_warp


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run NACA pressure-based solver to convergence, then continue a 1000-step trajectory."
    )
    parser.add_argument("-c", "--config", default=str(DEFAULT_CONFIG_PATH), help="Path to the YAML configuration file.")
    parser.add_argument("--trajectory-steps", type=int, default=1000, help="Number of post-stabilization steps to run.")
    parser.add_argument("--frame-interval", type=int, default=10, help="Save one frame every N continuation steps.")
    parser.add_argument(
        "--output-dir",
        default="outputs/naca_pressure_based/trajectory",
        help="Directory for stable state, frame dumps, and rendered images.",
    )
    parser.add_argument("--device", default=None, help="Override the device from the config.")
    parser.add_argument("--dpi", type=int, default=140, help="PNG frame DPI.")
    return parser


def _overlay_geometry(ax, solid_levelset: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
    level_min = float(np.min(solid_levelset))
    if level_min < 0.0:
        ax.contourf(
            x,
            y,
            solid_levelset.T,
            levels=[level_min, 0.0],
            colors=[(0.05, 0.05, 0.05, 0.92)],
        )
    ax.contour(x, y, solid_levelset.T, levels=[0.0], colors="white", linewidths=0.9)


def _render_frame(
    png_path: Path,
    *,
    state,
    config,
    step: int,
    continuation_step: int,
    aero: Optional[dict[str, float]],
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import colors

    fluid = np.asarray(state.fluid, dtype=bool)
    speed = np.sqrt(state.u * state.u + state.v * state.v)
    sound = np.sqrt(1.4 * 287.05 * state.temperature)
    mach = speed / np.maximum(sound, 1.0e-6)
    rho_inf = config.static_pressure / (287.05 * config.static_temperature)
    u_inf = config.mach * math.sqrt(1.4 * 287.05 * config.static_temperature)
    q_inf = max(0.5 * rho_inf * u_inf * u_inf, 1.0e-6)
    cp = (state.pressure - config.static_pressure) / q_inf

    nx, ny = state.u.shape
    dx = config.lx / nx
    dy = config.ly / ny
    x = np.linspace(0.5 * dx, config.lx - 0.5 * dx, nx)
    y = np.linspace(0.5 * dy, config.ly - 0.5 * dy, ny)
    extent = (0.0, config.lx, 0.0, config.ly)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    panels = [
        ("Speed", np.ma.masked_where(~fluid, speed), "viridis", None),
        ("Cp", np.ma.masked_where(~fluid, cp), "RdBu_r", colors.CenteredNorm(vcenter=0.0)),
    ]
    for ax, (title, field, cmap, norm) in zip(axes, panels):
        image = ax.imshow(field.T, origin="lower", extent=extent, aspect="equal", cmap=cmap, norm=norm)
        _overlay_geometry(ax, state.solid_levelset, x, y)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, shrink=0.9)

    subtitle = f"abs step={step}  continuation={continuation_step}"
    if aero is not None:
        subtitle += f"  CD={aero['cd']:.5f}  CL={aero['cl']:.5f}  CM={aero['cm']:.5f}"
    fig.suptitle(f"NACA Trajectory Frame\n{subtitle}", fontsize=13)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.device is not None:
        config = replace(config, device=args.device)

    output_dir = Path(args.output_dir).expanduser().resolve()
    frame_npz_dir = output_dir / "frames_npz"
    frame_png_dir = output_dir / "frames_png"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_npz_dir.mkdir(parents=True, exist_ok=True)
    frame_png_dir.mkdir(parents=True, exist_ok=True)

    grid, obstacle_sdf, freestream, solver = build_naca_solver_inputs(config)

    stable_warp_state, stable_history = run_pressure_based_solver_warp(
        grid,
        obstacle_sdf,
        chord=config.chord,
        freestream=freestream,
        config=solver,
        device=config.device,
        stop_on_convergence=True,
    )
    stable_state = stable_warp_state.to_numpy_state()
    stable_step = stable_history[-1].step if stable_history else 0

    write_naca_snapshot_npz(output_dir / "stable_state.npz", stable_state, freestream=freestream, solver=solver, history=stable_history)
    stable_aero = None
    if stable_history:
        stable_aero = {
            "cd": stable_history[-1].drag_coefficient,
            "cl": stable_history[-1].lift_coefficient,
            "cm": stable_history[-1].moment_coefficient,
        }
    _render_frame(
        frame_png_dir / "frame_0000.png",
        state=stable_state,
        config=config,
        step=stable_step,
        continuation_step=0,
        aero=stable_aero,
        dpi=args.dpi,
    )
    write_naca_snapshot_npz(frame_npz_dir / "frame_0000.npz", stable_state, freestream=freestream, solver=solver)

    frame_records: list[dict[str, float | int | str]] = [
        {
            "frame_index": 0,
            "absolute_step": int(stable_step),
            "continuation_step": 0,
            "dt": float(stable_state.dt),
            "npz": str(frame_npz_dir / "frame_0000.npz"),
            "png": str(frame_png_dir / "frame_0000.png"),
        }
    ]

    continuation_solver = replace(solver, pseudo_steps=args.trajectory_steps)
    frame_index = 1

    def _save_callback(step: int, state_dev, snapshot) -> None:
        nonlocal frame_index
        continuation_step = step - stable_step
        if continuation_step <= 0 or continuation_step % args.frame_interval != 0:
            return
        state_np = state_dev.to_numpy_state()
        npz_path = frame_npz_dir / f"frame_{frame_index:04d}.npz"
        png_path = frame_png_dir / f"frame_{frame_index:04d}.png"
        write_naca_snapshot_npz(npz_path, state_np, freestream=freestream, solver=solver)
        _render_frame(
            png_path,
            state=state_np,
            config=config,
            step=step,
            continuation_step=continuation_step,
            aero={
                "cd": snapshot.drag_coefficient,
                "cl": snapshot.lift_coefficient,
                "cm": snapshot.moment_coefficient,
            },
            dpi=args.dpi,
        )
        frame_records.append(
            {
                "frame_index": frame_index,
                "absolute_step": int(step),
                "continuation_step": int(continuation_step),
                "dt": float(snapshot.dt),
                "cd": float(snapshot.drag_coefficient),
                "cl": float(snapshot.lift_coefficient),
                "cm": float(snapshot.moment_coefficient),
                "npz": str(npz_path),
                "png": str(png_path),
            }
        )
        frame_index += 1

    _, continuation_history = run_pressure_based_solver_warp(
        grid,
        obstacle_sdf,
        chord=config.chord,
        freestream=freestream,
        config=continuation_solver,
        device=config.device,
        initial_state=stable_warp_state,
        step_offset=stable_step,
        stop_on_convergence=False,
        step_callback=_save_callback,
    )

    history_path = output_dir / "trajectory_history.npz"
    np.savez_compressed(
        history_path,
        residual_step=np.asarray([item.step for item in continuation_history], dtype=np.int32),
        residual_mass=np.asarray([item.mass for item in continuation_history], dtype=np.float32),
        residual_momentum=np.asarray([item.momentum for item in continuation_history], dtype=np.float32),
        residual_energy=np.asarray([item.energy for item in continuation_history], dtype=np.float32),
        residual_turbulence=np.asarray([item.turbulence for item in continuation_history], dtype=np.float32),
        residual_pressure_correction=np.asarray([item.pressure_correction for item in continuation_history], dtype=np.float32),
        normalized_mass=np.asarray([item.normalized_mass for item in continuation_history], dtype=np.float32),
        normalized_momentum=np.asarray([item.normalized_momentum for item in continuation_history], dtype=np.float32),
        normalized_energy=np.asarray([item.normalized_energy for item in continuation_history], dtype=np.float32),
        normalized_turbulence=np.asarray([item.normalized_turbulence for item in continuation_history], dtype=np.float32),
        drag_coefficient=np.asarray([item.drag_coefficient for item in continuation_history], dtype=np.float32),
        lift_coefficient=np.asarray([item.lift_coefficient for item in continuation_history], dtype=np.float32),
        moment_coefficient=np.asarray([item.moment_coefficient for item in continuation_history], dtype=np.float32),
        force_delta=np.asarray([item.force_delta for item in continuation_history], dtype=np.float32),
    )

    manifest = {
        "stable_step": int(stable_step),
        "trajectory_steps": int(args.trajectory_steps),
        "frame_interval": int(args.frame_interval),
        "num_frames": int(len(frame_records)),
        "stable_state_npz": str(output_dir / "stable_state.npz"),
        "trajectory_history_npz": str(history_path),
        "frames": frame_records,
    }
    with (output_dir / "trajectory_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)

    print(
        f"Stabilized at step {stable_step}, continued for {args.trajectory_steps} steps, "
        f"saved {len(frame_records)} frames to {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
