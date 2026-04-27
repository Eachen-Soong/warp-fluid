#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

from warp_fluid.core import CenteredField, GridSpec, MACField, VelocityBoundary
from warp_fluid.geom.levelset_grid import naca4_airfoil_levelset
from warp_fluid.ops.diff import cell_center_velocity
from warp_fluid.ops.mask import solid_mask_from_levelset
from warp_fluid.physics.advect import advect_mac_semi_lagrangian
from warp_fluid.physics.force import diffuse_velocity_explicit
from warp_fluid.physics.incompressible import make_incompressible
from warp_fluid.solver.optimize import LinearSolveConfig


DEFAULT_CONFIG_PATH = Path(__file__).with_name("configs") / "naca_airfoil.yaml"
SUPPORTED_VIDEO_FIELDS = {"speed", "pressure", "vorticity"}


@dataclass(frozen=True)
class NACAAirfoilVideoConfig:
    enabled: bool = False
    path: str = "outputs/naca_airfoil/naca_airfoil.mp4"
    fps: int = 24
    dpi: int = 120
    every: int = 1
    field: str = "speed"
    cmap: str = "viridis"
    bitrate: int = 2400

    def __post_init__(self) -> None:
        if self.fps < 1:
            raise ValueError("video.fps must be >= 1.")
        if self.dpi < 1:
            raise ValueError("video.dpi must be >= 1.")
        if self.every < 1:
            raise ValueError("video.every must be >= 1.")
        if self.field not in SUPPORTED_VIDEO_FIELDS:
            supported = ", ".join(sorted(SUPPORTED_VIDEO_FIELDS))
            raise ValueError(f"video.field must be one of: {supported}.")


@dataclass(frozen=True)
class NACAAirfoilConfig:
    nx: int = 256
    ny: int = 128
    lx: float = 12.0
    ly: float = 6.0
    airfoil_code: str = "2412"
    chord: float = 1.5
    leading_edge: tuple[float, float] = (3.0, 3.0)
    angle_of_attack_deg: float = 5.0
    inflow_speed: float = 1.0
    viscosity: float = 1.0e-3
    dt: float = 0.02
    steps: int = 300
    pressure_iterations: int = 160
    solver_method: str = "jacobi"
    obstacle_thickness: float = 0.0
    airfoil_samples: int = 256
    closed_trailing_edge: bool = True
    device: Optional[str] = None
    video: NACAAirfoilVideoConfig = field(default_factory=NACAAirfoilVideoConfig)

    def __post_init__(self) -> None:
        if self.nx < 1 or self.ny < 1:
            raise ValueError("nx and ny must be >= 1.")
        if self.lx <= 0.0 or self.ly <= 0.0:
            raise ValueError("lx and ly must be positive.")
        if self.chord <= 0.0:
            raise ValueError("chord must be positive.")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self.steps < 1:
            raise ValueError("steps must be >= 1.")
        if self.viscosity < 0.0:
            raise ValueError("viscosity must be non-negative.")
        if self.pressure_iterations < 1:
            raise ValueError("pressure_iterations must be >= 1.")
        if self.airfoil_samples < 16:
            raise ValueError("airfoil_samples must be >= 16.")
        if self.obstacle_thickness < 0.0:
            raise ValueError("obstacle_thickness must be non-negative.")
        if len(str(self.airfoil_code).strip()) != 4 or not str(self.airfoil_code).strip().isdigit():
            raise ValueError("airfoil_code must be a 4-digit NACA code such as '2412'.")


def _as_pair(name: str, values: Any) -> tuple[float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError(f"{name} must be a sequence of length 2.")
    return float(values[0]), float(values[1])


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level mapping in {path}.")
    return dict(data)


def config_from_mapping(data: Mapping[str, Any]) -> NACAAirfoilConfig:
    values = dict(data)
    video_values = values.pop("video", {})
    if video_values is None:
        video_values = {}
    if not isinstance(video_values, Mapping):
        raise ValueError("video must be a mapping.")
    unknown = set(values) - {
        "nx",
        "ny",
        "lx",
        "ly",
        "airfoil_code",
        "chord",
        "leading_edge",
        "angle_of_attack_deg",
        "inflow_speed",
        "viscosity",
        "dt",
        "steps",
        "pressure_iterations",
        "solver_method",
        "obstacle_thickness",
        "airfoil_samples",
        "closed_trailing_edge",
        "device",
    }
    if unknown:
        raise ValueError(f"Unknown NACA airfoil config keys: {sorted(unknown)}")
    unknown_video = set(video_values) - {"enabled", "path", "fps", "dpi", "every", "field", "cmap", "bitrate"}
    if unknown_video:
        raise ValueError(f"Unknown video config keys: {sorted(unknown_video)}")
    return NACAAirfoilConfig(
        nx=int(values.get("nx", NACAAirfoilConfig.nx)),
        ny=int(values.get("ny", NACAAirfoilConfig.ny)),
        lx=float(values.get("lx", NACAAirfoilConfig.lx)),
        ly=float(values.get("ly", NACAAirfoilConfig.ly)),
        airfoil_code=str(values.get("airfoil_code", NACAAirfoilConfig.airfoil_code)),
        chord=float(values.get("chord", NACAAirfoilConfig.chord)),
        leading_edge=_as_pair("leading_edge", values.get("leading_edge", NACAAirfoilConfig.leading_edge)),
        angle_of_attack_deg=float(values.get("angle_of_attack_deg", NACAAirfoilConfig.angle_of_attack_deg)),
        inflow_speed=float(values.get("inflow_speed", NACAAirfoilConfig.inflow_speed)),
        viscosity=float(values.get("viscosity", NACAAirfoilConfig.viscosity)),
        dt=float(values.get("dt", NACAAirfoilConfig.dt)),
        steps=int(values.get("steps", NACAAirfoilConfig.steps)),
        pressure_iterations=int(values.get("pressure_iterations", NACAAirfoilConfig.pressure_iterations)),
        solver_method=str(values.get("solver_method", NACAAirfoilConfig.solver_method)),
        obstacle_thickness=float(values.get("obstacle_thickness", NACAAirfoilConfig.obstacle_thickness)),
        airfoil_samples=int(values.get("airfoil_samples", NACAAirfoilConfig.airfoil_samples)),
        closed_trailing_edge=bool(values.get("closed_trailing_edge", NACAAirfoilConfig.closed_trailing_edge)),
        device=values.get("device", NACAAirfoilConfig.device),
        video=NACAAirfoilVideoConfig(
            enabled=bool(video_values.get("enabled", NACAAirfoilVideoConfig.enabled)),
            path=str(video_values.get("path", NACAAirfoilVideoConfig.path)),
            fps=int(video_values.get("fps", NACAAirfoilVideoConfig.fps)),
            dpi=int(video_values.get("dpi", NACAAirfoilVideoConfig.dpi)),
            every=int(video_values.get("every", NACAAirfoilVideoConfig.every)),
            field=str(video_values.get("field", NACAAirfoilVideoConfig.field)),
            cmap=str(video_values.get("cmap", NACAAirfoilVideoConfig.cmap)),
            bitrate=int(video_values.get("bitrate", NACAAirfoilVideoConfig.bitrate)),
        ),
    )


def load_config(path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> NACAAirfoilConfig:
    config_path = Path(path).expanduser().resolve()
    return config_from_mapping(_load_yaml_mapping(config_path))


def _visual_center_velocity(velocity: MACField, solid) -> tuple[np.ndarray, np.ndarray]:
    centered = cell_center_velocity(velocity, solid=solid)
    vel = centered.numpy()
    return vel[..., 0], vel[..., 1]


def _frame_from_fields(velocity: MACField, pressure: CenteredField, solid, field_name: str) -> np.ndarray:
    if field_name == "pressure":
        return pressure.numpy()
    u_center, v_center = _visual_center_velocity(velocity, solid)
    if field_name == "speed":
        return np.sqrt(u_center * u_center + v_center * v_center)

    dv_dx = np.gradient(v_center, velocity.grid.dx, axis=0)
    du_dy = np.gradient(u_center, velocity.grid.dy, axis=1)
    return dv_dx - du_dy


def save_flow_video(
    frames: list[np.ndarray],
    solid_fraction: np.ndarray,
    grid: GridSpec,
    step_ids: list[int],
    video_config: NACAAirfoilVideoConfig,
    *,
    airfoil_code: str,
) -> Path:
    if not frames:
        raise ValueError("No frames available for video export.")
    output_path = Path(video_config.path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib import colors

    fluid_fraction = np.clip(1.0 - solid_fraction, 0.0, 1.0).astype(np.float32)
    visible_values = [frame[fluid_fraction > 1.0e-4] for frame in frames if np.any(fluid_fraction > 1.0e-4)]
    if visible_values:
        data_min = min(float(np.min(values)) for values in visible_values)
        data_max = max(float(np.max(values)) for values in visible_values)
    else:
        data_min = min(float(np.min(frame)) for frame in frames)
        data_max = max(float(np.max(frame)) for frame in frames)
    if np.isclose(data_min, data_max):
        delta = 1.0 if np.isclose(data_max, 0.0) else abs(data_max) * 0.1
        data_min -= delta
        data_max += delta
    extent = (grid.x0, grid.x1, grid.y0, grid.y1)
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap(video_config.cmap).copy()
    cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    image = ax.imshow(
        frames[0].T,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=data_min,
        vmax=data_max,
        interpolation="bilinear",
        alpha=fluid_fraction.T,
    )
    obstacle_overlay = ax.imshow(
        solid_fraction.T,
        origin="lower",
        extent=extent,
        cmap="gray_r",
        norm=colors.Normalize(vmin=0.0, vmax=1.0),
        interpolation="bilinear",
        alpha=0.55 * solid_fraction.T,
    )
    ax.contour(
        solid_fraction.T,
        levels=[0.5],
        colors="white",
        linewidths=0.9,
        origin="lower",
        extent=extent,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"NACA {airfoil_code} airfoil ({video_config.field})")
    title = ax.text(
        0.02,
        0.98,
        f"step {step_ids[0]}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 4},
    )
    fig.colorbar(image, ax=ax, shrink=0.9, label=video_config.field)
    fig.tight_layout()

    suffix = output_path.suffix.lower()
    if suffix == ".mp4":
        writer = animation.FFMpegWriter(
            fps=video_config.fps,
            bitrate=video_config.bitrate,
            metadata={"title": f"NACA {airfoil_code} airfoil"},
        )
    elif suffix == ".gif":
        writer = animation.PillowWriter(fps=video_config.fps)
    else:
        raise ValueError(f"Unsupported video format '{suffix}'. Use .mp4 or .gif.")

    with writer.saving(fig, str(output_path), video_config.dpi):
        for frame, step in zip(frames, step_ids):
            image.set_data(frame.T)
            image.set_alpha(fluid_fraction.T)
            obstacle_overlay.set_alpha(0.55 * solid_fraction.T)
            title.set_text(f"step {step}")
            writer.grab_frame()
    plt.close(fig)
    return output_path


def run_naca_airfoil(
    config: NACAAirfoilConfig = NACAAirfoilConfig(),
    *,
    video_path: Optional[Union[str, Path]] = None,
) -> tuple[MACField, CenteredField]:
    grid = GridSpec.from_extent(config.nx, config.ny, config.lx, config.ly)
    velocity = MACField.zeros(grid, device=config.device)
    pressure = CenteredField.zeros(grid, device=config.device)
    obstacle_sdf = naca4_airfoil_levelset(
        grid,
        config.airfoil_code,
        chord=config.chord,
        leading_edge=config.leading_edge,
        angle=math.radians(config.angle_of_attack_deg),
        samples=config.airfoil_samples,
        closed_trailing_edge=config.closed_trailing_edge,
    )
    solid = solid_mask_from_levelset(grid, obstacle_sdf, thickness=config.obstacle_thickness, device=config.device)
    angle_rad = math.radians(config.angle_of_attack_deg)
    boundary = VelocityBoundary(
        inflow_west=(config.inflow_speed * math.cos(angle_rad), config.inflow_speed * math.sin(angle_rad)),
        no_slip_south=False,
        no_slip_north=False,
    )
    solve = LinearSolveConfig(method=config.solver_method, max_iterations=config.pressure_iterations)

    frames: list[np.ndarray] = []
    step_ids: list[int] = []
    video_config = config.video
    if video_path is not None:
        video_config = NACAAirfoilVideoConfig(
            enabled=True,
            path=str(video_path),
            fps=video_config.fps,
            dpi=video_config.dpi,
            every=video_config.every,
            field=video_config.field,
            cmap=video_config.cmap,
            bitrate=video_config.bitrate,
        )

    for step in range(1, config.steps + 1):
        velocity = advect_mac_semi_lagrangian(velocity, config.dt, solid=solid)
        velocity = diffuse_velocity_explicit(velocity, config.viscosity, config.dt, solid=solid)
        velocity, pressure, _ = make_incompressible(
            velocity,
            solid=solid,
            pressure=pressure,
            solve_config=solve,
            boundary=boundary,
        )
        if video_config.enabled and step % video_config.every == 0:
            frames.append(_frame_from_fields(velocity, pressure, solid, video_config.field))
            step_ids.append(step)

    if video_config.enabled:
        if not frames:
            frames.append(_frame_from_fields(velocity, pressure, solid, video_config.field))
            step_ids.append(config.steps)
        solid_fraction = solid.cell_numpy
        if solid_fraction is None:
            raise ValueError("Solid mask host data is unavailable for video export.")
        save_flow_video(frames, solid_fraction, grid, step_ids, video_config, airfoil_code=config.airfoil_code)
    return velocity, pressure


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the NACA airfoil example from a YAML config.")
    parser.add_argument(
        "-c",
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--video-path",
        default=None,
        help="Override the video output path from the YAML config and enable export.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable video export even if the YAML config enables it.",
    )
    parser.add_argument(
        "--solver-method",
        default=None,
        help="Override the linear pressure solver method from the YAML config.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.no_video and args.video_path is not None:
        raise SystemExit("--no-video cannot be combined with --video-path.")
    config = load_config(args.config)
    if args.no_video:
        config = NACAAirfoilConfig(
            nx=config.nx,
            ny=config.ny,
            lx=config.lx,
            ly=config.ly,
            airfoil_code=config.airfoil_code,
            chord=config.chord,
            leading_edge=config.leading_edge,
            angle_of_attack_deg=config.angle_of_attack_deg,
            inflow_speed=config.inflow_speed,
            viscosity=config.viscosity,
            dt=config.dt,
            steps=config.steps,
            pressure_iterations=config.pressure_iterations,
            solver_method=config.solver_method,
            obstacle_thickness=config.obstacle_thickness,
            airfoil_samples=config.airfoil_samples,
            closed_trailing_edge=config.closed_trailing_edge,
            device=config.device,
            video=NACAAirfoilVideoConfig(
                enabled=False,
                path=config.video.path,
                fps=config.video.fps,
                dpi=config.video.dpi,
                every=config.video.every,
                field=config.video.field,
                cmap=config.video.cmap,
                bitrate=config.video.bitrate,
            ),
        )
    if args.solver_method is not None:
        config = NACAAirfoilConfig(
            nx=config.nx,
            ny=config.ny,
            lx=config.lx,
            ly=config.ly,
            airfoil_code=config.airfoil_code,
            chord=config.chord,
            leading_edge=config.leading_edge,
            angle_of_attack_deg=config.angle_of_attack_deg,
            inflow_speed=config.inflow_speed,
            viscosity=config.viscosity,
            dt=config.dt,
            steps=config.steps,
            pressure_iterations=config.pressure_iterations,
            solver_method=args.solver_method,
            obstacle_thickness=config.obstacle_thickness,
            airfoil_samples=config.airfoil_samples,
            closed_trailing_edge=config.closed_trailing_edge,
            device=config.device,
            video=config.video,
        )
    velocity, pressure = run_naca_airfoil(config, video_path=args.video_path)
    pressure_values = pressure.numpy()
    print(
        "NACA airfoil flow finished: "
        f"airfoil={config.airfoil_code}, "
        f"steps={config.steps}, "
        f"video_field={config.video.field if config.video.enabled or args.video_path else 'disabled'}, "
        f"pressure_range=({float(np.min(pressure_values)):.6f}, {float(np.max(pressure_values)):.6f})"
    )
    if args.video_path is not None:
        print(f"Video written to {Path(args.video_path).expanduser()}")
    elif config.video.enabled and not args.no_video:
        print(f"Video written to {Path(config.video.path).expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
