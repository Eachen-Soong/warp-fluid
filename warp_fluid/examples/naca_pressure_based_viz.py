#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

from warp_fluid.core import GridSpec
from warp_fluid.examples.naca_pressure_based import DEFAULT_CONFIG_PATH, load_config
from warp_fluid.geom.levelset_grid import naca4_airfoil_levelset


def _masked_field(values: np.ndarray, fluid: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_where(~fluid, values)


def _resolve_npz_path(npz_path: Optional[str], config_path: Path) -> Path:
    if npz_path is not None:
        return Path(npz_path).expanduser().resolve()
    config = load_config(config_path)
    if not config.output_npz:
        raise ValueError("The config does not define output_npz, and no --npz path was provided.")
    return Path(config.output_npz).expanduser().resolve()


def _resolve_solid_levelset(data: np.lib.npyio.NpzFile, config, field_shape: tuple[int, int]) -> np.ndarray:
    if "solid_levelset" in data.files:
        solid_levelset = np.asarray(data["solid_levelset"], dtype=np.float64)
        if solid_levelset.shape == field_shape:
            return solid_levelset
    grid = GridSpec.from_extent(field_shape[0], field_shape[1], config.lx, config.ly)
    return np.asarray(
        naca4_airfoil_levelset(
            grid,
            config.airfoil_code,
            chord=config.chord,
            leading_edge=config.leading_edge,
            angle=np.deg2rad(config.airfoil_angle_deg),
            samples=256,
        ),
        dtype=np.float64,
    )


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
    ax.contour(
        x,
        y,
        solid_levelset.T,
        levels=[0.0],
        colors="white",
        linewidths=0.9,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize the output of warp_fluid.examples.naca_pressure_based.")
    parser.add_argument("--npz", default=None, help="Path to the .npz file produced by naca_pressure_based.")
    parser.add_argument(
        "-c",
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the YAML config used for the run. Used to recover physical domain extents.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the generated images. Defaults to a sibling directory next to the npz.",
    )
    parser.add_argument("--dpi", type=int, default=160, help="Figure DPI.")
    parser.add_argument(
        "--streamline-density",
        type=float,
        default=1.0,
        help="Density passed to matplotlib.streamplot for the speed panel.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively after writing them.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    npz_path = _resolve_npz_path(args.npz, config_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Could not find output npz: {npz_path}")

    data = np.load(npz_path)
    u = np.asarray(data["u"], dtype=np.float64)
    v = np.asarray(data["v"], dtype=np.float64)
    pressure = np.asarray(data["pressure"], dtype=np.float64)
    temperature = np.asarray(data["temperature"], dtype=np.float64)
    mach = np.asarray(data["mach"], dtype=np.float64)
    cp = np.asarray(data["cp"], dtype=np.float64)
    mu_t = np.asarray(data["mu_t"], dtype=np.float64)
    speed = np.sqrt(u * u + v * v)
    field_shape = u.shape

    solid_levelset = _resolve_solid_levelset(data, config, field_shape)
    if "fluid" in data.files and np.asarray(data["fluid"]).shape == field_shape:
        fluid = np.asarray(data["fluid"], dtype=np.float32) > 0.5
    else:
        fluid = solid_levelset > 0.0

    nx, ny = fluid.shape
    dx = config.lx / nx
    dy = config.ly / ny
    x = config.leading_edge[0] * 0.0 + np.linspace(0.5 * dx, config.lx - 0.5 * dx, nx)
    y = config.leading_edge[1] * 0.0 + np.linspace(0.5 * dy, config.ly - 0.5 * dy, ny)
    extent = (0.0, config.lx, 0.0, config.ly)

    if args.output_dir is None:
        output_dir = npz_path.parent / "viz"
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    from matplotlib import colors

    overview_path = output_dir / "naca_pressure_based_overview.png"
    history_path = output_dir / "naca_pressure_based_history.png"

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    panels = [
        ("Speed", _masked_field(speed, fluid), "viridis", None),
        ("Pressure", _masked_field(pressure, fluid), "coolwarm", None),
        ("Cp", _masked_field(cp, fluid), "RdBu_r", colors.CenteredNorm(vcenter=0.0)),
        ("Mach", _masked_field(mach, fluid), "magma", None),
        ("Temperature", _masked_field(temperature, fluid), "plasma", None),
        ("Turbulent Viscosity", _masked_field(mu_t, fluid), "cividis", None),
    ]

    for ax, (title, field, cmap, norm) in zip(axes.flat, panels):
        image = ax.imshow(
            field.T,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap=cmap,
            norm=norm,
        )
        _overlay_geometry(ax, solid_levelset, x, y)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, shrink=0.9)

    axes[0, 0].streamplot(
        x,
        y,
        u.T,
        v.T,
        color="white",
        linewidth=0.6,
        density=args.streamline_density,
        arrowsize=0.7,
    )

    residual_step = np.asarray(data["residual_step"], dtype=np.int32)
    normalized_mass = np.asarray(data["normalized_mass"], dtype=np.float64)
    normalized_momentum = np.asarray(data["normalized_momentum"], dtype=np.float64)
    normalized_energy = np.asarray(data["normalized_energy"], dtype=np.float64)
    normalized_turbulence = np.asarray(data["normalized_turbulence"], dtype=np.float64)
    residual_pressure_correction = np.asarray(data["residual_pressure_correction"], dtype=np.float64)
    drag = np.asarray(data["drag_coefficient"], dtype=np.float64)
    lift = np.asarray(data["lift_coefficient"], dtype=np.float64)
    moment = np.asarray(data["moment_coefficient"], dtype=np.float64)
    force_delta = np.asarray(data["force_delta"], dtype=np.float64)

    final_drag = float(drag[-1]) if drag.size else float("nan")
    final_lift = float(lift[-1]) if lift.size else float("nan")
    final_moment = float(moment[-1]) if moment.size else float("nan")
    fig.suptitle(
        "NACA Pressure-Based Overview\n"
        f"final CD={final_drag:.5f}, CL={final_lift:.5f}, CM={final_moment:.5f}",
        fontsize=14,
    )
    fig.savefig(overview_path, dpi=args.dpi)

    fig_hist, axes_hist = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    ax0 = axes_hist[0]
    ax0.semilogy(residual_step, normalized_mass, label="mass")
    ax0.semilogy(residual_step, normalized_momentum, label="momentum")
    ax0.semilogy(residual_step, normalized_energy, label="energy")
    ax0.semilogy(residual_step, normalized_turbulence, label="turbulence")
    ax0.semilogy(residual_step, np.maximum(residual_pressure_correction, 1.0e-30), label="pcorr")
    ax0.set_title("Residual History")
    ax0.set_xlabel("pseudo step")
    ax0.set_ylabel("residual")
    ax0.grid(True, which="both", alpha=0.3)
    ax0.legend()

    ax1 = axes_hist[1]
    ax1.plot(residual_step, drag, label="CD")
    ax1.plot(residual_step, lift, label="CL")
    ax1.plot(residual_step, moment, label="CM")
    ax1.plot(residual_step, force_delta, label="dF", linestyle="--")
    ax1.set_title("Aerodynamic Coefficients")
    ax1.set_xlabel("pseudo step")
    ax1.set_ylabel("coefficient")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    fig_hist.suptitle(npz_path.name, fontsize=13)
    fig_hist.savefig(history_path, dpi=args.dpi)

    print(f"Wrote overview figure to {overview_path}")
    print(f"Wrote history figure to {history_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)
        plt.close(fig_hist)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
