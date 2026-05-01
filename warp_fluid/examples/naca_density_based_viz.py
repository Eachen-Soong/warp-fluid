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

from warp_fluid.examples.naca_density_based import DEFAULT_CONFIG_PATH, load_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize the output of the density-based NACA example.")
    parser.add_argument("--npz", default=None, help="Path to the density-based output npz.")
    parser.add_argument("-c", "--config", default=str(DEFAULT_CONFIG_PATH), help="Path to the YAML config.")
    parser.add_argument("--output-dir", default=None, help="Directory for generated images.")
    parser.add_argument("--dpi", type=int, default=160, help="Figure DPI.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)
    npz_path = Path(args.npz).expanduser().resolve() if args.npz else Path(config.output_npz).expanduser().resolve()
    data = np.load(npz_path)
    fluid = np.asarray(data["fluid"], dtype=np.float32) > 0.5
    solid_levelset = np.asarray(data["solid_levelset"], dtype=np.float64)
    pressure = np.asarray(data["pressure"], dtype=np.float64)
    mach = np.asarray(data["mach"], dtype=np.float64)
    cp = np.asarray(data["cp"], dtype=np.float64)
    rho_grad = np.asarray(data["density_gradient"], dtype=np.float64)
    schlieren = (
        np.asarray(data["schlieren"], dtype=np.float64)
        if "schlieren" in data.files
        else np.log1p(12.0 * np.maximum(rho_grad, 0.0))
    )

    nx, ny = pressure.shape
    dx = config.lx / nx
    dy = config.ly / ny
    x = np.linspace(0.5 * dx, config.lx - 0.5 * dx, nx)
    y = np.linspace(0.5 * dy, config.ly - 0.5 * dy, ny)
    extent = (0.0, config.lx, 0.0, config.ly)

    outdir = (npz_path.parent / "viz") if args.output_dir is None else Path(args.output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    from matplotlib import colors

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    panels = [
        ("Mach", np.ma.masked_where(~fluid, mach), "magma", None),
        ("Cp", np.ma.masked_where(~fluid, cp), "RdBu_r", colors.CenteredNorm(vcenter=0.0)),
        ("Density Gradient", np.ma.masked_where(~fluid, rho_grad), "viridis", None),
        ("Schlieren", np.ma.masked_where(~fluid, schlieren), "gray", None),
    ]
    for ax, (title, field, cmap, norm) in zip(axes.flat, panels):
        im = ax.imshow(field.T, origin="lower", extent=extent, aspect="equal", cmap=cmap, norm=norm)
        ax.contourf(x, y, solid_levelset.T, levels=[float(np.min(solid_levelset)), 0.0], colors=[(0.05, 0.05, 0.05, 0.92)])
        ax.contour(x, y, solid_levelset.T, levels=[0.0], colors="white", linewidths=0.9)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.9)
    png_path = outdir / "naca_density_based_overview.png"
    fig.savefig(png_path, dpi=args.dpi)
    plt.close(fig)

    if "residual_step" in data.files:
        fig_hist, ax = plt.subplots(1, 1, figsize=(8, 4), constrained_layout=True)
        ax.semilogy(data["residual_step"], np.maximum(data["normalized_density"], 1.0e-30), label="rho")
        ax.semilogy(data["residual_step"], np.maximum(data["normalized_momentum"], 1.0e-30), label="mom")
        ax.semilogy(data["residual_step"], np.maximum(data["normalized_energy"], 1.0e-30), label="E")
        ax.set_xlabel("pseudo step")
        ax.set_ylabel("normalized residual")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        hist_path = outdir / "naca_density_based_history.png"
        fig_hist.savefig(hist_path, dpi=args.dpi)
        plt.close(fig_hist)
        print(f"Wrote history figure to {hist_path}")

    print(f"Wrote overview figure to {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
