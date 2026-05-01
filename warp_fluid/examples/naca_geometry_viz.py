#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
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
from warp_fluid.geom.levelset_grid import naca4_airfoil_levelset, naca4_airfoil_polygon


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize the NACA geometry used by naca_pressure_based.")
    parser.add_argument(
        "-c",
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the YAML config used for the run.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path. Defaults to outputs/naca_pressure_based/naca_geometry.png.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Figure DPI.")
    parser.add_argument("--show", action="store_true", help="Show the figure interactively after writing it.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    config = load_config(Path(args.config).expanduser().resolve())
    grid = GridSpec.from_extent(config.nx, config.ny, config.lx, config.ly)
    polygon = naca4_airfoil_polygon(
        config.airfoil_code,
        chord=config.chord,
        leading_edge=config.leading_edge,
        angle=math.radians(config.airfoil_angle_deg),
        samples=256,
    )
    levelset = naca4_airfoil_levelset(
        grid,
        config.airfoil_code,
        chord=config.chord,
        leading_edge=config.leading_edge,
        angle=math.radians(config.airfoil_angle_deg),
        samples=256,
    )
    solid = levelset <= 0.0
    extent = (0.0, config.lx, 0.0, config.ly)

    if args.output is None:
        output_path = Path("outputs/naca_pressure_based/naca_geometry.png").resolve()
    else:
        output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    from matplotlib import colors

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    axes[0].fill(polygon[:, 0], polygon[:, 1], color="#d97789", alpha=0.9, ec="black", lw=1.2)
    axes[0].set_title("Airfoil Polygon")
    axes[0].set_xlim(0.0, config.lx)
    axes[0].set_ylim(0.0, config.ly)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].grid(alpha=0.25)

    sdf_im = axes[1].imshow(levelset.T, origin="lower", extent=extent, aspect="equal", cmap="coolwarm")
    axes[1].contour(levelset.T, levels=[0.0], colors="black", linewidths=1.0, origin="lower", extent=extent)
    axes[1].set_title("Signed Distance Field")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(sdf_im, ax=axes[1], shrink=0.92)

    mask_im = axes[2].imshow(
        solid.T.astype(np.float32),
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=colors.ListedColormap(["#f7f7f7", "#f04f4f"]),
        vmin=0.0,
        vmax=1.0,
    )
    axes[2].contour(solid.T.astype(np.float32), levels=[0.5], colors="black", linewidths=0.9, origin="lower", extent=extent)
    axes[2].set_title(f"Solid Mask ({int(np.count_nonzero(solid))} cells)")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(mask_im, ax=axes[2], shrink=0.92)

    fig.suptitle(
        f"NACA {config.airfoil_code}  nx={config.nx} ny={config.ny}  chord={config.chord}",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=args.dpi)
    print(f"Wrote geometry figure to {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
