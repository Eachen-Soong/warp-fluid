# warp_fluid

`warp_fluid` is a Warp-based fluid simulation library for lightweight 2D Eulerian experiments.
It focuses on MAC-grid workflows, level-set obstacles, and matrix-free pressure solves built directly on `warp-lang`.
It now also includes an experimental 2D pressure-based compressible solver for NACA airfoil cases with `SIMPLE`, second-order upwind convection, and an SST `k-omega` closure.

## Features

- 2D Cartesian grid and field data structures
- Level-set geometry helpers for obstacles
- Advection, diffusion, and incompressibility operators
- Matrix-free iterative linear solvers for pressure projection
- Packaged example entry points for cylinder flow, incompressible NACA airfoil flow, pressure-based compressible NACA airfoil flow, and Tesla valve flow

## Installation

From the repository root:

```bash
pip install -e .
```

Core dependencies are installed automatically:

- `warp-lang`
- `numpy`
- `pyyaml`

If you want video export for the examples, install the optional extras:

```bash
pip install -e ".[examples]"
```

## Quick Start

Import the library:

```python
from warp_fluid import GridSpec, LinearSolveConfig, make_incompressible
```

Run packaged examples as modules:

```bash
python -m warp_fluid.examples.cylinder_flow
python -m warp_fluid.examples.naca_airfoil
python -m warp_fluid.examples.naca_pressure_based
python -m warp_fluid.examples.tesla_valve
```

Or use the installed console scripts:

```bash
warp-fluid-cylinder-flow
warp-fluid-naca-airfoil
warp-fluid-naca-pressure-based
warp-fluid-tesla-valve
```

## Minimal API

```python
from warp_fluid import GridSpec, LinearSolveConfig, make_incompressible
from warp_fluid.core import CenteredField, MACField, SolidMask, VelocityBoundary

velocity, pressure, stats = make_incompressible(
    velocity,
    solid=solid,
    pressure=pressure,
    solve_config=LinearSolveConfig(method="CG"),
    boundary=boundary,
)
```

The low-level linear solver is also available directly:

```python
from warp_fluid import LinearSolveConfig, solve_linear

solution, stats = solve_linear(operator, rhs, LinearSolveConfig(method="CG"))
```

## Example Configuration

The example YAML files are installed with the package under `warp_fluid/examples/configs/`.

Module entry points use those defaults automatically:

```bash
python -m warp_fluid.examples.cylinder_flow --solver-method CG
python -m warp_fluid.examples.naca_airfoil --solver-method CG
python -m warp_fluid.examples.naca_pressure_based
python -m warp_fluid.examples.tesla_valve --solver-method biCG-stab
```

Useful shared config keys:

- `pressure_iterations`
- `solver_method`
- `device`
- `video.enabled`
- `video.path`

## Repository Layout

```text
warp_fluid/
  core/        Grid, field, and boundary data structures
  geom/        Level-set geometry helpers
  ops/         Differential operators, interpolation, and masking
  physics/     Advection, forcing, and incompressibility projection
  solver/      Iterative linear solvers
  examples/    Packaged demos and YAML configs
```

## NACA Airfoil Examples

`warp_fluid` now includes a NACA 4-digit airfoil level-set generator plus two packaged airfoil examples:

- `naca_airfoil`: incompressible MAC-grid projection flow with an airfoil obstacle and angled inflow
- `naca_pressure_based`: experimental compressible pressure-based flow with `SIMPLE`, second-order upwind convection, and SST `k-omega`

The pressure-based solver is intentionally isolated from the older incompressible path. It is currently:

- 2D only
- steady/pseudo-time experimental rather than production-grade CFD
- based on a collocated finite-volume style update implemented in NumPy for flexibility
- includes west/top/bottom farfield conditions, an east pressure outlet, and solid-wall turbulence/temperature boundary handling
- reports `CD`, `CL`, `CM`, pressure-correction residuals, and normalized convergence histories in the output `npz`
- defaults to face-flux correction plus `SOR` for pressure correction; a matrix-free diagonal-preconditioned `PCG` path is available as an experimental option

It does not yet implement:

- multigrid or robust industrial-grade linear algebra
- wall functions, transition modeling, or force coefficient post-processing
- cut-cell geometry handling around the airfoil body

## Testing

```bash
pytest -q warp_fluid/solver/test_optimize.py
pytest -q warp_fluid/ops/test_mask.py
pytest -q warp_fluid/geom/levelset_grid/test_levelset.py
```
