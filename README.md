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
- `naca_density_based`: experimental compressible density-based Euler flow with conservative Rusanov fluxes and embedded-boundary apertures
- `naca_density_based`: experimental compressible density-based Euler flow with conservative Rusanov fluxes and embedded-boundary apertures

The pressure-based solver is intentionally isolated from the older incompressible path. It is currently:

- 2D only
- steady/pseudo-time experimental rather than production-grade CFD
- based on a collocated finite-volume style update implemented directly with Warp kernels
- includes west/top/bottom farfield conditions, an east pressure outlet, and solid-wall turbulence/temperature boundary handling
- reports `CD`, `CL`, `CM`, pressure-correction residuals, and normalized convergence histories in the output `npz`
- defaults to face-flux correction plus `SOR` for pressure correction; a matrix-free diagonal-preconditioned `PCG` path is available as an experimental option

It does not yet implement:

- multigrid or robust industrial-grade linear algebra
- wall functions, transition modeling, or force coefficient post-processing
- high-order or strictly conservative embedded-boundary treatment around the airfoil body

The density-based solver is a separate path intended for regimes where the pressure-based formulation is a poor fit, especially transonic and supersonic cases with shocks. It is currently:

- 2D only
- inviscid Euler rather than Navier-Stokes, so the wall model is slip-wall instead of no-slip
- conservative in `rho`, `rho*u`, `rho*v`, and total energy
- first-order accurate in space with Rusanov face fluxes, which is stable but numerically diffusive
- embedded-boundary aware through `fluid_fraction`, `aperture_x`, and `aperture_y`
- equipped with freestream left/top/bottom boundaries and a simple copy-out outflow on the right
- able to export `Mach`, `Cp`, density gradient, and a schlieren-like scalar for shock visualization

This means two practical consequences are important when reading the plots:

- the velocity on the airfoil surface is not expected to go to zero, because this is a slip-wall inviscid model rather than a viscous no-slip model
- weak shocks will be smeared unless the grid is reasonably fine, because the current flux is intentionally biased toward robustness

The density-based solver is a separate path intended for regimes where the pressure-based formulation is a poor fit, especially transonic and supersonic cases with shocks. It is currently:

- 2D only
- inviscid Euler rather than Navier-Stokes, so the wall model is slip-wall instead of no-slip
- conservative in `rho`, `rho*u`, `rho*v`, and total energy
- first-order accurate in space with Rusanov face fluxes, which is stable but numerically diffusive
- embedded-boundary aware through `fluid_fraction`, `aperture_x`, and `aperture_y`
- equipped with freestream left/top/bottom boundaries and a simple copy-out outflow on the right
- able to export `Mach`, `Cp`, density gradient, and a schlieren-like scalar for shock visualization

This means two practical consequences are important when reading the plots:

- the velocity on the airfoil surface is not expected to go to zero, because this is a slip-wall inviscid model rather than a viscous no-slip model
- weak shocks will be smeared unless the grid is reasonably fine, because the current flux is intentionally biased toward robustness

### `naca_pressure_based` Workflow

The entry point [warp_fluid/examples/naca_pressure_based.py](/home/ycsong/cfd/warp-fluid/warp_fluid/examples/naca_pressure_based.py) builds a 2D Cartesian grid, rasterizes a NACA 4-digit airfoil to a level-set, and then advances a steady compressible pressure-based solve in pseudo-time on a collocated cell-centered state.

The high-level loop is:

1. Build the grid and airfoil signed-distance field from the YAML config.
2. Initialize freestream density, pressure, temperature, velocity, `k`, `omega`, and wall-distance fields.
3. Keep the full state on the selected Warp device, typically `cuda:0` or another CUDA device.
4. At every pseudo-step, estimate a stable `dt` from convective, diffusive, and turbulence time scales.
5. Update temperature from convection, diffusion, and compressibility work terms.
6. Recompute density from the ideal-gas relation and re-apply outer and wall boundary conditions.
7. Recompute turbulent viscosity and auxiliary SST blending quantities.
8. Run several `SIMPLE` iterations:
   set up predicted velocities, assemble mass fluxes, solve a pressure-correction equation with `SOR`, correct face fluxes, recover cell velocities, and update pressure and density.
9. Update `k` and `omega` with SST production, diffusion, convection, and destruction terms.
10. Compute residuals and aerodynamic coefficients, then stop when the residual, force-change, and pressure-correction windows are all below their tolerances.

The current Warp implementation also includes several stabilizers intended to keep the solve bounded on GPU:

- `dt` is limited by convection, effective viscosity, and `omega`
- pressure correction is clipped before velocity correction
- velocity, thermodynamic variables, and turbulent viscosity are bounded relative to the freestream state
- solid-adjacent closed faces are excluded from pressure-flux correction

These choices make the example more robust, but they also mean it should still be treated as an experimental solver rather than a validated production CFD workflow.

### `naca_pressure_based` Parameters

Default values live in [warp_fluid/examples/configs/naca_pressure_based.yaml](/home/ycsong/cfd/warp-fluid/warp_fluid/examples/configs/naca_pressure_based.yaml).

| Key | Meaning |
| --- | --- |
| `nx`, `ny` | Number of grid cells in `x` and `y`. Higher values improve resolution but raise cost and stiffness. |
| `lx`, `ly` | Physical domain size. The airfoil and boundaries are interpreted inside this box. |
| `airfoil_code` | NACA 4-digit designation such as `2412` or `0012`. |
| `chord` | Airfoil chord length in domain units. |
| `leading_edge` | Airfoil leading-edge position as `[x, y]`. |
| `airfoil_angle_deg` | Geometric rotation of the airfoil itself in degrees. |
| `mach` | Freestream Mach number used to construct inflow speed and dynamic pressure. |
| `static_pressure` | Freestream static pressure. Also used on inflow, top, and bottom boundaries. |
| `static_temperature` | Freestream static temperature. Also used on inflow, top, and bottom boundaries. |
| `angle_of_attack_deg` | Flow angle of attack in degrees. This rotates the freestream velocity vector. |
| `reynolds_number` | Reynolds number used to infer laminar viscosity from chord and freestream speed. |
| `turbulence_intensity` | Freestream turbulence intensity used to initialize `k`. |
| `turbulent_viscosity_ratio` | Freestream ratio used to initialize `omega` from `k` and laminar viscosity. |
| `pseudo_steps` | Maximum number of pseudo-time steps in the outer steady solve. |
| `cfl` | Base convective CFL factor. The actual `dt` is additionally reduced by diffusion and turbulence limits. |
| `simple_iterations` | Number of inner `SIMPLE` pressure-velocity coupling iterations per pseudo-step. |
| `pressure_correction_iterations` | Maximum red-black `SOR` sweeps for the pressure-correction equation in each `SIMPLE` iteration. |
| `velocity_relaxation` | Under-relaxation applied to predicted velocity updates. Lower values are usually more stable. |
| `pressure_relaxation` | Under-relaxation applied when adding pressure correction back to pressure. |
| `temperature_relaxation` | Under-relaxation applied to temperature updates. |
| `turbulence_relaxation` | Under-relaxation applied to `k` and `omega` updates. |
| `pressure_velocity_coupling` | Pressure-velocity coupling scheme. The example currently supports `SIMPLE`. |
| `turbulence_model` | Turbulence closure name. The example currently supports `sst_k_omega`. |
| `convection_scheme` | Convective discretization name. The example currently supports `second_order_upwind`. |
| `pressure_linear_solver` | Pressure-correction linear solver choice. The Warp path currently uses `SOR`. |
| `pressure_sor_omega` | Over-relaxation factor for `SOR`. Larger values converge faster when stable, but can destabilize the solve. |
| `pressure_correction_tolerance` | Early-stop tolerance for the pressure-correction residual. |
| `pressure_pcg_max_iterations` | Maximum iterations for the experimental `PCG` path. Not used by the current Warp `SOR` flow. |
| `outlet_static_pressure` | Static pressure imposed at the east outlet. If omitted, falls back to `static_pressure`. |
| `wall_temperature` | Optional isothermal wall temperature. If omitted, wall temperature is extrapolated from the fluid side. |
| `convergence_tolerance` | Windowed threshold for normalized mass, momentum, energy, and turbulence residuals. |
| `force_coefficient_tolerance` | Windowed threshold for change in aerodynamic coefficients. |
| `convergence_window` | Number of recent steps used when checking convergence. |
| `device` | Warp execution device such as `cpu`, `cuda:0`, or `cuda:2`. |
| `output_npz` | Output path for the final field dump and residual history arrays. |

### Practical Tuning Notes

- If the solve stays finite but residuals plateau, first increase `simple_iterations` or `pressure_correction_iterations`.
- If the solve becomes oscillatory, reduce `cfl`, `velocity_relaxation`, `pressure_relaxation`, or `turbulence_relaxation`.
- If the airfoil is too close to a boundary, enlarge `lx`, `ly`, or move `leading_edge`.
- If a CUDA device fails to initialize, switch `device` to a valid card such as `cuda:0` or fall back to `cpu`.

### `naca_density_based` Workflow

The entry point [warp_fluid/examples/naca_density_based.py](/home/ycsong/cfd/warp-fluid/warp_fluid/examples/naca_density_based.py) uses the same grid and NACA level-set generation as the pressure-based example, but it advances a separate conservative state:

1. Build the Cartesian grid and NACA signed-distance field from YAML.
2. Convert the level-set into a smooth cell volume fraction and x/y face apertures.
3. Initialize the whole fluid domain with freestream density, momentum, and total energy.
4. Keep those arrays on the selected Warp device for the full pseudo-time loop.
5. Estimate a stable explicit step from the current maximum local wave speed, then reduce it further with a cut-cell safety factor.
6. On regular full-fluid cells, reconstruct second-order limited face states before computing the conservative Rusanov flux.
7. Blend open-face flux and slip-wall pressure flux using the embedded-boundary aperture.
8. Update each fluid cell conservatively, scaled by its fluid volume fraction.
9. Re-impose the outer freestream/outflow conditions, sanitize any non-physical states, and collect residual and force histories.
10. Write the final fields plus shock-oriented diagnostics such as `density_gradient` and `schlieren`.

This solver is deliberately much simpler than a production compressible CFD code. The current design goal is to get a stable conservative supersonic prototype in Warp with minimal CPU/GPU transfers, not to reproduce high-order shock resolution.

The current implementation uses a hybrid strategy:

- regular fluid cells away from boundaries use a MUSCL-style limited linear reconstruction
- boundary-adjacent, cut-cell, and incomplete neighborhoods automatically fall back to first-order states

This keeps the solver substantially more robust than a fully second-order embedded-boundary treatment, at the cost of leaving some shock smearing near the airfoil itself.

### `naca_density_based` Parameters

Default values live in [warp_fluid/examples/configs/naca_density_based.yaml](/home/ycsong/cfd/warp-fluid/warp_fluid/examples/configs/naca_density_based.yaml).

| Key | Meaning |
| --- | --- |
| `nx`, `ny` | Number of finite-volume cells in `x` and `y`. Higher values sharpen shocks but increase cost. |
| `lx`, `ly` | Physical domain size. These should be large enough that the farfield boundaries do not contaminate the airfoil solution. |
| `airfoil_code` | NACA 4-digit airfoil identifier such as `2412` or `0012`. |
| `chord` | Airfoil chord length in domain units. |
| `leading_edge` | Leading-edge position `[x, y]` inside the domain. |
| `airfoil_angle_deg` | Geometric rotation of the airfoil itself. |
| `mach` | Freestream Mach number. This directly sets inflow velocity and strongly affects shock strength. |
| `static_pressure` | Freestream static pressure used in the initial state and farfield boundaries. |
| `static_temperature` | Freestream static temperature used to compute density and sound speed. |
| `angle_of_attack_deg` | Flow angle of attack in degrees. |
| `reynolds_number` | Kept for interface consistency with the freestream struct, but not used by the inviscid Euler update. |
| `turbulence_intensity` | Also kept for interface compatibility; unused by the density-based solver. |
| `turbulent_viscosity_ratio` | Interface-compatible but unused in the density-based solver. |
| `pseudo_steps` | Maximum number of explicit pseudo-time iterations. |
| `cfl` | Explicit CFL number. Reducing this is the first stability knob when oscillations appear. |
| `dt_safety` | Extra global safety factor multiplied into the explicit time step after the CFL estimate. |
| `min_fluid_fraction` | Minimum effective cut-cell volume fraction used in the update. Larger values are more robust but smear embedded-boundary details more. |
| `max_velocity_factor` | Post-update velocity guard relative to the freestream velocity and sound-speed scale. If exceeded, the cell is clipped or reset. |
| `convergence_tolerance` | Windowed threshold for normalized density, momentum, and energy residuals. |
| `force_coefficient_tolerance` | Windowed threshold for change in `CD`, `CL`, and `CM`. |
| `convergence_window` | Number of recent steps used in the convergence check. |
| `device` | Warp execution device such as `cpu` or `cuda:0`. |
| `output_npz` | Output path for the field dump and residual history arrays. |

### Density-Based Practical Notes

- If shocks are hard to see, first increase `nx` and `ny`; the current first-order Rusanov flux is intentionally diffusive.
- The second-order reconstruction is only activated on regular full-fluid neighborhoods. Embedded-boundary regions still use a safer first-order fallback.
- If the run stays finite but converges very slowly, increase `pseudo_steps` before raising `cfl`.
- If oscillations or `NaN`s appear, reduce `cfl` first; next lower `dt_safety` or raise `min_fluid_fraction`.
- For shock visualization, prefer the `schlieren` field over raw Mach contours.
- The solver now resets non-finite or non-physical cells back to the previous state, then to freestream as a last resort. If that happens frequently, the time step is still too aggressive.

## Testing

```bash
pytest -q warp_fluid/solver/test_optimize.py
pytest -q warp_fluid/ops/test_mask.py
pytest -q warp_fluid/geom/levelset_grid/test_levelset.py
```
