# FokkerPlanck

Julia implementation of three solvers for the 2D Becker-Döring Fokker-Planck PDE with a coupled mobile-concentration ODE.

## Solvers

| Solver | Type | Diffusion | Notes |
|---|---|---|---|
| `Naive9Point()` | IMEX Strang | 9-point cross stencil | **Not** positivity-preserving when D_xy is large |
| `DiagonalMultiStencil(P)` | IMEX Strang | Multi-stencil M-matrix | Positivity guaranteed for all P ≥ P_min |
| `Kurganov()` | Explicit SSP-RK3 | KT central-upwind (eq. 4.20) | Diffusion CFL constraint |

For all three, advection uses MP5 reconstruction + SSP-RK3. The mobile concentration ODE (C10, C11) uses a backward-Euler implicit step.

## Quick start

From the repo root, activate the project and install dependencies:

```julia
using Pkg
Pkg.activate("FokkerPlanck")
Pkg.instantiate()
```

Then run a simulation:

```julia
using FokkerPlanck

grid   = make_grid(xmin=2.0, xmax=8.0, Nx=50, ymin=0.0, ymax=1.0, Ny=50)
params = ModelParams(0.5, 5.0, 2.0, 0.1; r=grid.r, sigma=0.9)
U0     = normalize_to_mass!(make_gaussian(grid), grid, 1.0)

# Auto-select P for the positivity-preserving multi-stencil solver
kin = compute_kinetics(grid.x, params)
D   = compute_diffusion_tensor(kin.d1, kin.d2)
P   = compute_P_min(D.Dxx, D.Dyy, D.Dxy, grid.r)

res = run_simulation(DiagonalMultiStencil(P), grid, params, U0, 1.0)
println("min(U) = ", minimum(res.U))   # always >= 0
```

## Example scripts

Example scripts to run are in [`scripts/`](scripts/) and [`experiments/`](experiments/).

## Reproducing the results

Output PDFs/PNGs are saved to `FokkerPlanck/output/`. Run from the repo root.

```bash
# Reproduce all figures from the poster-session presentation
julia --project=FokkerPlanck FokkerPlanck/experiments/slides_plots.jl

# Reproduce all figures from the final report
julia --project=FokkerPlanck FokkerPlanck/experiments/fig_[...].jl
```

This generates `slides_exp1.pdf`, `slides_exp1_profiles_t010.pdf`, and `slides_exp1_profiles_t025.pdf` in `FokkerPlanck/output/`.

## Module structure

```
src/
  FokkerPlanck.jl          Main module
  helpers.jl               MP5 reconstruction, SSP-RK3, row-major utilities
  grid.jl                  Grid struct + make_grid
  physics.jl               ModelParams, kinetics, diffusion tensor, velocities
  initial_conditions.jl    make_gaussian, normalize_to_mass!
  diagnostics.jl           total_mass, material_diagnostics
  timestepper.jl           AbstractFPSolver, run_simulation, step dispatch
  solvers/
    naive9point.jl         Naive9Point + build_diffusion_matrix
    multistencil.jl        DiagonalMultiStencil + build_diffusion_matrix
    kurganov.jl            Kurganov KT spatial operator
```

## The P-selection problem

The single-(1,1) diagonal stencil requires the mesh ratio r = dx/dy to satisfy

    D_xy / (2 D_yy)  ≤  r  ≤  2 D_xx / D_xy

When this fails (large D_xy), `compute_P_min` returns the smallest P ≥ 1 such
that adding the (1,P) and (P,1) stencils widens the window enough to
accommodate r. All off-diagonal entries of the resulting matrix are ≥ 0
(M-matrix property), guaranteeing that the implicit solve preserves positivity.
