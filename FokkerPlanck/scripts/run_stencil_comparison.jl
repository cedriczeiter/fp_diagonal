"""
demonstrate positivity failure of Naive9Point versus the multi-stencil guarantee, varying P.

This mirrors the core experiment from experiments/failing_stencil_examples.jl:
- Large C10/C11 → large D_xy → r lies OUTSIDE the single-(1,1) window
- Naive9Point and DiagonalMultiStencil(P=1) produce negative U values
- DiagonalMultiStencil(P=P_min) stays positive
"""

if !isdefined(Main, :FokkerPlanck)
    include(joinpath(@__DIR__, "../src/FokkerPlanck.jl"))
end
using .FokkerPlanck
using Plots, Printf

# ─── Setup: large coupling so P_min > 1 ──────────────────────────────────────
Nx = 50; Ny = 50
grid   = make_grid(xmin=2.0, xmax=8.0, Nx=Nx, ymin=0.0, ymax=1.0, Ny=Ny)
params = ModelParams(
    1.0,    # epsilon
    20.0,   # C10 — large, makes d1 large → large D_xy
    10.0,   # C11
    0.0;    # G10
    G11=0.0, r=grid.r, sigma=0.9,
)

U0 = make_gaussian(grid)
normalize_to_mass!(U0, grid, 1.0)
T_end = 0.3

# ─── Identify P_min ──────────────────────────────────────────────────────────
kin = compute_kinetics(grid.x, params)
D   = compute_diffusion_tensor(kin.d1, kin.d2)
P   = compute_P_min(D.Dxx, D.Dyy, D.Dxy, grid.r)
@printf("r = %.4f\n", grid.r)
@printf("max(Dxy) = %.4e, max(Dxx) = %.4e, max(Dyy) = %.4e\n",
        maximum(D.Dxy), maximum(D.Dxx), maximum(D.Dyy))
@printf("P_min = %d\n", P)

# ─── Run solvers ──────────────────────────────────────────────────────────────
solvers = [
    ("Naive9Point",                Naive9Point()),
    ("DiagMultiStencil(P=1)",      DiagonalMultiStencil(1)),
    ("DiagMultiStencil(P=$P)",     DiagonalMultiStencil(P)),
    ("Kurganov",                   Kurganov()),
]

results = []
for (name, solver) in solvers
    @printf("Running %-28s ...", name)
    res = run_simulation(solver, grid, params, U0, T_end; fixed_mobile=true)
    @printf("  min(U)=%+.2e  exploded=%s\n", minimum(res.U), res.exploded)
    push!(results, (name=name, res=res))
end

# ─── Plot min(U) over time ────────────────────────────────────────────────────
mkpath(joinpath(@__DIR__, "../output"))

p = plot(xlabel="t", ylabel="min(U)", title="Positivity comparison (fixed mobile)")
colors = [:blue, :red, :green, :orange]
for (i, (name, res)) in enumerate([(r.name, r.res) for r in results])
    plot!(p, res.times, res.min_u_hist, label=name, color=colors[i])
end
hline!(p, [0.0], ls=:dash, color=:black, label="0")
savefig(p, joinpath(@__DIR__, "../output/stencil_comparison_min_U.pdf"))

# ─── Final U heatmaps ─────────────────────────────────────────────────────────
ps = [heatmap(grid.x, grid.y, r.res.U',
              title=r.name, xlabel="x", ylabel="y",
              clim=(0, maximum(r.res.U .* (!r.res.exploded)) + 1e-30))
      for r in results]
p_grid = plot(ps..., layout=(2,2), size=(900,700))
savefig(p_grid, joinpath(@__DIR__, "../output/stencil_comparison_U_final.pdf"))

println("Plots saved to FokkerPlanckClean/output/")
