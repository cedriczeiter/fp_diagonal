"""
side-by-side comparison of all three solvers on a short-time run with moderate D_xy coupling.
"""

if !isdefined(Main, :FokkerPlanck)
    include(joinpath(@__DIR__, "../src/FokkerPlanck.jl"))
end

using .FokkerPlanck
using Plots, Printf

# ─── Grid and parameters ────────────────────────────────────────────────────
Nx = 40; Ny = 40
grid   = make_grid(xmin=2.0, xmax=8.0, Nx=Nx, ymin=0.0, ymax=1.0, Ny=Ny)
params = ModelParams(
    0.5,   # epsilon   — moderate D_xy coupling
    5.0,   # C10
    2.0,   # C11
    0.1;   # G10
    G11=0.0,
    r=grid.r, 
    sigma=0.9,
)

# ─── Initial condition ───────────────────────────────────────────────────────
U0 = make_gaussian(grid; sigma=0.5)
normalize_to_mass!(U0, grid, 1.0)

T_end = 0.5

# ─── Auto-select P for the multi-stencil solver ──────────────────────────────
kin = compute_kinetics(grid.x, params)
D   = compute_diffusion_tensor(kin.d1, kin.d2)
P   = compute_P_min(D.Dxx, D.Dyy, D.Dxy, grid.r)
@printf("Auto-selected P = %d\n", P)

# ─── Run all three solvers ────────────────────────────────────────────────────
println("Running Naive9Point...")
r9 = run_simulation(Naive9Point(), grid, params, U0, T_end)

println("Running DiagonalMultiStencil(P=$(P))...")
rm = run_simulation(DiagonalMultiStencil(P), grid, params, U0, T_end)

println("Running Kurganov...")
rk = run_simulation(Kurganov(), grid, params, U0, T_end)

# ─── Summary ─────────────────────────────────────────────────────────────────
@printf("\n%-25s  %6s  %8s  %8s  %8s\n",
        "Solver", "steps", "t_final", "min(U)", "exploded")
for (name, res) in [("Naive9Point", r9), ("DiagonalMultiStencil(P=$(P))", rm), ("Kurganov", rk)]
    @printf("%-25s  %6d  %8.4f  %8.2e  %s\n",
            name, length(res.times)-1, res.t, minimum(res.U), res.exploded)
end

# ─── Plots ────────────────────────────────────────────────────────────────────
mkpath(joinpath(@__DIR__, "../output"))

p_u = plot(
    heatmap(grid.x, grid.y, r9.U', title="Naive9Point  t=$(T_end)", xlabel="x", ylabel="y"),
    heatmap(grid.x, grid.y, rm.U', title="MultiStencil(P=$(P))  t=$(T_end)", xlabel="x", ylabel="y"),
    heatmap(grid.x, grid.y, rk.U', title="Kurganov  t=$(T_end)", xlabel="x", ylabel="y"),
    layout=(1,3), size=(1200,350),
)
savefig(p_u, joinpath(@__DIR__, "../output/comparison_U_final.pdf"))

p_min = plot(r9.times, r9.min_u_hist, label="Naive9Point", xlabel="t", ylabel="min(U)")
plot!(p_min, rm.times, rm.min_u_hist, label="MultiStencil(P=$(P))")
plot!(p_min, rk.times, rk.min_u_hist, label="Kurganov")
hline!(p_min, [0.0], ls=:dash, color=:black, label="")
savefig(p_min, joinpath(@__DIR__, "../output/comparison_min_U.pdf"))

println("Plots saved to FokkerPlanckClean/output/")
