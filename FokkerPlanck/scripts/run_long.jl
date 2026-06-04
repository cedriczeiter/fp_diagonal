"""
long-time horizon experiment with the DiagonalMultiStencil solver.

(large D_xy coupling, fixed_mobile=true to keep the tensor constant).
"""

if !isdefined(Main, :FokkerPlanck)
    include(joinpath(@__DIR__, "../src/FokkerPlanck.jl"))
end
using .FokkerPlanck
using Plots, Printf

# ─── Grid and parameters ────────────────────────────────────────────────────
Nx = 60; Ny = 60
grid   = make_grid(xmin=2.0, xmax=8.0, Nx=Nx, ymin=0.0, ymax=1.0, Ny=Ny)
params = ModelParams(
    1.0,    # epsilon — strong D_xy coupling
    10.0,   # C10
    5.0,    # C11
    0.0;    # G10
    G11=0.0, r=grid.r, sigma=0.9,
)

U0 = make_gaussian(grid)
normalize_to_mass!(U0, grid, 1.0)

T_end = 100.0

# ─── Compute P ───────────────────────────────────────────────────────────────
kin = compute_kinetics(grid.x, params)
D   = compute_diffusion_tensor(kin.d1, kin.d2)
P   = compute_P_min(D.Dxx, D.Dyy, D.Dxy, grid.r)
@printf("Using DiagonalMultiStencil with P = %d\n", P)

# ─── Run ─────────────────────────────────────────────────────────────────────
println("Running long simulation (T_end=$(T_end))...")
res = run_simulation(
    DiagonalMultiStencil(P), grid, params, U0, T_end;
    fixed_mobile=true,
)
@printf("Finished: t=%.4f, %d steps, min(U)=%.4e, exploded=%s\n",
        res.t, length(res.times)-1, minimum(res.U), res.exploded)

# ─── Plots ────────────────────────────────────────────────────────────────────
mkpath(joinpath(@__DIR__, "../output"))

p1 = heatmap(grid.x, grid.y, res.U',
             title="U at t=$(res.t)", xlabel="x", ylabel="y", color=:viridis)
p2 = plot(res.times, res.mass_rel_hist,
          label="rel. mass drift", xlabel="t", ylabel="|Δmass|/mass₀", yscale=:log10)
p3 = plot(res.times, res.min_u_hist,
          label="min(U)", xlabel="t", ylabel="min(U)")
hline!(p3, [0.0], ls=:dash, color=:black, label="")

p = plot(p1, p2, p3, layout=(1,3), size=(1200,350))
savefig(p, joinpath(@__DIR__, "../output/long_run_diagnostics.pdf"))
println("Plot saved to FokkerPlanckClean/output/long_run_diagnostics.pdf")
