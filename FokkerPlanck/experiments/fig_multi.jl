"""
Figure 4 — Multi-stencil window.

Panel A: P_min(r) admissibility step-function for the physical model
         (C10=3, C11=1, ε=1): P_min(r) = ceil(max(r/4, 1/r)).

Panel B: Adaptive geometric grid (α=0.12, x∈[2,150]), r sweeping 3→67.
         Per-cell P_min selection (column by column); solver stays positive.

Panel C: Uniform high-r grid (r=30, P_min=8), T=1.
"""

if !isdefined(Main, :FokkerPlanck)
    include(joinpath(@__DIR__, "../src/FokkerPlanck.jl"))
end
using .FokkerPlanck, Plots, Printf
import Plots: mm

const OUTDIR = joinpath(@__DIR__, "../output")
mkpath(OUTDIR)

M = 4mm

# ── Panel A : P_min(r) step function ─────────────────────────────────────────
# Use a representative x to get the tensor, then compute P_min vs r.
par_a = ModelParams(1.0, 3.0, 1.0, 0.0; r=1.0, sigma=0.9)
x_rep = [10.0]
kin_a = compute_kinetics(x_rep, par_a)
D_a   = compute_diffusion_tensor(kin_a.d1, kin_a.d2)

r_vals    = collect(range(0.5, 40.0, length=4000))
P_min_vals = [compute_P_min(D_a.Dxx, D_a.Dyy, D_a.Dxy, Float64(r)) for r in r_vals]

p_a = plot(r_vals, P_min_vals,
    title  = "Admissibility window P_min(r)",
    xlabel = "grid aspect ratio r = Δx/Δy",
    ylabel = "required stencil order P_min",
    color  = :black, lw=1.5, legend=false,
    ylims  = (0.5, 11),
    xlims  = (0, 40),
    titlefontsize=10, guidefontsize=9, tickfontsize=8,
    left_margin=M, bottom_margin=M, top_margin=2mm, right_margin=M,
)
# Dashed vertical lines at r = 4, 8, 12, ..., 32 (where P steps)
for r_step in 4:4:36
    vline!(p_a, [Float64(r_step)], ls=:dash, color=:black, lw=0.5, alpha=0.5, label="")
end
# Annotated points
r_diamond = 5.0
P_diamond = compute_P_min(D_a.Dxx, D_a.Dyy, D_a.Dxy, r_diamond)
scatter!(p_a, [r_diamond], [P_diamond], marker=:diamond, color=:blue, ms=7, label="")
annotate!(p_a, r_diamond+0.8, P_diamond-0.45, text("P=$P_diamond", 8, :blue))

r_circle = 30.0
P_circle = compute_P_min(D_a.Dxx, D_a.Dyy, D_a.Dxy, r_circle)
scatter!(p_a, [r_circle], [P_circle], marker=:circle, color=:red, ms=7, label="")
annotate!(p_a, r_circle-12, P_circle+0.5, text("r=$r_circle, P=$P_circle", 8, :red))
vline!(p_a, [r_circle], ls=:dash, color=:red, lw=1.2, label="")

# ── Panel B : Geometric grid (α=0.12, x∈[2,150]) ────────────────────────────
# Build geometric x-cells + uniform y-cells manually so the y-grid doesn't also
# grow (which would push r_max far beyond 67).  The per-cell P selection in
# build_diffusion_matrix handles the variable r column by column.
println("\n=== Panel B: geometric grid α=0.12, x∈[2,150] ===")

alpha_b = 0.12
Nx_b    = 29
xmin_b, xmax_b = 2.0, 150.0
dx0_b = (xmax_b - xmin_b) * alpha_b / ((1 + alpha_b)^Nx_b - 1)
dx_cells_b = [dx0_b * (1 + alpha_b)^(i-1) for i in 1:Nx_b]
x_b = zeros(Nx_b)
let pos = xmin_b
    for i in 1:Nx_b
        x_b[i] = pos + dx_cells_b[i]/2
        pos += dx_cells_b[i]
    end
end

# Uniform y-cells: dy = dx0_b/3 so base r ≈ 3 near x=2
dy_b = dx0_b / 3.0
Ny_b = 11
y_b  = [(j-0.5)*dy_b for j in 1:Ny_b]
geo_grid = Grid(Nx_b, Ny_b, x_b, y_b, dx0_b, dy_b, dx0_b/dy_b, alpha_b,
                dx_cells_b, fill(dy_b, Ny_b))

r_lo_b = dx_cells_b[1]   / dy_b
r_hi_b = dx_cells_b[end] / dy_b
@printf("  Nx=%d Ny=%d  r_min=%.1f  r_max=%.1f\n", Nx_b, Ny_b, r_lo_b, r_hi_b)

# C10=0.1, C11=0.02 gives d1/d2≈5, so p1=r/6 → global P_min=ceil(r_max/6)=12.
# This matches the report's "global P_min=12" for this panel.
par_b = ModelParams(1.0, 0.10, 0.02, 0.0; r=geo_grid.r, sigma=0.9)
kin_b = compute_kinetics(geo_grid.x, par_b)
D_b   = compute_diffusion_tensor(kin_b.d1, kin_b.d2)
# global P_min (for information only — per-cell selection used in simulation)
P_b_global = compute_P_min(D_b.Dxx, D_b.Dyy, D_b.Dxy, geo_grid)
@printf("  global P_min=%d (per-cell selection used)\n", P_b_global)

# Centre Gaussian near x≈12 (low end of domain) so that the rapid cell-size
# growth (red dotted lines) is visible in the still-active region
U0_b = normalize_to_mass!(
    make_gaussian(geo_grid; cx=12.0, cy=dy_b*Ny_b/2, sigma=3.0), geo_grid)
# Pass global P_min; build_diffusion_matrix will use per-cell P for geometric grid
res_b = run_simulation(DiagonalMultiStencil(P_b_global), geo_grid, par_b, U0_b, 5.0;
                       fixed_mobile=true)
@printf("  min(U)=%+.3e  exploded=%s\n", minimum(res_b.U), res_b.exploded)

p_b = heatmap(geo_grid.x, geo_grid.y, clamp.(res_b.U, 0.0, Inf)',
    title  = "Adaptive grid α=$(alpha_b), r∈[$(round(Int,r_lo_b)), $(round(Int,r_hi_b))]",
    xlabel = "x (non-uniform)", ylabel = "y",
    color  = :viridis,
    titlefontsize=10, guidefontsize=9, tickfontsize=8,
    left_margin=M, bottom_margin=M, top_margin=2mm, right_margin=M,
)
for i in 3:3:geo_grid.Nx
    vline!(p_b, [geo_grid.x[i]], ls=:dot, color=:red, lw=0.8, label="")
end

# ── Panel C : Uniform r=30 grid ───────────────────────────────────────────────
println("\n=== Panel C: uniform r=30 ===")

# Enlarged domain (40×20) so the mass stays interior; IC centred in domain.
dx_c = 3.0; dy_c = 0.1
Nx_c = 40; Ny_c = 20
x_c  = [2.0 + (i-0.5)*dx_c for i in 1:Nx_c]
y_c  = [(j-0.5)*dy_c for j in 1:Ny_c]
uni_grid = Grid(Nx_c, Ny_c, x_c, y_c, dx_c, dy_c, dx_c/dy_c, 0.0,
                fill(dx_c, Nx_c), fill(dy_c, Ny_c))

par_c = ModelParams(1.0, 3.0, 1.0, 0.0; r=uni_grid.r, sigma=0.9)
kin_c = compute_kinetics(uni_grid.x, par_c)
D_c   = compute_diffusion_tensor(kin_c.d1, kin_c.d2)
P_c   = compute_P_min(D_c.Dxx, D_c.Dyy, D_c.Dxy, uni_grid.r)
@printf("  r=%.1f  P_min=%d\n", uni_grid.r, P_c)

cx_c = 0.5*(x_c[1] + x_c[end])   # centre of domain
cy_c = 0.5*(y_c[1] + y_c[end])
U0_c = normalize_to_mass!(
    make_gaussian(uni_grid; cx=cx_c, cy=cy_c, sigma=3.0), uni_grid)
T_c = 1.0
res_c = run_simulation(DiagonalMultiStencil(P_c), uni_grid, par_c, U0_c, T_c;
                       fixed_mobile=true)
@printf("  min(U)=%+.3e  exploded=%s\n", minimum(res_c.U), res_c.exploded)

p_c = heatmap(uni_grid.x, uni_grid.y, clamp.(res_c.U, 0.0, Inf)',
    title  = "r=$(round(Int,uni_grid.r)), P_min=$P_c, T=$(T_c)",
    xlabel = "x", ylabel = "y",
    color  = :viridis,
    titlefontsize=10, guidefontsize=9, tickfontsize=8,
    left_margin=M, bottom_margin=M, top_margin=2mm, right_margin=M,
)

fig = plot(p_a, p_b, p_c,
    layout = (1,3),
    size   = (1200, 380),
    plot_title = "")

savefig(fig, joinpath(OUTDIR, "fig_multi.pdf"))
savefig(fig, joinpath(OUTDIR, "fig_multi.png"))
mkpath(joinpath(@__DIR__, "../../docs/images"))
savefig(fig, joinpath(@__DIR__, "../../docs/images/fig_multi.pdf"))
println("Saved fig_multi")
