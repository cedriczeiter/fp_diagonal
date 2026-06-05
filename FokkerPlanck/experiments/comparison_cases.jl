"""
Three solver comparison experiments.

Each case runs Naive9Point, DiagonalMultiStencil(P=1), DiagonalMultiStencil(P_min),
and Kurganov side-by-side.  For each case we save:
  - _snapshots.pdf  : IC panel + 4 final-state heatmaps
  - _min_u.pdf      : min(U) over time for all 4 solvers
  - _negpart.pdf    : log₁₀ of the negative part max(0,−U) per solver
  - _combined.pdf   : all panels on one image

Case 1: Near-axis-aligned (ε=0.01, D_xy≈0). control, all solvers agree.
Case 2: Moderate coupling (P_min=1). solvers agree, small differences in
        distribution shape and min(U).
Case 3: Near-singular tensor (C10=0.1, C11=1, ε=1, point-mass IC). Ax<0 for
        P=1, Naive9Point produces a clear negative halo; DiagMS(P=1) stagnates
        near zero; DiagMS(P_min) and Kurganov stay strictly positive.
"""

if !isdefined(Main, :FokkerPlanck)
    include(joinpath(@__DIR__, "../src/FokkerPlanck.jl"))
    using .FokkerPlanck
end
using Plots, Printf
import Plots: mm

const OUTDIR = joinpath(@__DIR__, "../output")
mkpath(OUTDIR)

function run_case(label, params, grid, U0, T_end; fixed_mobile=false)
    kin = compute_kinetics(grid.x, params)
    D   = compute_diffusion_tensor(kin.d1, kin.d2)
    P   = compute_P_min(D.Dxx, D.Dyy, D.Dxy, grid.r)

    @printf("\n=== %s  (P_min=%d, r=%.2f) ===\n", label, P, grid.r)
    @printf("  max Dxx=%.2e  Dxy=%.2e  Dyy=%.2e\n",
            maximum(D.Dxx), maximum(D.Dxy), maximum(D.Dyy))

    solvers = [
        ("Naive9Point",        Naive9Point()),
        ("DiagMS(P=1)",        DiagonalMultiStencil(1)),
        ("DiagMS(P=$P)",       DiagonalMultiStencil(P)),
        ("Kurganov",           Kurganov()),
    ]
    results = []
    for (name, solver) in solvers
        res = run_simulation(solver, grid, params, U0, T_end; fixed_mobile=fixed_mobile)
        @printf("  %-22s  min(U)=%+.3e  steps=%4d  exploded=%s\n",
                name, minimum(res.U), length(res.times)-1, res.exploded)
        push!(results, (name=name, res=res))
    end
    return results, P
end

# ─── Plot helper ─────────────────────────────────────────────────────────────
function comparison_plot(results, grid, U0, label, fname)
    colors = [:blue, :red, :green, :orange]
    M = 5mm  # shared panel margin

    # Peak value across all final states (for shared colour scale)
    peak = max(maximum(U0),
               maximum(filter(isfinite, vcat([r.res.U[:] for r in results]...))),
               1e-16)

    # ── IC panel ──────────────────────────────────────────────────────────────
    p_ic = heatmap(grid.x, grid.y, U0', title="IC (t=0)",
                   color=:Blues, clims=(0, peak), xlabel="x",
                   titlefontsize=10, colorbar=false,
                   left_margin=M, bottom_margin=M, top_margin=M, right_margin=M)

    # ── Final-state heatmaps ──────────────────────────────────────────────────
    hmaps = [heatmap(grid.x, grid.y, r.res.U',
                     title=r.name, color=:Blues, clims=(0, peak),
                     xlabel="x", titlefontsize=10, colorbar=false,
                     left_margin=M, bottom_margin=M, top_margin=M, right_margin=M)
             for r in results]

    # ── min(U) over time ──────────────────────────────────────────────────────
    p_min = plot(title="min(U) — $label", xlabel="t", ylabel="min(U)",
                 legend=:bottomleft, titlefontsize=10,
                 left_margin=M, bottom_margin=M, top_margin=M, right_margin=M)
    for (i, r) in enumerate(results)
        plot!(p_min, r.res.times, r.res.min_u_hist, label=r.name, color=colors[i])
    end
    hline!(p_min, [0.0], ls=:dash, color=:black, lw=1, label="")

    # ── Negative-part panels: log₁₀(max(0, −U)) ──────────────────────────────
    worst_neg = max(maximum(maximum(max.(0.0, -r.res.U)) for r in results), 1e-30)
    log_floor = worst_neg * 1e-6
    log_clims = (log10(log_floor), log10(max(worst_neg, log_floor * 10)))

    neg_panels = [begin
        neg_u  = max.(0.0, -r.res.U)
        log_ng = log10.(max.(neg_u, log_floor))
        max_neg = maximum(neg_u)
        heatmap(grid.x, grid.y, log_ng',
                title=@sprintf("%s  neg(log₁₀)  max=%.2e", r.name, max_neg),
                color=:Reds, clims=log_clims,
                xlabel="x", titlefontsize=9, colorbar=false,
                left_margin=M, bottom_margin=M, top_margin=M, right_margin=M)
    end for r in results]

    # ── Save individual PDFs ──────────────────────────────────────────────────
    savefig(plot(p_ic, hmaps..., layout=(1,5), size=(1800,380),
                 plot_title="$label — final state"),
            joinpath(OUTDIR, "$(fname)_snapshots.pdf"))

    savefig(plot(p_min, size=(700,420)),
            joinpath(OUTDIR, "$(fname)_min_u.pdf"))

    savefig(plot(neg_panels..., layout=(2,2), size=(1000,760),
                 plot_title="$label — neg part log₁₀"),
            joinpath(OUTDIR, "$(fname)_negpart.pdf"))

    # ── Combined: snapshots row + min_u + neg panels row ─────────────────────
    p_combined = plot(p_ic, hmaps..., p_min, neg_panels...,
                      layout=(2,5), size=(2100,760),
                      plot_title=label)
    savefig(p_combined, joinpath(OUTDIR, "$(fname)_combined.pdf"))
    println("  → saved $(fname)_*.pdf")
    return p_min, p_combined
end

# ─── Case 1: Near-axis-aligned (D_xy ≈ 0) ───────────────────────────────────
grid1 = make_grid(xmin=2.0, xmax=8.0, Nx=40, ymin=0.0, ymax=1.0, Ny=40)
p1    = ModelParams(0.01, 3.0, 1.0, 0.0; r=grid1.r, sigma=0.9)
U01   = normalize_to_mass!(make_gaussian(grid1; sigma=1.5), grid1)
res1, _ = run_case("Case 1: Near-axis-aligned (ε=0.01)", p1, grid1, U01, 0.3;
                   fixed_mobile=true)

# ─── Case 2: Moderate coupling (P_min = 1) ───────────────────────────────────
grid2 = make_grid(xmin=2.0, xmax=8.0, Nx=40, ymin=0.0, ymax=1.0, Ny=40)
p2    = ModelParams(0.4, 8.0, 3.0, 0.0; r=grid2.r, sigma=0.9)
U02   = normalize_to_mass!(make_gaussian(grid2; sigma=1.5), grid2)
res2, _ = run_case("Case 2: Moderate coupling (ε=0.4, P_min=1)", p2, grid2, U02, 0.3;
                   fixed_mobile=true)

# ─── Case 3: Near-singular tensor → Naive9Point clearly negative ─────────────
# C10=0.1, C11=1, ε=1 → r_upper≈1.1 << r=6 → P_min large, Ax << 0 for P=1.
# Tiny C10 keeps advection velocity small (no blow-up).  Point-mass IC maximises
# gradients so the M-matrix violation in Naive9Point appears immediately.
grid3 = make_grid(xmin=2.0, xmax=8.0, Nx=50, ymin=0.0, ymax=1.0, Ny=50)
p3    = ModelParams(1.0, 0.1, 1.0, 0.0; r=grid3.r, sigma=0.9)
# Narrow Gaussian: sigma ≈ 1.5 grid cells in x
U03   = normalize_to_mass!(
    make_gaussian(grid3;
        cx = 0.5*(grid3.x[1]+grid3.x[end]),
        cy = 0.5*(grid3.y[1]+grid3.y[end]),
        sigma = 0.06*(grid3.x[end]-grid3.x[1])),
    grid3)
res3, P3 = run_case("Case 3: Near-singular (C10=0.1, C11=1, P_min=$( begin
    kin = compute_kinetics(grid3.x, p3)
    D   = compute_diffusion_tensor(kin.d1, kin.d2)
    compute_P_min(D.Dxx, D.Dyy, D.Dxy, grid3.r)
end ))", p3, grid3, U03, 0.2; fixed_mobile=true)

# ─── Plots ───────────────────────────────────────────────────────────────────
pm1, pc1 = comparison_plot(res1, grid1, U01, "Case 1: axis-aligned",     "comp_case1")
pm2, pc2 = comparison_plot(res2, grid2, U02, "Case 2: moderate coupling", "comp_case2")
pm3, pc3 = comparison_plot(res3, grid3, U03, "Case 3: near-singular",     "comp_case3")

# Combined overview: 3 min_u panels stacked
p_overview = plot(pm1, pm2, pm3, layout=(3,1), size=(800,1050),
                  plot_title="Comparison cases — min(U) overview")
savefig(p_overview, joinpath(OUTDIR, "comp_all_cases.pdf"))
println("  → comp_all_cases.pdf (combined overview)")

println("\nAll comparison cases done.")
