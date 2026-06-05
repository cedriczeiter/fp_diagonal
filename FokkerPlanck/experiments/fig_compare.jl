"""
Figure 3 — Positivity comparison on a cross-term stress test.

ε=1, C10=3, C11=1, G10=0. Domain [2,8]×[0,1], Nx=60, Ny=10 → r≈0.916.

Top row: C(·,T_snap=0.002) on a shared Blues scale [0, 0.2]; cells with C < 0
         drawn as solid red rectangles (scheme-agnostic, pixel-exact overlay).
Bottom:  global min C(·,t) over t∈[0,1] for all three schemes (linear).
"""

if !isdefined(Main, :FokkerPlanck)
    include(joinpath(@__DIR__, "../src/FokkerPlanck.jl"))
end
using .FokkerPlanck, Plots, Printf
import Plots: mm

const OUTDIR = joinpath(@__DIR__, "../output")
mkpath(OUTDIR)

# ── Setup ─────────────────────────────────────────────────────────────────────
grid   = make_grid(xmin=2.0, xmax=8.0, Nx=60, ymin=0.0, ymax=1.0, Ny=10)
params = ModelParams(1.0, 3.0, 1.0, 0.0; r=grid.r, sigma=0.9)

kin = compute_kinetics(grid.x, params)
D   = compute_diffusion_tensor(kin.d1, kin.d2)
P   = compute_P_min(D.Dxx, D.Dyy, D.Dxy, grid.r)
@printf("r=%.3f  P_min=%d\n", grid.r, P)

# Normalised to mass 10. Peak ≈160 at t=0 → violations ~-7e-3 matching report.
# (Mass-1 normalisation gives violations 10× too small; M₀=10 scales linearly.)
U0 = normalize_to_mass!(
    make_gaussian(grid; cx=0.5*(grid.x[1]+grid.x[end]),
                        cy=0.5*(grid.y[1]+grid.y[end]), sigma=0.1), grid, 10.0)

T_snap = 0.002
T_end  = 1.0

sol_colors = [:blue, :orange, :green]
solvers = [
    ("DiagMS (P=$P)",   DiagonalMultiStencil(P), sol_colors[1]),
    ("Naive 9-point",   Naive9Point(),            sol_colors[2]),
    ("Kurganov-Tadmor", Kurganov(),               sol_colors[3]),
]

snaps = []
hists = []

for (label, solver, col) in solvers
    r1 = run_simulation(solver, grid, params, copy(U0), T_snap; fixed_mobile=true)
    # History run as two segments so both use the small dt enforced by T_snap:
    #   segment A: t=0→T_snap  (dt capped at T_snap=0.003 → captures violations)
    #   segment B: t=T_snap→T_end  (continues from where A left off)
    r2a = run_simulation(solver, grid, params, copy(U0), T_snap;        fixed_mobile=true)
    r2b = run_simulation(solver, grid, params, copy(r2a.U), T_end - T_snap; fixed_mobile=true)
    t_hist   = vcat(r2a.times,          r2a.times[end] .+ r2b.times[2:end])
    minu_hist= vcat(r2a.min_u_hist,     r2b.min_u_hist[2:end])
    push!(snaps, (label=label, U=r1.U, color=col))
    push!(hists, (label=label, times=t_hist, min_u=minu_hist, color=col))
    @printf("  %-22s  min(T_snap)=%+.3e  min(T_end)=%+.3e\n",
            label, minimum(r1.U), minimum(minu_hist))
end

# ── Plots ─────────────────────────────────────────────────────────────────────
M = 4mm

# Blues scale from 0 → v_max; violation cells drawn as solid red rectangles.
v_max = 0.2
v_lim = (0.0, v_max)
@printf("  colorscale [0, %.2f]  (negatives drawn red)\n", v_max)

# Half cell widths for rectangle drawing
dx_h = (grid.x[2] - grid.x[1]) / 2
dy_h = (grid.y[2] - grid.y[1]) / 2

hm_panels = []
for s in snaps
    p = heatmap(grid.x, grid.y, max.(s.U', 0.0),
        title    = "$(s.label), t=$(T_snap)",
        color    = :Blues, clims=v_lim,
        colorbar = false,
        xlabel   = "x", ylabel = "y",
        titlefontsize=10, guidefontsize=9, tickfontsize=8,
        left_margin=M, bottom_margin=M, top_margin=2mm, right_margin=1mm,
    )
    # Draw a red filled rectangle for each negative cell
    for i in 1:grid.Nx, j in 1:grid.Ny
        if s.U[i, j] < 0
            x0, x1 = grid.x[i] - dx_h, grid.x[i] + dx_h
            y0, y1 = grid.y[j] - dy_h, grid.y[j] + dy_h
            plot!(p, [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                seriestype=:shape, fillcolor=:red, linecolor=:red,
                linewidth=0, label="")
        end
    end
    push!(hm_panels, p)
end

# Standalone colorbar for the blues (positive) range.
y_cb = collect(range(v_lim[1], v_lim[2], length=100))
p_cb = heatmap(
    [0.0], y_cb, fill(NaN32, 100, 1),
    color=:Blues, clims=v_lim, colorbar=true,
    framestyle=:none, ticks=nothing,
    left_margin=0mm, bottom_margin=M, top_margin=2mm, right_margin=12mm,
)

# min C(·,t) plot — linear scale
p_min = plot(
    title  = "Positivity — all schemes, t ∈ [0, $(T_end)]",
    xlabel = "t",
    ylabel = "min C(·,t)",
    legend = :bottomright,
    xlims  = (0.0, T_end),
    titlefontsize=11, guidefontsize=10, tickfontsize=9, legendfontsize=9,
    left_margin=M, bottom_margin=M, top_margin=2mm, right_margin=M,
)
hline!(p_min, [0.0], ls=:dash, color=:black, lw=0.8, label="")
for h in hists

    if occursin("DiagMS", h.label)
        plot!(p_min, h.times, min.(h.min_u, 0.0), label=h.label, color=h.color, lw=2.5)
        continue
    end

    plot!(p_min, h.times, min.(h.min_u, 0.0), label=h.label, color=h.color, lw=1.5)
end

print(hists)

# 3 equal heatmaps + colorbar-only panel + full-width bottom plot.
l = @layout [
    [a{0.285w} b{0.285w} c{0.285w} d{0.145w}]
    e
]
fig = plot(hm_panels[1], hm_panels[2], hm_panels[3], p_cb, p_min,
           layout=l, size=(1050, 640))

savefig(fig, joinpath(OUTDIR, "fig_compare.pdf"))
savefig(fig, joinpath(OUTDIR, "fig_compare.png"))
mkpath(joinpath(@__DIR__, "../../docs/images"))
savefig(fig, joinpath(@__DIR__, "../../docs/images/fig_compare.pdf"))
println("Saved fig_compare")
