"""
Figure 5 — Full model with source term (G_{1,0}=1e-19, T=200).

G10=1e-19, C10=0.10, C11=0.02, ε=1.  Domain [2,60]×[0,4], 100×8 grid.
Gaussian seed M₀=0.01 centred at (x,y)=(15,2).  DiagMS with auto P_min.
Mobile concentrations C10, C11 evolve freely via absorption/emission ODEs.

Top row:    five snapshots at t=0,5,50,100,200 on a shared scale fixed to
            the t=5 peak.
Bottom row: mobile concentrations | max cumulative mass drift | mean ⟨x⟩(t).
"""

if !isdefined(Main, :FokkerPlanck)
    include(joinpath(@__DIR__, "../src/FokkerPlanck.jl"))
end
using .FokkerPlanck, Plots, Printf
import Plots: mm

const OUTDIR = joinpath(@__DIR__, "../output")
mkpath(OUTDIR)

# ── Setup ─────────────────────────────────────────────────────────────────────
grid   = make_grid(xmin=2.0, xmax=60.0, Nx=100, ymin=0.0, ymax=4.0, Ny=8)
params = ModelParams(1.0, 0.10, 0.02, 1e-19; r=grid.r, sigma=0.9)

kin = compute_kinetics(grid.x, params)
D   = compute_diffusion_tensor(kin.d1, kin.d2)
P   = compute_P_min(D.Dxx, D.Dyy, D.Dxy, grid.r)
@printf("r=%.3f  P_min=%d\n", grid.r, P)

U0 = normalize_to_mass!(
    make_gaussian(grid; cx=15.0, cy=2.0, sigma=1.0), grid, 0.01)

solver     = DiagonalMultiStencil(P)
snap_times = [0.0, 5.0, 50.0, 100.0, 200.0]
snaps      = [copy(U0)]

C10_curr = params.C10; C11_curr = params.C11
U_curr   = copy(U0)
mass0    = total_mass(U0, grid)

times_all    = [0.0]
C10_all      = [C10_curr]
C11_all      = [C11_curr]
mass_max_all = [0.0]   # cumulative max relative drift
x_mean_all   = [sum(grid.x[i]*U0[i,j]*grid.dx_cells[i]*grid.dy_cells[j]
                    for i in 1:grid.Nx, j in 1:grid.Ny) / mass0]

let C10_curr=C10_curr, C11_curr=C11_curr, U_curr=U_curr, mass_max=0.0
    for k in 2:lastindex(snap_times)
        dt_seg = snap_times[k] - snap_times[k-1]
        p_seg  = ModelParams(params.epsilon, C10_curr, C11_curr, params.G10;
                             r=params.r, sigma=params.sigma)
        res = run_simulation(solver, grid, p_seg, U_curr, dt_seg)

        push!(snaps, copy(res.U))
        U_curr   = res.U; C10_curr = res.C10; C11_curr = res.C11

        t_start = snap_times[k-1]
        for (i, t_rel) in enumerate(res.times)
            t_abs = t_start + t_rel
            t_abs > times_all[end] || continue
            push!(times_all, t_abs)
            push!(C10_all, res.C10_hist[i])
            push!(C11_all, res.C11_hist[i])
            mass_max = max(mass_max, res.mass_rel_hist[i])
            push!(mass_max_all, mass_max)
        end

        m = total_mass(U_curr, grid)
        xm = sum(grid.x[i]*U_curr[i,j]*grid.dx_cells[i]*grid.dy_cells[j]
                 for i in 1:grid.Nx, j in 1:grid.Ny) / max(m, 1e-30)
        push!(x_mean_all, xm)

        @printf("  t=%-6.0f  C10=%.3e  C11=%.3e  min(U)=%+.3e  mass_err=%.1e\n",
                snap_times[k], C10_curr, C11_curr, minimum(res.U), res.mass_rel_hist[end])
    end
end

# ── Plots ─────────────────────────────────────────────────────────────────────
M = 3mm

peak = max(maximum(snaps[2]), 1e-30)   # fix scale to t=5 peak

snap_panels = []
for (k, S) in enumerate(snaps)
    p = heatmap(grid.x, grid.y, clamp.(S, 0.0, peak)',
        title    = "t=$(snap_times[k])",
        color    = :viridis,
        clims    = (0.0, peak),
        colorbar = false,
        xlabel   = "x",
        ylabel   = (k == 1 ? "y" : ""),
        titlefontsize=9, guidefontsize=8, tickfontsize=7,
        left_margin=M, bottom_margin=M, top_margin=2mm, right_margin=1mm,
    )
    push!(snap_panels, p)
end

# Standalone colorbar: NaN body is invisible; GR still renders the colorbar bar from clims.
z_cb = collect(range(0.0, peak, length=100))
p_cb = heatmap([0.0], z_cb, fill(NaN32, 100, 1),
    color=:viridis, clims=(0.0, peak), colorbar=true,
    framestyle=:none, ticks=nothing,
    left_margin=0mm, bottom_margin=M, top_margin=2mm, right_margin=12mm)

# Mobile concentrations
p_conc = plot(
    title="Mobile concentrations", xlabel="t", ylabel="Concentration",
    legend=:topright,
    titlefontsize=10, guidefontsize=9, tickfontsize=8, legendfontsize=8,
    left_margin=M, bottom_margin=M, top_margin=2mm, right_margin=M,
)
plot!(p_conc, times_all, C10_all, label="C₁₀(t)", color=1, lw=1.5)
plot!(p_conc, times_all, C11_all, label="C₁₁(t)", color=2, lw=1.5)

# Cumulative max mass drift
p_mass = plot(
    title="Mass conservation", xlabel="t",
    ylabel="cummax |ΔM/M₀|",
    yscale=:log10,
    legend=false,
    titlefontsize=10, guidefontsize=9, tickfontsize=8,
    left_margin=M, bottom_margin=M, top_margin=2mm, right_margin=M,
)
plot!(p_mass, times_all, max.(mass_max_all, 1e-18), color=:black, lw=1.5)

# Mean cluster x-size with markers at snapshot times
p_xmean = plot(
    snap_times, x_mean_all,
    title="Mean cluster x-size", xlabel="t", ylabel="⟨x⟩",
    marker=:diamond, color=4, lw=1.5,
    legend=false,
    titlefontsize=10, guidefontsize=9, tickfontsize=8,
    left_margin=M, bottom_margin=M, top_margin=2mm, right_margin=M,
)

# Flat layout: 5 equal heatmaps + narrow colorbar on top, 3 equal line plots below.
# Using @layout avoids the sub-figure nesting that squeezes the top-right corner.
# Last heatmap gets extra width so its body matches the others after colorbar overhead.
# 5 equal heatmaps + colorbar-only panel + 3 equal line plots.
l = @layout [
    [a{0.168w} b{0.168w} c{0.168w} d{0.168w} e{0.168w} f{0.16w}]
    [g h i]
]
fig = plot(snap_panels..., p_cb, p_conc, p_mass, p_xmean,
           layout=l, size=(1200, 560))

savefig(fig, joinpath(OUTDIR, "fig_full.pdf"))
savefig(fig, joinpath(OUTDIR, "fig_full.png"))
mkpath(joinpath(@__DIR__, "../../docs/images"))
savefig(fig, joinpath(@__DIR__, "../../docs/images/fig_full.pdf"))
println("Saved fig_full")
