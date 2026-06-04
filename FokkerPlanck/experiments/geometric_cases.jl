"""
Three experiments for the geometric (non-uniform) grid with DiagonalMultiStencil.

Case 1: α_dom(P) for P=2,3,4,6.
        For each P, runs α = 0.9·α_dom(P) (admissible → positive) and
        α = 1.1·α_dom(P) (inadmissible → admissibility guard fires).
        Also plots the theoretical α_max(P) side-by-side for comparison.

Case 2: P-sweep on a fixed geometric grid with α=0.02.
        Runs DiagonalMultiStencil(P) for P=1,...,P_min+1.  P<P_min fires the
        admissibility guard; P≥P_min stays strictly positive.

Case 3: α-sweep for fixed P=3.
        Sweeps α from 0 (uniform) up past α_dom(P=3)≈0.037.
        Shows that min(U) stays non-negative up to the limit and the guard
        fires for α > α_dom.
"""

if !isdefined(Main, :FokkerPlanck)
    include(joinpath(@__DIR__, "../src/FokkerPlanck.jl"))
end
using .FokkerPlanck, Plots, Printf
import Plots: mm

const OUTDIR = joinpath(@__DIR__, "../output")
mkpath(OUTDIR)

M = 6mm

# ── shared setup ──────────────────────────────────────────────────────────────
const NX, NY  = 20, 20
const EPSILON = 1.0
const C10_0   = 3.0
const C11_0   = 1.0
const T_END   = 0.05

# reference uniform grid to pin down ρ_min and r0
const G_REF = make_grid(xmin=2.0, xmax=8.0, Nx=NX, ymin=0.0, ymax=1.0, Ny=NY)
const P_REF = let
    p   = ModelParams(EPSILON, C10_0, C11_0, 0.0; r=G_REF.r, sigma=0.9)
    kin = compute_kinetics(G_REF.x, p)
    D   = compute_diffusion_tensor(kin.d1, kin.d2)
    compute_P_min(D.Dxx, D.Dyy, D.Dxy, G_REF.r)
end
const RHO_MIN = let
    p   = ModelParams(EPSILON, C10_0, C11_0, 0.0; r=G_REF.r, sigma=0.9)
    kin = compute_kinetics(G_REF.x, p)
    minimum(kin.d1 ./ max.(kin.d2, 1e-30))
end
const R0 = G_REF.r   # ≈ 6 for [2,8]×[0,1] with equal Nx/Ny

# α_dom: domain-specific admissible alpha for our grid (binding constraint is r_max ≤ P*(1+ρ))
alpha_dom(P) = (P * (1.0 + RHO_MIN) / R0)^(1.0 / (NX - 1)) - 1.0

@printf("Reference: r0=%.2f  ρ_min=%.2f  P_min(uniform)=%d\n", R0, RHO_MIN, P_REF)
@printf("α_dom(P) = (P·(1+ρ)/r0)^{1/(Nx-1)} − 1 = (P·%.1f/%.1f)^{1/%d} − 1\n",
        1+RHO_MIN, R0, NX-1)

# ─────────────────────────────────────────────────────────────────────────────
# Case 1: α_dom and α_max bounds for P = 2, 3, 4, 6
# ─────────────────────────────────────────────────────────────────────────────
println("\n=== Case 1: admissible α range per P ===")

Ps = [2, 3, 4, 6]
adom_vals = alpha_dom.(Ps)
amax_vals = [compute_alpha_max(P, RHO_MIN, NX, NY) for P in Ps]

p1_bounds = plot(xlabel="P", ylabel="α",
                 title="Case 1: α_dom (our grid, r0=$(round(R0,digits=1))) vs α_max (optimal r0)",
                 left_margin=M, bottom_margin=M, legend=:topleft)
plot!(p1_bounds, Ps, adom_vals, marker=:circle, label="α_dom (domain-specific)", lw=1.5)
plot!(p1_bounds, Ps, amax_vals, marker=:square, ls=:dash, label="α_max (optimal r0)", lw=1.5)

p1_minU = plot(xlabel="t", ylabel="min(U)",
               title="Case 1: min(U) for α = 0.9·α_dom(P)  [all valid]",
               legend=:bottomleft, left_margin=M, bottom_margin=M)

for (P, adom, amax) in zip(Ps, adom_vals, amax_vals)
    @printf("  P=%d  α_dom=%.4f  α_max(opt)=%.4f\n", P, adom, amax)

    # α = 0.9·α_dom → admissible, simulation must stay positive
    alpha_ok = 0.9 * adom
    g_ok  = make_grid(xmin=2.0, xmax=8.0, Nx=NX, ymin=0.0, ymax=1.0, Ny=NY, alpha=alpha_ok)
    par   = ModelParams(EPSILON, C10_0, C11_0, 0.0; r=g_ok.r, sigma=0.9)
    U0    = normalize_to_mass!(make_gaussian(g_ok), g_ok)
    res   = run_simulation(DiagonalMultiStencil(P), g_ok, par, U0, T_END; fixed_mobile=true)
    @printf("    α=%.4f (0.9·α_dom)  min(U)=%+.3e  exploded=%s\n",
            alpha_ok, minimum(res.U), res.exploded)
    plot!(p1_minU, res.times, res.min_u_hist, label="P=$P, α=$(round(alpha_ok,digits=4))")

    # α = 1.1·α_dom → inadmissible, guard must fire before any time step
    alpha_bad = 1.1 * adom
    g_bad = make_grid(xmin=2.0, xmax=8.0, Nx=NX, ymin=0.0, ymax=1.0, Ny=NY, alpha=alpha_bad)
    par_b = ModelParams(EPSILON, C10_0, C11_0, 0.0; r=g_bad.r, sigma=0.9)
    kin_b = compute_kinetics(g_bad.x, par_b)
    D_b   = compute_diffusion_tensor(kin_b.d1, kin_b.d2)
    guard_fired = false
    try
        build_diffusion_matrix(DiagonalMultiStencil(P), D_b.Dxx, D_b.Dyy, D_b.Dxy, g_bad)
    catch
        guard_fired = true
    end
    @printf("    α=%.4f (1.1·α_dom)  guard fired: %s\n", alpha_bad, guard_fired)
end
hline!(p1_minU, [0.0], ls=:dash, color=:black, lw=1, label="")

p1_panel = plot(p1_bounds, p1_minU, layout=(1,2), size=(1100,420),
                plot_title="Case 1: admissible α per P")
savefig(p1_panel, joinpath(OUTDIR, "geom_case1_alpha_bound.pdf"))
println("  → geom_case1_alpha_bound.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Case 2: P-sweep on a geometric grid with α=0.02
# (α_dom: P=2→0.017, P=3→0.037 so α=0.02 requires P=3+)
# ─────────────────────────────────────────────────────────────────────────────
println("\n=== Case 2: P-sweep on geometric grid with α=0.02 ===")

ALPHA2 = 0.02
g2  = make_grid(xmin=2.0, xmax=8.0, Nx=NX, ymin=0.0, ymax=1.0, Ny=NY, alpha=ALPHA2)
par2 = ModelParams(EPSILON, C10_0, C11_0, 0.0; r=g2.r, sigma=0.9)
kin2 = compute_kinetics(g2.x, par2)
D2   = compute_diffusion_tensor(kin2.d1, kin2.d2)
P2   = compute_P_min(D2.Dxx, D2.Dyy, D2.Dxy, g2)
@printf("  α=%.2f  r0=%.3f  P_min=%d\n", ALPHA2, g2.r, P2)
@printf("  (α_dom: P=2→%.3f, P=3→%.3f, P=%d→%.3f)\n",
        alpha_dom(2), alpha_dom(3), P2, alpha_dom(P2))

U02 = normalize_to_mass!(make_gaussian(g2), g2)

p2_minU = plot(xlabel="t", ylabel="min(U)",
               title="Case 2: P-sweep, α=$(ALPHA2), P_min=$P2",
               legend=:bottomleft, left_margin=M, bottom_margin=M)

for P in 1:P2+1
    if alpha_dom(P) < ALPHA2
        @printf("  P=%d  α_dom=%.4f < α=%.2f → guard fires (skip)\n",
                P, alpha_dom(P), ALPHA2)
        continue
    end
    res   = run_simulation(DiagonalMultiStencil(P), g2, par2, copy(U02), T_END;
                           fixed_mobile=true)
    label = P == P2 ? "P=$P (P_min)" : "P=$P"
    @printf("  P=%d  min(U)=%+.3e  exploded=%s\n", P, minimum(res.U), res.exploded)
    plot!(p2_minU, res.times, res.min_u_hist, label=label,
          lw=(P == P2 ? 2.5 : 1.2))
end
hline!(p2_minU, [0.0], ls=:dash, color=:black, lw=1, label="")

res2_best = run_simulation(DiagonalMultiStencil(P2), g2, par2, copy(U02), T_END;
                           fixed_mobile=true)
p2_hmap = heatmap(g2.x, g2.y, res2_best.U',
                  title="DiagMS(P=$P2) final state  (α=$(ALPHA2))",
                  color=:Blues, clims=(0, maximum(U02)),
                  xlabel="x", ylabel="y", left_margin=M, bottom_margin=M)

p2_panel = plot(p2_minU, p2_hmap, layout=(1,2), size=(1100,420),
                plot_title="Case 2: P-sweep on geometric grid")
savefig(p2_panel, joinpath(OUTDIR, "geom_case2_psweep.pdf"))
println("  → geom_case2_psweep.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Case 3: α-sweep for P=3  (α_dom≈0.037 on our grid)
# ─────────────────────────────────────────────────────────────────────────────
println("\n=== Case 3: α-sweep for P=3 ===")

P3    = 3
adom3 = alpha_dom(P3)
@printf("  P=%d  α_dom=%.4f  α_max(opt)=%.4f\n", P3, adom3, compute_alpha_max(P3, RHO_MIN, NX, NY))

# α = 0, 25%, 50%, 75%, 95%, 110% of α_dom
fracs  = [0.0, 0.25, 0.50, 0.75, 0.95, 1.10]
alphas = fracs .* adom3

p3_minU = plot(xlabel="t", ylabel="min(U)",
               title="Case 3: α-sweep, P=$P3, α_dom=$(round(adom3,digits=4))",
               legend=:bottomright, left_margin=M, bottom_margin=M)

final_hmaps = []
for (frac, alpha) in zip(fracs, alphas)
    g3   = make_grid(xmin=2.0, xmax=8.0, Nx=NX, ymin=0.0, ymax=1.0, Ny=NY, alpha=alpha)
    par3 = ModelParams(EPSILON, C10_0, C11_0, 0.0; r=g3.r, sigma=0.9)
    kin3 = compute_kinetics(g3.x, par3)
    D3   = compute_diffusion_tensor(kin3.d1, kin3.d2)

    guard_fired = false
    try
        build_diffusion_matrix(DiagonalMultiStencil(P3), D3.Dxx, D3.Dyy, D3.Dxy, g3)
    catch
        guard_fired = true
    end

    if guard_fired
        @printf("  α=%.4f (%.0f%%·α_dom)  guard fired — skipping run\n", alpha, 100*frac)
        continue
    end

    U03 = normalize_to_mass!(make_gaussian(g3), g3)
    res = run_simulation(DiagonalMultiStencil(P3), g3, par3, U03, T_END; fixed_mobile=true)
    @printf("  α=%.4f (%.0f%%·α_dom)  min(U)=%+.3e  steps=%d\n",
            alpha, 100*frac, minimum(res.U), length(res.times)-1)
    label = frac == 0.0 ? "α=0 (uniform)" : "α=$(round(alpha,digits=4)) ($(round(Int,100*frac))%)"
    plot!(p3_minU, res.times, res.min_u_hist, label=label)
    push!(final_hmaps,
          heatmap(g3.x, g3.y, res.U',
                  title="α=$(round(alpha,sigdigits=3))",
                  color=:Blues, clims=(0, maximum(U03)),
                  xlabel="x", titlefontsize=10, colorbar=false,
                  left_margin=M, bottom_margin=M))
end
hline!(p3_minU, [0.0], ls=:dash, color=:black, lw=1, label="")

n_hmaps = length(final_hmaps)
p3_panel = plot(p3_minU, final_hmaps...,
                layout=(1, 1 + n_hmaps), size=(400*(1 + n_hmaps), 420),
                plot_title="Case 3: α-sweep, P=$P3")
savefig(p3_panel, joinpath(OUTDIR, "geom_case3_alpha_sweep.pdf"))
println("  → geom_case3_alpha_sweep.pdf")

println("\nAll geometric cases done.")
