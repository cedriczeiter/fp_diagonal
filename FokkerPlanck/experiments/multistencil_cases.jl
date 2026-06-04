"""
Five experiments specific to the multi-stencil solver.

Case 1: P-sweep — run DiagonalMultiStencil(P) for P=1,...,P_min+1 on a
        strong-coupling case (C10=3, C11=1, ε=1 → P_min=2).  Shows min(U) as
        a function of P and the positivity threshold at P_min.  Also plots
        the Ax weight field (negative where P_min>1 is needed).

Case 2: Convergence with grid refinement — free-decay eigenfunction on [0,1]²
        with Dxy=0 (isotropic). Verifies O(h) spatial convergence of DiagMS.

Case 3: Point-mass initial condition — near-singular tensor (C10=0.1, C11=1,
        ε=1 → r_upper≈1.1 << r=6 → P_min=6), narrow Gaussian (sigma=6% domain).
        DiagMS(P_min) stays non-negative; Naive9Point produces clear negative
        rings (Ax<<0 for P=1).  Also shows the Ax weight decomposition.

Case 4: Convergence with Dxy≠0 — exact anisotropic Gaussian dispersion solution.
        Compares P=1 (solves a perturbed PDE outside the r-window) against
        P=P_min=2 (correct).  P=1 errors plateau; P=2 converges O(h).

Case 5: P_min map — for the physical model (C10=3) and varying mesh ratios r,
        shows the spatially-varying P requirement P_min(x).  Motivates why
        multi-stencil is needed on stretched or geometric grids.
"""

if !isdefined(Main, :FokkerPlanck)
    include(joinpath(@__DIR__, "../src/FokkerPlanck.jl"))
end
using .FokkerPlanck, LinearAlgebra, SparseArrays, Plots, Printf
import Plots: mm

const OUTDIR = joinpath(@__DIR__, "../output")
mkpath(OUTDIR)

# ─────────────────────────────────────────────────────────────────────────────
# Case 1: P-sweep + Ax/Ay weight fields
# ─────────────────────────────────────────────────────────────────────────────
println("=== Case 1: P-sweep + Ax/Ay weight fields ===")
grid_ps = make_grid(xmin=2.0, xmax=8.0, Nx=50, ymin=0.0, ymax=1.0, Ny=50)
par_ps  = ModelParams(1.0, 3.0, 1.0, 0.0; r=grid_ps.r, sigma=0.9)

kin_ps = compute_kinetics(grid_ps.x, par_ps)
D_ps   = compute_diffusion_tensor(kin_ps.d1, kin_ps.d2)
P_min  = compute_P_min(D_ps.Dxx, D_ps.Dyy, D_ps.Dxy, grid_ps.r)
@printf("  r=%.2f  P_min=%d  max(Dxy)=%.2e\n", grid_ps.r, P_min, maximum(D_ps.Dxy))

U0_ps = normalize_to_mass!(make_gaussian(grid_ps; sigma=0.5), grid_ps)
T_ps  = 0.08

M = 6mm
p_sweep = plot(xlabel="t", ylabel="min(U)", title="Case 1: P-sweep  (P_min=$P_min)",
               left_margin=M, bottom_margin=M)
for P in 1:P_min+1
    res = run_simulation(DiagonalMultiStencil(P), grid_ps, par_ps, U0_ps, T_ps;
                         fixed_mobile=true)
    label = P == P_min ? "P=$P (min)" : "P=$P"
    plot!(p_sweep, res.times, res.min_u_hist, label=label,
          lw=(P == P_min ? 2.5 : 1.0))
    @printf("  P=%d  min(U)=%+.3e  exploded=%s\n", P, minimum(res.U), res.exploded)
end
hline!(p_sweep, [0.0], ls=:dash, color=:black, lw=1, label="0")

# Ax field: shows where single-stencil fails (Ax < 0 → P_min > 1 needed)
Ax_single = [D_ps.Dxx[i] - grid_ps.r * D_ps.Dxy[i] / 2 for i in 1:grid_ps.Nx, j in 1:grid_ps.Ny]
Dxy_field = [D_ps.Dxy[i] for i in 1:grid_ps.Nx, j in 1:grid_ps.Ny]

p_ax = heatmap(grid_ps.x, grid_ps.y, Ax_single',
               title="Ax = Dxx − r·Dxy/2  (neg → P_min>1)",
               color=:RdBu, xlabel="x", titlefontsize=10,
               clims=(-maximum(abs.(Ax_single)), maximum(abs.(Ax_single))),
               left_margin=M, bottom_margin=M)
p_dxy = heatmap(grid_ps.x, grid_ps.y, Dxy_field',
                title="D_xy field", color=:viridis, xlabel="x",
                titlefontsize=10, left_margin=M, bottom_margin=M)

p1_panel = plot(p_sweep, p_ax, p_dxy, layout=(1,3), size=(1400,420),
                plot_title="Case 1: P-sweep")
savefig(p1_panel, joinpath(OUTDIR, "ms_case1_psweep.pdf"))
savefig(plot(p_sweep, size=(700,420)), joinpath(OUTDIR, "ms_case1_psweep_minU.pdf"))
println("  → ms_case1_psweep.pdf + ms_case1_psweep_minU.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Case 2: Convergence with grid refinement
#
# Free-decay eigenfunction on [0,1]²:
#   u(x,y,t) = exp(-λ·t)·cos(πx)·cos(πy),  λ = π²·(Dxx+Dyy)
# No source term needed — exact solution to ∂_t u = Dxx·∂_xx u + Dyy·∂_yy u.
# Uses dt = T_end/nsteps chosen so simulation lands at exactly T_end.
# ─────────────────────────────────────────────────────────────────────────────
println("\n=== Case 2: Grid-refinement convergence (Dxy=0) ===")

D_xx_val = 1.0; D_yy_val = 0.5; D_xy_val = 0.0
lambda_exact = (D_xx_val + D_yy_val) * π^2

u_exact_decay(x, y, t) = exp(-lambda_exact * t) * cos(π*x) * cos(π*y)

function run_free_decay(L, U0_mat, nsteps, T_end)
    N = size(U0_mat, 1); n = N*N
    dt = T_end / nsteps
    A = sparse(I, n, n) - dt * L
    u = zeros(n)
    for i in 1:N, j in 1:N; u[(i-1)*N+j] = U0_mat[i,j]; end
    for _ in 1:nsteps
        u = A \ u
    end
    U_out = zeros(N, N)
    for i in 1:N, j in 1:N; U_out[i,j] = u[(i-1)*N+j]; end
    return U_out
end

Ns   = [8, 16, 32, 64]
T_mf = 0.1
err_vals = Float64[]

for N in Ns
    dx    = 1.0 / N
    x_g   = [(i - 0.5)*dx for i in 1:N]
    y_g   = x_g
    nsteps = max(1, ceil(Int, T_mf / (0.5*dx)))
    U0_mf  = [u_exact_decay(x, y, 0.0) for x in x_g, y in y_g]

    g_mf    = Grid(N, N, x_g, y_g, dx, dx, 1.0, 0.0, fill(dx,N), fill(dx,N))
    Dxx_vec = fill(D_xx_val, N)
    Dyy_vec = fill(D_yy_val, N)
    Dxy_vec = fill(D_xy_val, N)

    L_ms = build_diffusion_matrix(DiagonalMultiStencil(1), Dxx_vec, Dyy_vec, Dxy_vec, g_mf)
    U_ms = run_free_decay(L_ms, copy(U0_mf), nsteps, T_mf)
    U_ref = [u_exact_decay(x, y, T_mf) for x in x_g, y in y_g]
    push!(err_vals, maximum(abs.(U_ms .- U_ref)))
    @printf("  N=%3d  h=%.4f  nsteps=%4d  err=%.3e\n", N, dx, nsteps, err_vals[end])
end

hs   = 1.0 ./ Ns
ref1 = err_vals[1] .* (hs ./ hs[1]).^1
ref2 = err_vals[1] .* (hs ./ hs[1]).^2

p_conv = plot(hs, err_vals, marker=:circle, label="DiagMS (Dxy=0)",
              xlabel="h", ylabel="max |u − u_exact|",
              title="Case 2: convergence (free decay, T=$(T_mf))",
              xscale=:log10, yscale=:log10,
              left_margin=M, bottom_margin=M)
plot!(p_conv, hs, ref1, ls=:dash, label="O(h)")
plot!(p_conv, hs, ref2, ls=:dot,  label="O(h²)")

savefig(plot(p_conv, size=(650,440)), joinpath(OUTDIR, "ms_case2_convergence.pdf"))
println("  → ms_case2_convergence.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Case 3: Point-mass initial condition + neg-part visualization + Ax/Ay fields
# ─────────────────────────────────────────────────────────────────────────────
println("\n=== Case 3: Point-mass initial condition ===")
grid_pt = make_grid(xmin=2.0, xmax=8.0, Nx=50, ymin=0.0, ymax=1.0, Ny=50)
par_pt  = ModelParams(1.0, 0.1, 1.0, 0.0; r=grid_pt.r, sigma=0.9)

kin_pt = compute_kinetics(grid_pt.x, par_pt)
D_pt   = compute_diffusion_tensor(kin_pt.d1, kin_pt.d2)
P_pt   = compute_P_min(D_pt.Dxx, D_pt.Dyy, D_pt.Dxy, grid_pt.r)
@printf("  r=%.2f  P_min=%d  max(Dxy)=%.2e\n", grid_pt.r, P_pt, maximum(D_pt.Dxy))

U0_pt = normalize_to_mass!(
    make_gaussian(grid_pt;
        cx = 0.5*(grid_pt.x[1]+grid_pt.x[end]),
        cy = 0.5*(grid_pt.y[1]+grid_pt.y[end]),
        sigma = 0.06*(grid_pt.x[end]-grid_pt.x[1])),
    grid_pt)
T_pt = 0.06

r_ms  = run_simulation(DiagonalMultiStencil(P_pt), grid_pt, par_pt, U0_pt, T_pt; fixed_mobile=true)
r_9pt = run_simulation(Naive9Point(),               grid_pt, par_pt, U0_pt, T_pt; fixed_mobile=true)

@printf("  DiagMS(P=%d)  min(U)=%+.3e\n", P_pt, minimum(r_ms.U))
@printf("  Naive9Point   min(U)=%+.3e\n", minimum(r_9pt.U))

peak_pt = max(maximum(U0_pt), 1e-16)
log_floor = peak_pt * 1e-6

neg_9pt   = max.(0.0, -r_9pt.U)
neg_ms    = max.(0.0, -r_ms.U)
worst_neg = max(maximum(neg_9pt), maximum(neg_ms), 1e-30)
lc = (log10(peak_pt * 1e-6), log10(max(worst_neg, peak_pt * 1e-5)))

p3a = heatmap(grid_pt.x, grid_pt.y, r_ms.U',  title="DiagMS(P=$P_pt) final U",
              color=:Blues, clims=(0, peak_pt), xlabel="x",
              left_margin=M, bottom_margin=M)
p3b = heatmap(grid_pt.x, grid_pt.y, r_9pt.U', title="Naive9Point final U",
              color=:RdBu, clims=(-peak_pt, peak_pt), xlabel="x",
              left_margin=M, bottom_margin=M)
p3c = plot(r_ms.times,  r_ms.min_u_hist,  label="DiagMS(P=$P_pt)",
           xlabel="t", ylabel="min(U)", title="Case 3: min(U) vs time",
           left_margin=M, bottom_margin=M)
plot!(p3c, r_9pt.times, r_9pt.min_u_hist, label="Naive9Point")
hline!(p3c, [0.0], ls=:dash, color=:black, lw=1, label="")
p3d_9pt = heatmap(grid_pt.x, grid_pt.y, log10.(max.(neg_9pt, log_floor))',
                  title=@sprintf("Naive9Point neg(log₁₀)  max=%.2e", maximum(neg_9pt)),
                  color=:Reds, clims=lc, xlabel="x", titlefontsize=10,
                  left_margin=M, bottom_margin=M)
p3d_ms  = heatmap(grid_pt.x, grid_pt.y, log10.(max.(neg_ms, log_floor))',
                  title=@sprintf("DiagMS(P=%d) neg(log₁₀)  max=%.2e",
                                 P_pt, maximum(neg_ms)),
                  color=:Reds, clims=lc, xlabel="x", titlefontsize=10,
                  left_margin=M, bottom_margin=M)

Ax_pt_field = [D_pt.Dxx[i] - grid_pt.r * D_pt.Dxy[i] / 2
               for i in 1:grid_pt.Nx, j in 1:grid_pt.Ny]
p3e = heatmap(grid_pt.x, grid_pt.y, Ax_pt_field',
              title="Ax = Dxx − r·Dxy/2  (neg → P_min>1)",
              color=:RdBu, xlabel="x",
              clims=(-maximum(abs.(Ax_pt_field)), maximum(abs.(Ax_pt_field))),
              left_margin=M, bottom_margin=M)

p3_panel = plot(p3a, p3b, p3c, p3d_9pt, p3d_ms, p3e,
                layout=(2,3), size=(1500,780),
                plot_title="Case 3: point-mass IC, P_min=$P_pt")
savefig(p3_panel, joinpath(OUTDIR, "ms_case3_point_mass.pdf"))
println("  → ms_case3_point_mass.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Case 4: Convergence with Dxy ≠ 0
#
# Exact solution: anisotropic Gaussian dispersion on ℝ²,
#   u(x,y,t) = 1/(4πt √det D) · exp(−Q(x−cx, y−cy) / 4t)
# where Q(ξ,η) = (Dyy ξ² − Dxy ξη + Dxx η²) / det D,
# and the diffusion tensor D = [[Dxx, Dxy/2],[Dxy/2, Dyy]].
#
# We choose Dxy=1.2 so that P_min=2 at r=1 (upper bound r_hi=2Dxx/Dxy=5/3<2).
# We then compare DiagMS(P=1) against DiagMS(P=2):
#   - P=1 clips Ax<0 to zero, solving a perturbed PDE → errors plateau
#   - P=2 solves the correct PDE → O(h) convergence
# ─────────────────────────────────────────────────────────────────────────────
println("\n=== Case 4: Convergence with Dxy ≠ 0 ===")

D_xx4 = 1.0; D_yy4 = 0.5; D_xy4 = 1.2
r4    = 1.0   # square grid (Nx=Ny)
P4    = compute_P_min(fill(D_xx4, 1), fill(D_yy4, 1), fill(D_xy4, 1), r4)
det_D4  = D_xx4 * D_yy4 - (D_xy4/2)^2
r_lo4   = D_xy4 / (2*D_yy4)
r_hi4   = 2*D_xx4 / D_xy4
@printf("  Tensor: Dxx=%.1f Dyy=%.1f Dxy=%.1f  r=%.1f  P_min=%d\n",
        D_xx4, D_yy4, D_xy4, r4, P4)
@printf("  P=1 window: [%.3f, %.3f]  (r=%.1f is outside → P=1 solves wrong PDE)\n",
        r_lo4, r_hi4, r4)

cx4 = 0.5; cy4 = 0.5    # center of [0,1]²
t0_4  = 0.008            # start time: Gaussian already formed, stays in domain
T_end4 = 0.020           # integration interval

# Q(ξ,η) = (Dyy ξ² − Dxy ξη + Dxx η²) / det D  is the exponent quadratic form
function u_exact4(x, y, t)
    ξ = x - cx4;  η = y - cy4
    Q = (D_yy4 * ξ^2 - D_xy4 * ξ * η + D_xx4 * η^2) / det_D4
    return exp(-Q / (4t)) / (4π * t * sqrt(det_D4))
end

Ns4       = [8, 16, 32, 64]
err_P1    = Float64[]
err_Pmin  = Float64[]

for N in Ns4
    h      = 1.0 / N
    x_g    = [(i - 0.5) * h for i in 1:N]
    y_g    = x_g
    nsteps = max(1, ceil(Int, T_end4 / (0.4 * h)))
    dt4    = T_end4 / nsteps

    U0_4  = [u_exact4(x_g[i], y_g[j], t0_4) for i in 1:N, j in 1:N]
    U_ref = [u_exact4(x_g[i], y_g[j], t0_4 + T_end4) for i in 1:N, j in 1:N]

    g4       = Grid(N, N, x_g, y_g, h, h, r4, 0.0, fill(h,N), fill(h,N))
    Dxx_vec4 = fill(D_xx4, N)
    Dyy_vec4 = fill(D_yy4, N)
    Dxy_vec4 = fill(D_xy4, N)

    for (P_test, err_list) in ((1, err_P1), (P4, err_Pmin))
        L4 = build_diffusion_matrix(DiagonalMultiStencil(P_test),
                                    Dxx_vec4, Dyy_vec4, Dxy_vec4, g4)
        A4 = sparse(I, N*N, N*N) - dt4 * L4
        u4 = vec(copy(U0_4))
        for _ in 1:nsteps
            u4 = A4 \ u4
        end
        push!(err_list, maximum(abs.(reshape(u4, N, N) .- U_ref)))
    end
    @printf("  N=%3d  h=%.4f  err(P=1)=%.3e  err(P=%d)=%.3e\n",
            N, h, err_P1[end], P4, err_Pmin[end])
end

hs4    = 1.0 ./ Ns4
ref1_4 = err_Pmin[1] .* (hs4 ./ hs4[1]).^1
ref2_4 = err_Pmin[1] .* (hs4 ./ hs4[1]).^2

p_conv_dxy = plot(hs4, err_P1,   marker=:square, label="P=1  (clips Ax<0 → wrong PDE)",
                  xlabel="h", ylabel="max |u − u_exact|",
                  title="Case 4: convergence, Dxy=$(D_xy4), P_min=$(P4), r=$(r4)",
                  xscale=:log10, yscale=:log10,
                  left_margin=M, bottom_margin=M)
plot!(p_conv_dxy, hs4, err_Pmin, marker=:circle,  label="P=$P4 (P_min)")
plot!(p_conv_dxy, hs4, ref1_4,   ls=:dash,  label="O(h)")
plot!(p_conv_dxy, hs4, ref2_4,   ls=:dot,   label="O(h²)")

savefig(plot(p_conv_dxy, size=(700, 460)), joinpath(OUTDIR, "ms_case4_convergence_dxy.pdf"))
println("  → ms_case4_convergence_dxy.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Case 5: P_min map for the physical model
#
# For the Becker-Döring kinetics (C10=3, C11=1, ε=1), compute the diffusion
# tensor D(x) = (Dxx(x), Dxy(x), Dyy(x)) along the x-axis.  For each
# candidate mesh ratio r, compute the pointwise P_min(x) required so that
# r lies inside the three-stencil admissibility window [D_xy/(2P·Dyy), 2P·Dxx/D_xy].
#
# Physically: r=1 always works with P=1, but stretched or geometric grids
# push r away from 1 and raise the P requirement — especially near x=x_star
# where D_xy/D_yy is large (high coupling).
# ─────────────────────────────────────────────────────────────────────────────
println("\n=== Case 5: P_min map for physical model ===")

Nx5  = 300
x5   = range(2.0, 8.0, length=Nx5) |> collect
par5 = ModelParams(1.0, 3.0, 1.0, 0.0; r=1.0, sigma=0.9)
kin5 = compute_kinetics(x5, par5)
D5   = compute_diffusion_tensor(kin5.d1, kin5.d2)

# P_min for a single cell at index i and mesh ratio r_test
pmin_cell(i, r_test) = compute_P_min([D5.Dxx[i]], [D5.Dyy[i]], [D5.Dxy[i]], r_test)

r_tests     = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
r_colors    = [:blue, :green, :orange, :red, :purple, :brown]

p_pmin = plot(xlabel="x", ylabel="P_min(x)",
              title="Case 5: required P_min along x  (C10=3, C11=1, ε=1)",
              left_margin=M, bottom_margin=M, ylims=(0.5, nothing))

for (r_test, col) in zip(r_tests, r_colors)
    P_min_x = [pmin_cell(i, r_test) for i in 1:Nx5]
    plot!(p_pmin, x5, P_min_x, label="r=$r_test", color=col, lw=1.5)
end

# Overlay Dxy/Dyy (the local anisotropy ratio ρ) on a twin axis to explain the peaks
rho_x = D5.Dxy ./ (2 .* D5.Dyy)   # = 1 + d1/d2 − 1 = r_lo; P_min grows when r > 1+ρ
p_rho = plot(x5, rho_x, color=:black, ls=:dash, lw=1,
             label="D_xy/(2·Dyy)  (=r_lo)", alpha=0.6,
             xlabel="x", ylabel="r_lo(x) = D_xy/(2Dyy)", left_margin=M, bottom_margin=M)

p5_panel = plot(p_pmin, p_rho, layout=(2,1), size=(800, 600),
                plot_title="Case 5: P_min map")
savefig(p5_panel, joinpath(OUTDIR, "ms_case5_pmin_map.pdf"))

# Standalone P_min map
savefig(plot(p_pmin, size=(800, 460)), joinpath(OUTDIR, "ms_case5_pmin_map_only.pdf"))
println("  → ms_case5_pmin_map.pdf + ms_case5_pmin_map_only.pdf")

# Print peak P_min for each r
for r_test in r_tests
    P_peak = maximum(pmin_cell(i, r_test) for i in 1:Nx5)
    @printf("  r=%.1f  peak P_min=%d\n", r_test, P_peak)
end

# ─── Combined: one image with key plot from each case ─────────────────────────
p_ms_all = plot(p_sweep, p_conv, p3c, p_conv_dxy, p_pmin,
                layout=(1,5), size=(2400,420),
                plot_title="Multi-stencil cases overview")
savefig(p_ms_all, joinpath(OUTDIR, "ms_all_cases.pdf"))
println("  → ms_all_cases.pdf (combined)")

println("\nAll multi-stencil cases done.")
