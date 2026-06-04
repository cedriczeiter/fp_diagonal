"""
Three long-time horizon experiments with DiagonalMultiStencil.

Run 1: Free-mobile relaxation. no source, C10/C11 evolve freely.
        Gaussian spreads as mass transfers into clusters; verify mass conservation.

Run 2: Source-driven accumulation. G10 > 0 drives X_mat growing linearly in t.
        Compare X_mat(t) to the exact budget X0 + G10*t.

Run 3: Full two-species physics. both G10, G11 > 0, free mobile concentrations.
        C10, C11 evolve; mass grows; shows the coupled ODE dynamics.

Each run saves:
  - Individual diagnostics PDF
  - 4-snapshot evolution PDF (IC, T/3, 2T/3, T) in both linear and log₁₀ scale
Combined:
  - long_runs_all.pdf (key panels from all runs)
"""

if !isdefined(Main, :FokkerPlanck)
    include(joinpath(@__DIR__, "../src/FokkerPlanck.jl"))
end
using .FokkerPlanck, Plots, Printf
import Plots: mm

const OUTDIR = joinpath(@__DIR__, "../output")
mkpath(OUTDIR)

# ─── Helper: run in segments to collect snapshots ────────────────────────────
function run_with_snapshots(solver, grid, params0, U0, T_end, n_snaps=4)
    snap_times  = [T_end * k / (n_snaps - 1) for k in 0:(n_snaps-1)]
    snaps       = Matrix{Float64}[]
    U_curr      = copy(U0)
    C10_curr    = params0.C10
    C11_curr    = params0.C11
    push!(snaps, copy(U0))    # t = 0 snapshot

    for k in 2:n_snaps
        dt_seg = snap_times[k] - snap_times[k-1]
        p_seg  = ModelParams(params0.epsilon, C10_curr, C11_curr, params0.G10;
                             G11=params0.G11, r=params0.r, sigma=params0.sigma)
        r = run_simulation(solver, grid, p_seg, U_curr, dt_seg)
        push!(snaps, copy(r.U))
        U_curr   = r.U
        C10_curr = r.C10
        C11_curr = r.C11
    end
    return snaps, snap_times
end

function evolution_panels(snaps, snap_times, grid)
    peak = max(maximum(snaps[1]), 1e-16)
    M = 5mm
    lin_panels = [heatmap(grid.x, grid.y, clamp.(S, 0.0, peak)',
                          title=@sprintf("t=%.3g (lin)", snap_times[k]),
                          color=:Blues, clims=(0, peak),
                          xlabel="x", colorbar=false, titlefontsize=10,
                          left_margin=M, bottom_margin=M, top_margin=M, right_margin=M)
                  for (k, S) in enumerate(snaps)]
    log_panels = [heatmap(grid.x, grid.y,
                          log10.(max.(clamp.(S, 0.0, Inf), peak*1e-6))',
                          title=@sprintf("t=%.3g (log₁₀)", snap_times[k]),
                          color=:viridis,
                          clims=(log10(peak)-6, log10(peak)),
                          xlabel="x", colorbar=false, titlefontsize=10,
                          left_margin=M, bottom_margin=M, top_margin=M, right_margin=M)
                  for (k, S) in enumerate(snaps)]
    return lin_panels, log_panels
end

# ─── Run 1: Free-mobile relaxation ──────────────────────────────────────────
println("=== Run 1: Free-mobile relaxation (T_end=100) ===")
grid1   = make_grid(xmin=2.0, xmax=8.0, Nx=50, ymin=0.0, ymax=1.0, Ny=50)
params1 = ModelParams(0.5, 0.3, 0.1, 0.0; G11=0.0, r=grid1.r, sigma=0.9)

U01 = normalize_to_mass!(make_gaussian(grid1; cx=3.5, cy=0.4, sigma=0.6), grid1)
kin1 = compute_kinetics(grid1.x, params1)
D1   = compute_diffusion_tensor(kin1.d1, kin1.d2)
P1   = compute_P_min(D1.Dxx, D1.Dyy, D1.Dxy, grid1.r)
@printf("  P_min=%d, max(Dxy)=%.2e\n", P1, maximum(D1.Dxy))

r1 = run_simulation(DiagonalMultiStencil(P1), grid1, params1, U01, 100.0)
@printf("  t=%.3f  min(U)=%.3e  max rel mass drift=%.2e\n",
        r1.t, minimum(r1.U), maximum(r1.mass_rel_hist))

# Collect snapshots
snaps1, stimes1 = run_with_snapshots(DiagonalMultiStencil(P1), grid1, params1, U01, 100.0)
lin1, log1 = evolution_panels(snaps1, stimes1, grid1)

M = 6mm
p1a = plot(r1.times, max.(1e-16, r1.mass_rel_hist), xlabel="t", ylabel="|Δmass|/mass₀",
           title="Run 1: mass conservation", label="rel drift", yscale=:log10,
           left_margin=M, bottom_margin=M)
p1b = heatmap(grid1.x, grid1.y, r1.U', xlabel="x", ylabel="y",
              title="U at t=$(round(r1.t,digits=2))", color=:viridis,
              left_margin=M, bottom_margin=M)

p1_panel = plot(p1a, p1b, layout=(1,2), size=(1000,380))
savefig(p1_panel, joinpath(OUTDIR, "long_run1_equilibrium.pdf"))

p1_evo = plot(lin1..., log1..., layout=(2,4), size=(1600,700),
              plot_title="Run 1: free-mobile relaxation — evolution")
savefig(p1_evo, joinpath(OUTDIR, "long_run1_evolution.pdf"))
println("  → long_run1_equilibrium.pdf + long_run1_evolution.pdf")

# ─── Run 2: Source-driven X_mat growth ──────────────────────────────────────
println("\n=== Run 2: Source-driven growth (G10=0.2, T_end=100) ===")
grid2   = make_grid(xmin=2.0, xmax=8.0, Nx=50, ymin=0.0, ymax=1.0, Ny=50)
params2 = ModelParams(0.5, 0.5, 0.2, 0.2; G11=0.0, r=grid2.r, sigma=0.9)

U02 = normalize_to_mass!(make_gaussian(grid2), grid2, 0.5)
kin2 = compute_kinetics(grid2.x, params2)
D2   = compute_diffusion_tensor(kin2.d1, kin2.d2)
P2   = compute_P_min(D2.Dxx, D2.Dyy, D2.Dxy, grid2.r)

r2 = run_simulation(DiagonalMultiStencil(P2), grid2, params2, U02, 100.0)
@printf("  t=%.3f  C10=%.4f  min(U)=%.3e\n", r2.t, r2.C10, minimum(r2.U))

snaps2, stimes2 = run_with_snapshots(DiagonalMultiStencil(P2), grid2, params2, U02, 100.0)
lin2, log2 = evolution_panels(snaps2, stimes2, grid2)

p2a = plot(r2.times, max.(1e-16, r2.X_budget_rel_hist), xlabel="t",
           ylabel="|X_mat − (X₀+G10·t)| / X₀",
           title="Run 2: X_mat budget (rel. error)", label="", yscale=:log10,
           left_margin=M, bottom_margin=M)
p2b = plot(r2.times, r2.C10_hist, xlabel="t", ylabel="concentration",
           title="Run 2: mobile concentrations", label="C10",
           left_margin=M, bottom_margin=M)
plot!(p2b, r2.times, r2.C11_hist, label="C11")

p2_panel = plot(p2a, p2b, layout=(1,2), size=(1000,380))
savefig(p2_panel, joinpath(OUTDIR, "long_run2_source.pdf"))
p2_evo = plot(lin2..., log2..., layout=(2,4), size=(1600,700),
              plot_title="Run 2: source-driven growth — evolution")
savefig(p2_evo, joinpath(OUTDIR, "long_run2_evolution.pdf"))
println("  → long_run2_source.pdf + long_run2_evolution.pdf")

# ─── Run 3: Full two-species dynamics ────────────────────────────────────────
println("\n=== Run 3: Full two-species physics (T_end=100) ===")
grid3   = make_grid(xmin=2.0, xmax=8.0, Nx=50, ymin=0.0, ymax=1.0, Ny=50)
params3 = ModelParams(0.5, 2.0, 1.0, 0.15; G11=0.05, r=grid3.r, sigma=0.9)

U03 = normalize_to_mass!(make_gaussian(grid3; sigma=1.2), grid3, 0.3)
kin3 = compute_kinetics(grid3.x, params3)
D3   = compute_diffusion_tensor(kin3.d1, kin3.d2)
P3   = compute_P_min(D3.Dxx, D3.Dyy, D3.Dxy, grid3.r)
@printf("  P_min=%d\n", P3)

r3 = run_simulation(DiagonalMultiStencil(P3), grid3, params3, U03, 100.0)
@printf("  t=%.3f  C10=%.4f  C11=%.4f  min(U)=%.3e\n",
        r3.t, r3.C10, r3.C11, minimum(r3.U))

snaps3, stimes3 = run_with_snapshots(DiagonalMultiStencil(P3), grid3, params3, U03, 100.0)
lin3, log3 = evolution_panels(snaps3, stimes3, grid3)

p3a = plot(r3.times, r3.C10_hist, xlabel="t", label="C10",
           title="Run 3: species concentrations",
           left_margin=M, bottom_margin=M)
plot!(p3a, r3.times, r3.C11_hist, label="C11")
p3b = plot(r3.times, max.(1e-16, r3.mass_rel_hist), xlabel="t",
           ylabel="|Δmass|/mass₀", title="Run 3: mass conservation",
           label="", yscale=:log10,
           left_margin=M, bottom_margin=M)
p3c = heatmap(grid3.x, grid3.y, r3.U', xlabel="x", ylabel="y",
              title="U at t=$(round(r3.t,digits=2))", color=:viridis,
              left_margin=M, bottom_margin=M)

p3_panel = plot(p3a, p3b, p3c, layout=(1,3), size=(1300,400))
savefig(p3_panel, joinpath(OUTDIR, "long_run3_two_species.pdf"))
p3_evo = plot(lin3..., log3..., layout=(2,4), size=(1600,700),
              plot_title="Run 3: two-species dynamics — evolution")
savefig(p3_evo, joinpath(OUTDIR, "long_run3_evolution.pdf"))
println("  → long_run3_two_species.pdf + long_run3_evolution.pdf")

# ─── Combined ─────────────────────────────────────────────────────────────────
p_long_all = plot(p1a, p1b, p2a, p2b, p3a, p3b,
                  layout=(3,2), size=(1200,1100),
                  plot_title="Long runs overview")
savefig(p_long_all, joinpath(OUTDIR, "long_runs_all.pdf"))
println("  → long_runs_all.pdf (combined)")

println("\nAll long runs done.")
