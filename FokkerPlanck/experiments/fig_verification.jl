"""
Figure 2 — Grid convergence: L¹ error vs mesh size for three discretisations.

Dxx=Dyy=1, Dxy=1 (r=1 square grid, admissibility window [0.5, 2]).
Gaussian IC at t0=0.01 → exact anisotropic Gaussian at T=0.11. Zero drift.
dt = 0.25h² for all methods (implicit methods use backward-Euler in diffusion;
KT uses explicit SSP-RK3 with the same dt, which equals the diffusion CFL).

Domain [0,10]², N cells per side.  Grid struct built directly to bypass the
xmin≥2 guard in make_grid (which only applies to the physical kinetics model).
"""

if !isdefined(Main, :FokkerPlanck)
    include(joinpath(@__DIR__, "../src/FokkerPlanck.jl"))
end
using .FokkerPlanck, LinearAlgebra, SparseArrays, Plots, Printf
import Plots: mm

const OUTDIR = joinpath(@__DIR__, "../output")
mkpath(OUTDIR)

# Constant diffusion tensor: Dxx=Dyy=1, Dxy=1, r=1 (square cells)
const DXX, DYY, DXY = 1.0, 1.0, 1.0

# Exact solution: Gaussian with covariance Σ(t) = Σ(t0) + 2·D·(t−t0)
# D = [[Dxx, Dxy/2],[Dxy/2, Dyy]]
const L_DOM = 10.0
const CX, CY, SIGMA0 = 5.0, 5.0, 1.0
const T0, T_END = 0.01, 0.11

function exact_gaussian(x, y, t)
    dt = t - T0
    s11 = SIGMA0^2 + 2*DXX*dt
    s12 = DXY*dt
    s22 = SIGMA0^2 + 2*DYY*dt
    det_s = s11*s22 - s12^2
    dx = x - CX; dy = y - CY
    Q = (s22*dx^2 - 2*s12*dx*dy + s11*dy^2) / det_s
    return exp(-0.5*Q) / (2π*sqrt(det_s))
end

# KT pure-diffusion RHS (explicit, zero advection, constant tensor)
function kt_diff_rhs!(rhs, U, N, h)
    fill!(rhs, 0.0)
    mm3(a,b,c) = (a>0&&b>0&&c>0) ? min(a,b,c) : (a<0&&b<0&&c<0) ? max(a,b,c) : 0.0
    ux = similar(U); uy = similar(U)
    for j in 1:N, i in 1:N
        ux[i,j] = i==1 ? (U[2,j]-U[1,j])/h : i==N ? (U[N,j]-U[N-1,j])/h :
                  mm3(U[i,j]-U[i-1,j], 0.5*(U[i+1,j]-U[i-1,j]), U[i+1,j]-U[i,j]) / h
    end
    for i in 1:N, j in 1:N
        uy[i,j] = j==1 ? (U[i,2]-U[i,1])/h : j==N ? (U[i,N]-U[i,N-1])/h :
                  mm3(U[i,j]-U[i,j-1], 0.5*(U[i,j+1]-U[i,j-1]), U[i,j+1]-U[i,j]) / h
    end
    for j in 1:N, i in 1:N-1
        Px = DXX*(U[i+1,j]-U[i,j])/h + 0.25*DXY*(uy[i,j]+uy[i+1,j])
        rhs[i,j] += Px/h; rhs[i+1,j] -= Px/h
    end
    for i in 1:N, j in 1:N-1
        Py = DYY*(U[i,j+1]-U[i,j])/h + 0.25*DXY*(ux[i,j]+ux[i,j+1])
        rhs[i,j] += Py/h; rhs[i,j+1] -= Py/h
    end
end

function ssprk3_step(U, dt, N, h)
    rhs = zeros(N, N)
    kt_diff_rhs!(rhs, U, N, h)
    U1 = U .+ dt.*rhs
    kt_diff_rhs!(rhs, U1, N, h)
    U2 = 0.75.*U .+ 0.25.*(U1 .+ dt.*rhs)
    kt_diff_rhs!(rhs, U2, N, h)
    return (1/3).*U .+ (2/3).*(U2 .+ dt.*rhs)
end

# ── Convergence loop ──────────────────────────────────────────────────────────
Ns = [16, 32, 64, 128, 256]

err_diag  = Float64[]
err_naive = Float64[]
err_kt    = Float64[]

for N in Ns
    h = L_DOM / N
    x_g = [(i-0.5)*h for i in 1:N]
    y_g = [(j-0.5)*h for j in 1:N]
    g   = Grid(N, N, x_g, y_g, h, h, 1.0, 0.0, fill(h,N), fill(h,N))

    Dxx_v = fill(DXX, N); Dyy_v = fill(DYY, N); Dxy_v = fill(DXY, N)

    dt     = 0.25 * h^2
    nsteps = ceil(Int, (T_END - T0) / dt)
    dt     = (T_END - T0) / nsteps

    U0    = [exact_gaussian(x_g[i], y_g[j], T0)    for i in 1:N, j in 1:N]
    U_ref = [exact_gaussian(x_g[i], y_g[j], T_END) for i in 1:N, j in 1:N]

    for (solver, err_list) in ((DiagonalMultiStencil(1), err_diag), (Naive9Point(), err_naive))
        L_mat = build_diffusion_matrix(solver, Dxx_v, Dyy_v, Dxy_v, g)
        A = sparse(I, N*N, N*N) - dt * L_mat
        u = _pack_rowmajor(U0)
        for _ in 1:nsteps; u = A \ u; end
        U_sol = similar(U0); _unpack_rowmajor!(U_sol, u)
        push!(err_list, h^2 * sum(abs.(U_sol .- U_ref)))
    end

    U_kt = copy(U0)
    for _ in 1:nsteps; U_kt = ssprk3_step(U_kt, dt, N, h); end
    push!(err_kt, h^2 * sum(abs.(U_kt .- U_ref)))

    @printf("  N=%3d  h=%.4f  DiagMS=%.2e  Naive9=%.2e  KT=%.2e\n",
            N, h, err_diag[end], err_naive[end], err_kt[end])
end

# ── Plot ──────────────────────────────────────────────────────────────────────
hs   = L_DOM ./ Ns
ref2 = err_diag[1] .* (hs ./ hs[1]).^2          # O(h²) aligned with DiagMS at coarsest
ref1 = err_kt[1]   .* (hs ./ hs[1]).^1          # O(h)  aligned with KT at coarsest

p = plot(
    title  = "Grid convergence — D_xx=D_yy=1, D_xy=1, t∈[$(T0), $(T_END)]",
    xlabel = "mesh size h",
    ylabel = "L¹ error (vs exact Gaussian)",
    xscale = :log10,
    yscale = :log10,
    legend = :topleft,
    size   = (680, 460),
    left_margin=8mm, bottom_margin=6mm, right_margin=6mm, top_margin=10mm,
    titlefontsize=11, guidefontsize=10, tickfontsize=9, legendfontsize=9,
)
plot!(p, hs, err_diag,  marker=:circle,    color=1, lw=2, label="DiagMS (P=1)")
plot!(p, hs, err_naive, marker=:square,    color=2, lw=2, label="Naive 9-point")
plot!(p, hs, err_kt,    marker=:diamond,   color=3, lw=2, label="Kurganov-Tadmor")
plot!(p, hs, ref2, ls=:dash,  color=:gray, lw=1.2, label="O(h²)")
plot!(p, hs, ref1, ls=:dot,   color=:gray, lw=1.2, label="O(h)")

savefig(p, joinpath(OUTDIR, "fig_verification.pdf"))
savefig(p, joinpath(OUTDIR, "fig_verification.png"))
mkpath(joinpath(@__DIR__, "../../docs/images"))
savefig(p, joinpath(@__DIR__, "../../docs/images/fig_verification.pdf"))
println("Saved fig_verification")
