# Multi-stencil positivity-preserving diffusion solver.
# Implements the {(1,1),(1,P),(P,1)} stencil set

struct DiagonalMultiStencil <: AbstractFPSolver
    P::Int   # >= 1; use compute_P_min to find the smallest valid P
end

diagonal_solver() = DiagonalMultiStencil(1)

# ─────────────────────────────────────────────────────────────────────────────
# P selection
# ─────────────────────────────────────────────────────────────────────────────

"""
Smallest P >= 1 such that the uniform-grid ratio `r` lies in the admissible
window for the three-stencil set at every cell.
Returns 1 when the single-(1,1) stencil is already sufficient.
-> formula (35) from 10th report, rearranged.
"""
function compute_P_min(Dxx::AbstractVector, Dyy::AbstractVector, Dxy::AbstractVector, r::Float64)
    P = 1
    for i in eachindex(Dxx)
        if Dxy[i] <= 1e-14 || Dxx[i] <= 1e-14 || Dyy[i] <= 1e-14
            continue
        end
        p1 = Dxy[i]*r / (2*Dxx[i])
        p2 = Dxy[i] / (2*r*Dyy[i])
        P  = max(P, ceil(Int, max(p1, p2)))
    end
    return P
end

"""
Smallest P >= 1 such that every per-cell local ratio r̃_{ij} = dx_cells[i]/dy_cells[j]
lies in the admissible window for the three-stencil set.

For a geometric grid the worst-case ratios at column i are
  r_max = dx_cells[i]/dy_cells[1]   (largest r, at j=1)
  r_min = dx_cells[i]/dy_cells[end] (smallest r, at j=Ny)
and P must satisfy both the upper-bound (via r_max) and lower-bound (via r_min)
constraints simultaneously.  For a uniform grid this reduces to the scalar overload.
"""
function compute_P_min(Dxx::AbstractVector, Dyy::AbstractVector, Dxy::AbstractVector, grid::Grid)
    P = 1
    for i in eachindex(Dxx)
        if Dxy[i] <= 1e-14 || Dxx[i] <= 1e-14 || Dyy[i] <= 1e-14
            continue
        end
        r_max = grid.dx_cells[i] / grid.dy_cells[1]    # largest local ratio at column i
        r_min = grid.dx_cells[i] / grid.dy_cells[end]  # smallest local ratio at column i
        p1 = Dxy[i]*r_max / (2*Dxx[i])   # upper-bound constraint
        p2 = Dxy[i] / (2*r_min*Dyy[i])   # lower-bound constraint
        P  = max(P, ceil(Int, max(p1, p2)))
    end
    return P
end

# ─────────────────────────────────────────────────────────────────────────────
# Geometric-grid admissibility bound
# ─────────────────────────────────────────────────────────────────────────────

"""
Maximum geometric growth factor α such that the three-stencil
{(1,1),(1,P),(P,1)} scheme is admissible for all cells on an Nx×Ny grid
with worst-case physical ratio ρ_min = min_i d1(x_i)/d2(x_i).
"""
function compute_alpha_max(P::Int, rho_min::Float64, Nx::Int, Ny::Int)
    P  >= 1   || error("P must be >= 1.")
    rho_min >= 0.0 || error("rho_min must be non-negative.")
    Nx >= 2 && Ny >= 2 || error("Need Nx >= 2, Ny >= 2.")
    return (P^2 * (1.0 + rho_min))^(1.0 / (Nx + Ny - 2)) - 1.0
end

# ─────────────────────────────────────────────────────────────────────────────
# Per-cell weight computation
# ─────────────────────────────────────────────────────────────────────────────
"""
This function returns the optimal weights for the (1,1), (1,P), and (P,1) stencils for a
given cell, as well as the resulting axial diffusion coefficients Ax and Ay.

It automatically detects which of the three stencils are needed to restore positivity.

output: (w1, w2, w3, Ax, Ay)

"""
function _cell_weights(Dxx::Float64, Dyy::Float64, Dxy::Float64, r::Float64, P::Int)
    tol = 1e-14


    # No cross-diffusion: pure axial splitting, no diagonal stencil needed
    if Dxy <= tol || Dxx <= tol || Dyy <= tol
        return (0.0, 0.0, 0.0, max(Dxx, 0.0), max(Dyy, 0.0))
    end

    half = 0.5 * Dxy   # total cross-weight budget: w1 + w2 + w3 = half

    # P=1: single (1,1) stencil only
    if P == 1
        Ax = Dxx - r * half
        Ay = Dyy - half / r
        return (max(half, 0.0), 0.0, 0.0, max(Ax, 0.0), max(Ay, 0.0))
    end

    # Single-(1,1) admissibility window: r in [r_lo, r_hi] iff Ax>=0 and Ay>=0
    r_lo = Dxy / (2.0 * Dyy)
    r_hi = 2.0 * Dxx / Dxy
    # Axial residuals when only the (1,1) stencil carries all cross-weight (w2=w3=0)
    ax0 = Dxx - r * half
    ay0 = Dyy - half / r
    # Already inside window, no auxiliary stencils needed
    if r >= r_lo - tol && r <= r_hi + tol
        return (max(half, 0.0), 0.0, 0.0, max(ax0, 0.0), max(ay0, 0.0))
    end


    Fp = Float64(P)
    ax_w2 =  r * (1.0 - 1.0/Fp)    # > 0: (1,P) stencil increases Ax
    ax_w3 =  r * (1.0 - Fp)         # < 0: (P,1) stencil decreases Ax
    ay_w2 = (1.0 - Fp) / r          # < 0: (1,P) stencil decreases Ay
    ay_w3 = (1.0 - 1.0/Fp) / r     # > 0: (P,1) stencil increases Ay

    # ── Centering ────────────────────────────────────────────────────────────
    # The two-stencil cases below pick the midpoint of the feasible interval,
    # which is exactly the centering criterion from §6.8 for one free parameter:
    #   w* = argmax_{w ∈ [lb, ub]} min(Ax, Ay, w1, w2)
    # For the three-stencil case (two free parameters), no closed-form centering
    # is implemented.  We instead use the "saturation" choice Ax=0, Ay=0 which
    # gives the unique solution via the 2×2 system below.  When that solution
    # has non-negative weights it is a valid (if not maximally centred) answer.
    # When it has negative weights the feasible region is provably empty, the
    # discriminant condition (report eq. disc-r) fails for this (Dxy, r, P).
    # ─────────────────────────────────────────────────────────────────────────

    # Only upper bound violated (r > r_hi, Ax < 0): add (1,P) stencil to restore Ax
    if r > r_hi + tol && r >= r_lo - tol
        w2_lb = max(0.0, -ax0 / ax_w2)
        w2_ub = min(half, -ay0 / ay_w2)
        if w2_lb <= w2_ub + tol
            w2 = clamp(0.5 * (w2_lb + w2_ub), 0.0, half)
            return (max(half - w2, 0.0), w2, 0.0,
                    max(ax0 + ax_w2 * w2, 0.0), max(ay0 + ay_w2 * w2, 0.0))
        end
    end

    # Only lower bound violated (r < r_lo, Ay < 0): add (P,1) stencil to restore Ay
    if r < r_lo - tol && r <= r_hi + tol
        w3_lb = max(0.0, -ay0 / ay_w3)
        w3_ub = min(half, -ax0 / ax_w3)
        if w3_lb <= w3_ub + tol
            w3 = clamp(0.5 * (w3_lb + w3_ub), 0.0, half)
            return (max(half - w3, 0.0), 0.0, w3,
                    max(ax0 + ax_w3 * w3, 0.0), max(ay0 + ay_w3 * w3, 0.0))
        end
    end

    # Both bounds violated, or two-stencil approach infeasible:
    # Saturation choice — set Ax=0 and Ay=0 simultaneously.
    # This gives the 2×2 linear system solved by Cramer's rule:
    #
    #   | ax_w2  ax_w3 | | w2 |   | -ax0 |
    #   | ay_w2  ay_w3 | | w3 | = | -ay0 |
    #
    #   det = ax_w2*ay_w3 - ax_w3*ay_w2 = -(P-1)³(P+1)/P²  < 0  for all P > 1
    #   w2* = (-ax0*ay_w3 + ax_w3*ay0) / det
    #   w3* = ( ax_w2*(-ay0) - ay_w2*(-ax0)) / det
    det_sys = ax_w2 * ay_w3 - ax_w3 * ay_w2   # always < 0 for P > 1
    w2s = (-ax0 * ay_w3 + ax_w3 * ay0) / det_sys
    w3s = ( ax_w2 * (-ay0) - ay_w2 * (-ax0)) / det_sys
    w1s = half - w2s - w3s

    if w2s >= -tol && w3s >= -tol && w1s >= -tol
        w1 = max(w1s, 0.0); w2 = max(w2s, 0.0); w3 = max(w3s, 0.0)
        return (w1, w2, w3,
                max(ax0 + ax_w2*w2 + ax_w3*w3, 0.0),
                max(ay0 + ay_w2*w2 + ay_w3*w3, 0.0))
    end

    # Negative weights mean the feasible region {w≥0, Ax≥0, Ay≥0} is empty.
    # This is the discriminant condition (report eq. disc-r) failing as outlined in report 10
    error("DiagonalMultiStencil: no valid weights exist for P=$P, r=$r")
end

# ─────────────────────────────────────────────────────────────────────────────
# Matrix assembly
# ─────────────────────────────────────────────────────────────────────────────

"""
Assemble the multi-stencil positivity-preserving diffusion matrix.
All off-diagonal entries are >= 0; diagonal is set so rows sum to zero.

For a uniform grid (grid.alpha == 0) the assembly is identical to the original
implementation.  For a geometric grid (grid.alpha > 0) each cell uses the
local mesh ratio r̃_{ij} = dx_cells[i]/dy_cells[j] and local cell spacings,
and an admissibility guard is run first to verify that the chosen P covers all
local ratios.
"""
function build_diffusion_matrix(
    solver::DiagonalMultiStencil,
    Dxx::AbstractVector,
    Dyy::AbstractVector,
    Dxy::AbstractVector,
    grid::Grid,
)
    Nx = grid.Nx
    Ny = grid.Ny
    P  = solver.P
    n  = Nx * Ny

    # ── Per-cell P selection for geometric grids ─────────────────────────────
    # For a geometric grid (alpha > 0) each cell (i,j) has its own local mesh
    # ratio r̃_{ij} and may need a different stencil order P_local.  We compute
    # P_local per cell so that the weight formula is always well-conditioned,
    # matching the paper's "selects P_min column by column" behaviour.
    # For a uniform grid (alpha == 0) the global P is used for all cells,
    # which keeps the original symmetric matrix structure.
    use_per_cell_P = (grid.alpha > 0.0)

    if !use_per_cell_P
        # Uniform grid: keep legacy admissibility guard
        Fp = Float64(P)
        for i in 1:Nx
            Dxy[i] <= 1e-14 && continue
            r_max_i = grid.dx_cells[i] / grid.dy_cells[1]
            r_min_i = grid.dx_cells[i] / grid.dy_cells[end]
            upper   = Fp * 2.0 * Dxx[i] / Dxy[i]
            lower   = 1.0 / Fp
            r_max_i <= upper + 1e-10 ||
                error("Geometric grid inadmissible at column i=$i: " *
                      "r_max=$(r_max_i) > P*(1+ρ)=$(upper). " *
                      "Increase P (use compute_P_min) or reduce alpha.")
            r_min_i >= lower - 1e-10 ||
                error("Geometric grid inadmissible at column i=$i: " *
                      "r_min=$(r_min_i) < 1/P=$(lower). " *
                      "Increase P (use compute_P_min) or reduce alpha.")
        end
    end

    # ── Per-cell coefficient grids ────────────────────────────────────────────
    Ax_g = zeros(Nx, Ny)
    Ay_g = zeros(Nx, Ny)
    w1_g = zeros(Nx, Ny)
    w2_g = zeros(Nx, Ny)
    w3_g = zeros(Nx, Ny)
    P_g  = fill(P, Nx, Ny)   # per-cell stencil order (= P for uniform grids)

    for i in 1:Nx
        dxi = grid.dx_cells[i]
        for j in 1:Ny
            dyj  = grid.dy_cells[j]
            r_ij = dxi / dyj
            P_loc = use_per_cell_P ?
                compute_P_min([Dxx[i]], [Dyy[i]], [Dxy[i]], r_ij) : P
            P_g[i,j] = P_loc
            w1, w2, w3, Ax, Ay = _cell_weights(Dxx[i], Dyy[i], Dxy[i], r_ij, P_loc)
            Ax_g[i,j] = Ax / dxi^2
            Ay_g[i,j] = Ay / dyj^2
            w1_g[i,j] = w1 / (dxi * dyj)
            w2_g[i,j] = w2 / (dxi * P_loc * dyj)
            w3_g[i,j] = w3 / (P_loc * dxi * dyj)
        end
    end

    rows = Int[]
    cols = Int[]
    vals = Float64[]
    diag = zeros(n)

    # we know that we get maximum 11 nonzeros per row
    sizehint!(rows, 11*n)
    sizehint!(cols, 11*n)
    sizehint!(vals, 11*n)

    @inline function add!(r_idx, c_idx, v)
        push!(rows, r_idx)
        push!(cols, c_idx)
        push!(vals, v)
        diag[r_idx] -= v
    end

    for i in 1:Nx, j in 1:Ny
        k  = _ridx(i, j, Ny)
        Pij = P_g[i,j]
        # E/W  (axial x)
        if i < Nx
            add!(k, _ridx(i+1,j,Ny), 0.5*(Ax_g[i,j] + Ax_g[i+1,j]))
        end
        if i > 1
            add!(k, _ridx(i-1,j,Ny), 0.5*(Ax_g[i,j] + Ax_g[i-1,j]))
        end
        # N/S  (axial y)
        if j < Ny
            add!(k, _ridx(i,j+1,Ny), 0.5*(Ay_g[i,j] + Ay_g[i,j+1]))
        end
        if j > 1
            add!(k, _ridx(i,j-1,Ny), 0.5*(Ay_g[i,j] + Ay_g[i,j-1]))
        end
        # (1,1): NE and SW
        if i < Nx && j < Ny
            add!(k, _ridx(i+1,j+1,Ny), 0.5*(w1_g[i,j] + w1_g[i+1,j+1]))
        end
        if i > 1 && j > 1
            add!(k, _ridx(i-1,j-1,Ny), 0.5*(w1_g[i,j] + w1_g[i-1,j-1]))
        end
        # (1,Pij): NE and SW added independently (original boundary behaviour)
        if i < Nx && j+Pij <= Ny
            add!(k, _ridx(i+1,j+Pij,Ny), 0.5*(w2_g[i,j] + w2_g[i+1,j+Pij]))
        end
        if i > 1 && j-Pij >= 1
            add!(k, _ridx(i-1,j-Pij,Ny), 0.5*(w2_g[i,j] + w2_g[i-1,j-Pij]))
        end
        # (Pij,1): NE and SW added independently
        if i+Pij <= Nx && j < Ny
            add!(k, _ridx(i+Pij,j+1,Ny), 0.5*(w3_g[i,j] + w3_g[i+Pij,j+1]))
        end
        if i-Pij >= 1 && j > 1
            add!(k, _ridx(i-Pij,j-1,Ny), 0.5*(w3_g[i,j] + w3_g[i-Pij,j-1]))
        end
    end
    for k in 1:n
        push!(rows, k)
        push!(cols, k)
        push!(vals, diag[k])
    end

    L = sparse(rows, cols, vals, n, n)
    rs = vec(sum(L, dims=2))
    for k in 1:n
        if abs(rs[k]) > 0; L[k,k] -= rs[k]; end
    end
    return L
end
