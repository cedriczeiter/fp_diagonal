# Naive 9-point cross stencil for the mixed derivative.
# NOT positivity-preserving when D_xy is large relative to D_xx or D_yy.
# Used for comparison only.

struct Naive9Point <: AbstractFPSolver end

"""

Assemble the 9-point diffusion matrix:
  E/W:  +Dxx[i]/dx^2
  N/S:  +Dyy[i]/dy^2
  NE/SW: +Dxy[i]/(4 dx dy)
  SE/NW: -Dxy[i]/(4 dx dy)

Diagonal is set so each row sums exactly to zero (mass conservation).

Returns a sparse matrix L such that L * u_vec gives the diffusion term
"""
function build_diffusion_matrix(::Naive9Point,
    Dxx::AbstractVector,
    Dyy::AbstractVector,
    Dxy::AbstractVector,
    grid::Grid,
)
    Nx = grid.Nx
    Ny = grid.Ny
    n = Nx*Ny

    rows = Int[]
    cols = Int[]
    vals = Float64[]
    diag = zeros(n)

    @inline function add!(r, c, v)
        push!(rows, r)
        push!(cols, c)
        push!(vals, v)
        diag[r] -= v
    end

    for i in 1:Nx
        inv_dx2   = 1/grid.dx_cells[i]^2
        for j in 1:Ny
        k = _ridx(i,j,Ny)
            inv_dy2   = 1/grid.dy_cells[j]^2
            inv_4dxdy = 1/(4*grid.dx_cells[i]*grid.dy_cells[j])
            if i < Nx # E
                add!(k, _ridx(i+1,j,Ny), Dxx[i]*inv_dx2)
            end
            if i > 1 # W
                add!(k, _ridx(i-1,j,Ny), Dxx[i]*inv_dx2)
            end
            if j < Ny # N
                add!(k, _ridx(i,j+1,Ny), Dyy[i]*inv_dy2)
            end
            if j > 1 # S
                add!(k, _ridx(i,j-1,Ny), Dyy[i]*inv_dy2)
            end
            if i < Nx && j < Ny # NE
                add!(k, _ridx(i+1,j+1,Ny),  Dxy[i]*inv_4dxdy)
            end
            if i > 1 && j > 1 # SW
                add!(k, _ridx(i-1,j-1,Ny),  Dxy[i]*inv_4dxdy)
            end
            if i < Nx && j > 1 # SE
                add!(k, _ridx(i+1,j-1,Ny), -Dxy[i]*inv_4dxdy)
            end
            if i > 1 && j < Ny # NW
                add!(k, _ridx(i-1,j+1,Ny), -Dxy[i]*inv_4dxdy)
            end
        end
    end
    # Now add the diagonal entries so each row sums to zero (mass conservation).
    for k in 1:n
        push!(rows,k)
        push!(cols,k)
        push!(vals,diag[k])
    end

    L = sparse(rows, cols, vals, n, n)

    row_sums = vec(sum(L, dims=2))
    for k in 1:n
        if abs(row_sums[k]) > 0.0
            L[k, k] -= row_sums[k]
        end
    end
    return L
end
