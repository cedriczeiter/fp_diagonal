# Kurganov-Tadmor (2000) central-upwind explicit solver.
# Both advection (MP5) and diffusion (KT flux, eq. 4.20) are integrated
# explicitly with SSP-RK3.

struct Kurganov <: AbstractFPSolver end

# ─────────────────────────────────────────────────────────────────────────────
# KT slope limiter (minmod3)
# ─────────────────────────────────────────────────────────────────────────────

@inline function _mm3(a::Float64, b::Float64, c::Float64)
    if a > 0.0 && b > 0.0 && c > 0.0
        return min(a, b, c)
    elseif a < 0.0 && b < 0.0 && c < 0.0
        return max(a, b, c)
    else
        return 0.0
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Minmod-limited cell slopes for the KT cross-term (eq. 4.20)
# only needed for diffusion, not advection
# ─────────────────────────────────────────────────────────────────────────────

function _kt_slopes!(ux::AbstractMatrix, uy::AbstractMatrix, U::AbstractMatrix, grid::Grid)
    
    Nx, Ny = size(U)
    dx = grid.dx
    dy = grid.dy
    theta=1.0 # we always use theta=1.0 (from paper)
    for j in 1:Ny, i in 1:Nx
        if i==1
            ux[i,j]=(U[2,j]-U[1,j])/dx
        elseif i==Nx
            ux[i,j]=(U[Nx,j]-U[Nx-1,j])/dx
        else
            dm=U[i,j]-U[i-1,j]
            dp=U[i+1,j]-U[i,j]
            ux[i,j]=_mm3(theta*dm, 0.5*(dm+dp), theta*dp)/dx
        end
    end
    for i in 1:Nx, j in 1:Ny
        if j==1
            uy[i,j]=(U[i,2]-U[i,1])/dy
        elseif j==Ny
            uy[i,j]=(U[i,Ny]-U[i,Ny-1])/dy
        else
            dm=U[i,j]-U[i,j-1]
            dp=U[i,j+1]-U[i,j]
            uy[i,j]=_mm3(theta*dm, 0.5*(dm+dp), theta*dp)/dy
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# KT diffusion RHS (eq. 4.20)
# ─────────────────────────────────────────────────────────────────────────────

function _kt_diffusion_rhs!(
    rhs::AbstractMatrix,
    U::AbstractMatrix,
    ux::AbstractMatrix,
    uy::AbstractMatrix,
    Dxx::AbstractVector,
    Dyy::AbstractVector,
    Dxy::AbstractVector,
    grid::Grid,
)
    Nx, Ny = size(U)
    dx = grid.dx
    dy = grid.dy

    # x-direction flux, at interfaces i+1/2,j
    for j in 1:Ny, i in 1:Nx-1
        Px = 0.5*(Dxx[i]+Dxx[i+1])*(U[i+1,j]-U[i,j])/dx +
             0.25*(Dxy[i]*uy[i,j] + Dxy[i+1]*uy[i+1,j])
        rhs[i,j] += Px/dx
        rhs[i+1,j] -= Px/dx
    end
    # y-direction flux, at interfaces i,j+1/2
    for i in 1:Nx, j in 1:Ny-1
        Py = Dyy[i]*(U[i,j+1]-U[i,j])/dy +
             0.25*Dxy[i]*(ux[i,j]+ux[i,j+1])
        rhs[i,j] += Py/dy
        rhs[i,j+1] -= Py/dy
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Combined KT spatial RHS: advection (MP5) + diffusion (KT)
# ─────────────────────────────────────────────────────────────────────────────

function _kt_spatial_rhs!(
    rhs::AbstractMatrix,
    U::AbstractMatrix,
    ux::AbstractMatrix,
    uy::AbstractMatrix,
    Fx::AbstractVector,
    Fy::AbstractVector,
    Dxx::AbstractVector,
    Dyy::AbstractVector,
    Dxy::AbstractVector,
    grid::Grid,
    adv_cache,
)
    fill!(rhs, 0.0)
    _add_mp5_advection_rhs!(rhs, U, Fx, Fy, grid, adv_cache) # updates rhs in-place
    _kt_slopes!(ux, uy, U, grid) # updates ux, uy in-place
    _kt_diffusion_rhs!(rhs, U, ux, uy, Dxx, Dyy, Dxy, grid) # updates rhs in-place
end
