# Low-level numerical helpers: MP5 reconstruction, SSP-RK3, row-major utilities.

# ─────────────────────────────────────────────────────────────────────────────
# Scalar limiters
# ─────────────────────────────────────────────────────────────────────────────

@inline _minmod(a, b) = a * b > 0.0 ? sign(a) * min(abs(a), abs(b)) : 0.0
@inline _median(x, y, z) = x + _minmod(y - x, z - x)

# ─────────────────────────────────────────────────────────────────────────────
# MP5 interface flux (Suresh & Huynh 1997)
# ─────────────────────────────────────────────────────────────────────────────

function compute_mp5_flux_1d!(F::AbstractVector{T}, u::AbstractVector{T}, v::AbstractVector{T}, alpha_mp5=4.0, beta_mp5=4.0) where T<:Real
    N = length(u)
    fill!(F, zero(T))

    @inline get_u(idx) = u[clamp(idx, 1, N)]
    @inline get_v(idx) = v[clamp(idx, 1, N)]

    F[1] = zero(T)
    F[N + 1] = zero(T)

    for i in 1:(N - 1)
        vel = 0.5 * (get_v(i) + get_v(i + 1))   # interface velocity

        # Stencil for i
        u_m2 = get_u(i - 2)
        u_m1 = get_u(i - 1)
        u_i  = get_u(i)
        u_p1 = get_u(i + 1)
        u_p2 = get_u(i + 2)
        u_p3 = get_u(i + 3)

        # --- LEFT INTERFACE RECONSTRUCTION (f_L) ---
        #formula (2.1)
        u_L_orig = (2.0*u_m2 - 13.0*u_m1 + 47.0*u_i + 27.0*u_p1 - 3.0*u_p2) / 60.0

        #formula (2.19) applied to d_i and d_i-1
        d_i   = u_p1 + u_m1 - 2.0 * u_i
        d_m1  = u_i + u_m2 - 2.0 * u_m1

        u_UL_L = u_i + alpha_mp5 * (u_i - u_m1) #2.8
        u_MP_L = u_i + _minmod(u_p1 - u_i, alpha_mp5 * (u_i - u_m1)) #2.12

        u_AV_L = (u_i + u_p1) * 0.5 #2.16
        u_FL_L = u_i + (u_i - u_m1) * 0.5 #2.15
        u_FR_L = u_p1 + (u_p1 - u_p2) * 0.5 #2.15
        u_MD_L = _median(u_AV_L, u_FL_L, u_FR_L) #2.17

        u_LC_L = u_i + (u_i - u_m1) * 0.5 + (beta_mp5 / 3.0) * _minmod(d_m1, d_i) #2.22 and 2.20

        #calculating our interval of 2.25
        u_min_L = max(min(u_i, u_p1, u_MD_L), min(u_i, u_UL_L, u_LC_L)) #2.24a
        u_max_L = min(max(u_i, u_p1, u_MD_L), max(u_i, u_UL_L, u_LC_L)) #2.24b

        if (u_L_orig - u_i) * (u_L_orig - u_MP_L) >= 0.0  #check if it is in interval 2.25, if not move it to the median of the interval using 2.26 -> uses 2.30 criterion
            u_L = _median(u_L_orig, u_min_L, u_max_L) #2.26
        else
            u_L = u_L_orig
        end

        # --- RIGHT INTERFACE RECONSTRUCTION (f_R) ---
        u_R_orig = (2.0*u_p3 - 13.0*u_p2 + 47.0*u_p1 + 27.0*u_i - 3.0*u_m1) / 60.0

        d_p1 =  u_i  + u_p2 - 2.0 * u_p1
        d_p2 =  u_p1 + u_p3 - 2.0 * u_p2

        u_UL_R = u_p1 + alpha_mp5 * (u_p1 - u_p2)
        u_MP_R = u_p1 + _minmod(u_i - u_p1, alpha_mp5 * (u_p1 - u_p2))

        u_AV_R = (u_p1 + u_i) * 0.5
        u_FL_R = u_p1 + (u_p1 - u_p2) * 0.5
        u_FR_R = u_p1 + (u_p1 - u_i) * 0.5
        u_MD_R = _median(u_AV_R, u_FL_R, u_FR_R)

        u_LC_R = u_p1 + (u_p1 - u_p2) * 0.5 + (beta_mp5 / 3.0) * _minmod(d_p2, d_p1)

        u_min_R = max(min(u_p1, u_i, u_MD_R), min(u_p1, u_UL_R, u_LC_R))
        u_max_R = min(max(u_p1, u_i, u_MD_R), max(u_p1, u_UL_R, u_LC_R))

        if (u_R_orig - u_p1) * (u_R_orig - u_MP_R) >= 0.0
            u_R = _median(u_R_orig, u_min_R, u_max_R)
        else
            u_R = u_R_orig
        end

        # upwind the flux based on velocity direction
        F[i+1] = max(vel, 0.0) * u_L + min(vel, 0.0) * u_R
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# SSP-RK3
# ─────────────────────────────────────────────────────────────────────────────

# pre-allocate temporary arrays
function _build_mp5_cache(nx::Int, ny::Int)
    return (
        u_row=zeros(Float64, nx), v_row=zeros(Float64, nx), F_x=zeros(Float64, nx+1),
        u_col=zeros(Float64, ny), v_col=zeros(Float64, ny), F_y=zeros(Float64, ny+1),
    )
end

function _ssprk3_step!(
    u_out::AbstractMatrix,
    u_in::AbstractMatrix,
    dt::Float64,
    rhs::AbstractMatrix,
    s1::AbstractMatrix,
    s2::AbstractMatrix,
    rhs_func!::Function,
)
    nx, ny = size(u_in)
    rhs_func!(rhs, u_in)
    @inbounds for j in 1:ny, i in 1:nx; s1[i,j] = u_in[i,j] + dt*rhs[i,j]; end
    rhs_func!(rhs, s1)
    @inbounds for j in 1:ny, i in 1:nx; s2[i,j] = 0.75*u_in[i,j] + 0.25*(s1[i,j] + dt*rhs[i,j]); end
    rhs_func!(rhs, s2)
    @inbounds for j in 1:ny, i in 1:nx
        u_out[i,j] = (1/3)*u_in[i,j] + (2/3)*(s2[i,j] + dt*rhs[i,j])
    end
    return u_out
end

# ─────────────────────────────────────────────────────────────────────────────
# MP5 advection RHS: rhs[i,j] -= (F[i+1] - F[i])/dx  for each row/col in both directions (2D extension of 1D MP5)
# ─────────────────────────────────────────────────────────────────────────────

function _add_mp5_advection_rhs!(
    rhs::AbstractMatrix,
    U::AbstractMatrix,
    Fx::AbstractVector,
    Fy::AbstractVector,
    grid,
    cache,
)
    Nx, Ny = size(U)
    for j in 1:Ny
        @views cache.u_row .= U[:, j]
        cache.v_row .= Fx
        compute_mp5_flux_1d!(cache.F_x, cache.u_row, cache.v_row)
        for i in 1:Nx
            rhs[i,j] -= (cache.F_x[i+1] - cache.F_x[i]) / grid.dx_cells[i]
        end
    end
    for i in 1:Nx
        @views cache.u_col .= U[i, :]
        fill!(cache.v_col, Fy[i])
        compute_mp5_flux_1d!(cache.F_y, cache.u_col, cache.v_col)
        for j in 1:Ny
            rhs[i,j] -= (cache.F_y[j+1] - cache.F_y[j]) / grid.dy_cells[j]
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Row-major sparse indexing
# ─────────────────────────────────────────────────────────────────────────────

@inline _ridx(i::Int, j::Int, Ny::Int) = (i-1)*Ny + j

function _pack_rowmajor(U::AbstractMatrix)
    Nx, Ny = size(U)
    v = zeros(Float64, Nx*Ny)
    for i in 1:Nx, j in 1:Ny; v[_ridx(i,j,Ny)] = U[i,j]; end
    return v
end

function _unpack_rowmajor!(U::AbstractMatrix, v::AbstractVector)
    Nx, Ny = size(U)
    for i in 1:Nx, j in 1:Ny; U[i,j] = v[_ridx(i,j,Ny)]; end
    return U
end
