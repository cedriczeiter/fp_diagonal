# Physical simplified model as presented in problem statement.

# Convention: D depends only on the x-coordinate (the cluster-size axis).

struct ModelParams
    epsilon::Float64   # d2 scale factor
    C10::Float64       # initial mobile concentration (species 1)
    C11::Float64       # initial mobile concentration (species 2)
    G10::Float64       # source rate for C10
    G11::Float64       # source rate for C11 (not physically relevant, but included for testing)
    r::Float64         # dx/dy (must match grid)
    sigma::Float64     # CFL safety factor (0 < sigma <= 1), used to do sigma*CFL
end

# initialize and check parameters
function ModelParams(epsilon, C10, C11, G10; G11=0.0, r=1.0, sigma=0.9)
    epsilon > 0 || error("epsilon must be positive.")
    r > 0.0     || error("r = dx/dy must be > 0.")
    0 < sigma <= 2.0 || error("sigma must be in (0, 2].")
    C10 >= -1e-14 && C11 >= -1e-14 && G10 >= -1e-14 && G11 >= -1e-14 ||
        error("Concentrations and source rates must be non-negative.")
    return ModelParams(epsilon, C10, C11, G10, G11, r, sigma)
end

# ─────────────────────────────────────────────────────────────────────────────
# Energy and kinetics
# ─────────────────────────────────────────────────────────────────────────────

@inline _E10(x) = 24.47 - 34.0 * (x^(2/3) - (x-1)^(2/3))
@inline _E11(x) = 27.47 - 34.0 * (x^(2/3) - (x-1)^(2/3))

"""
Return (beta10, beta11, alpha10, alpha11, d1, d2), all vectors of length Nx.
d1 = max(beta10*C10 + alpha10, 0),  d2 = max(beta11*C11 + alpha11, 0).
"""
function compute_kinetics(
    x_grid::AbstractVector,
    params::ModelParams;
    C10::Float64 = params.C10,
    C11::Float64 = params.C11,
)
    n  = length(x_grid)
    beta10  = zeros(n); beta11  = zeros(n)
    alpha10 = zeros(n); alpha11 = zeros(n)
    d1 = zeros(n); d2 = zeros(n)
    for i in 1:n
        x = Float64(x_grid[i])
        x >= 2.0 || error("Energy formula requires x >= 2, got x=$(x).")
        b10 = sqrt(x); b11 = params.epsilon * sqrt(x)
        a10 = b10 * exp(-_E10(x)); a11 = b11 * exp(-_E11(x))
        beta10[i]=b10; beta11[i]=b11; alpha10[i]=a10; alpha11[i]=a11
        d1[i] = max(b10*C10 + a10, 0.0)
        d2[i] = max(b11*C11 + a11, 0.0)
    end
    return (beta10=beta10, beta11=beta11, alpha10=alpha10, alpha11=alpha11, d1=d1, d2=d2)
end

"""
Returns (Dxx, Dyy, Dxy)

Dxx = (d1+d2)/2,  Dyy = d2/2,  Dxy = d2.
"""
function compute_diffusion_tensor(d1::AbstractVector, d2::AbstractVector)
    return (Dxx = 0.5*(d1 .+ d2), Dyy = 0.5*d2, Dxy = copy(d2))
end

"""
    Returns (Fx, Fy)
"""
function compute_velocities(
    x_grid::AbstractVector,
    C10::Float64,
    C11::Float64,
    params::ModelParams,
)
    kin = compute_kinetics(x_grid, params; C10=C10, C11=C11)
    Fx  = (kin.beta10 .* C10 .- kin.alpha10) .+ (kin.beta11 .* C11 .- kin.alpha11)
    Fy  = kin.beta11 .* C11 .- kin.alpha11
    return (Fx=Fx, Fy=Fy)
end

# ─────────────────────────────────────────────────────────────────────────────
# Concentration ODE helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
Returns (A10, B10, A11, B11)

Absorption/emission integrals needed for the implicit C-ODE update.
"""
function compute_absorption(
    U::AbstractMatrix,
    grid::Grid,
    params::ModelParams,
)
    kin = compute_kinetics(grid.x, params)
    A10 = B10 = A11 = B11 = 0.0
    for i in 1:grid.Nx
        dxi = grid.dx_cells[i]
        for j in 1:grid.Ny
            cell_area = dxi * grid.dy_cells[j]
            A10 += kin.alpha10[i] * U[i,j] * cell_area
            B10 += kin.beta10[i]  * U[i,j] * cell_area
            A11 += kin.alpha11[i] * U[i,j] * cell_area
            B11 += kin.beta11[i]  * U[i,j] * cell_area
        end
    end
    return (A10=A10, B10=B10, A11=A11, B11=B11)
end

"""
returns (C10_new, C11_new)

Backward-Euler implicit update for the mobile concentrations.
"""
function step_concentrations(
    C10::Float64,
    C11::Float64,
    dt::Float64,
    abs,
    params::ModelParams;
    fixed::Bool = false,
)
    C10_new = fixed ? C10 : (C10 + dt*(params.G10 + abs.A10)) / (1.0 + dt*abs.B10)
    C11_new = fixed ? C11 : (C11 + dt*(params.G11 + abs.A11)) / (1.0 + dt*abs.B11)
    C10_new >= -1e-14 && C11_new >= -1e-14||
        error("Mobile concentrations became negative: C10=$(C10_new), C11=$(C11_new).")
    return (max(C10_new,0.0), max(C11_new,0.0))
end

# ─────────────────────────────────────────────────────────────────────────────
# CFL helpers
# ─────────────────────────────────────────────────────────────────────────────

function _adv_cfl(Fx::AbstractVector, Fy::AbstractVector, grid::Grid)
    denom = maximum(abs.(Fx)) / minimum(grid.dx_cells) + maximum(abs.(Fy)) / minimum(grid.dy_cells)
    return denom > 1e-30 ? 1.0/denom : Inf
end

function _kt_diff_cfl(Dxx::AbstractVector, Dyy::AbstractVector, grid::Grid)
    inv_dx2 = 1/minimum(grid.dx_cells)^2; inv_dy2 = 1/minimum(grid.dy_cells)^2
    rate = maximum(4.0.*Dxx.*inv_dx2 .+ 4.0.*Dyy.*inv_dy2)
    return rate > 0 ? 2.0/rate : Inf
end

# ─────────────────────────────────────────────────────────────────────────────
# Evaluate all coefficients at given (C10, C11)
# ─────────────────────────────────────────────────────────────────────────────

function _eval_coeffs(grid::Grid, C10::Float64, C11::Float64, params::ModelParams)
    kin  = compute_kinetics(grid.x, params; C10=C10, C11=C11)
    D    = compute_diffusion_tensor(kin.d1, kin.d2)
    vel  = compute_velocities(grid.x, C10, C11, params)
    return merge(kin, D, vel)   # NamedTuple with all fields
end
