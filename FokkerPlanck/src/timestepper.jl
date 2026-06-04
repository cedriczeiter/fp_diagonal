# Shared run loop and per-solver step dispatch.
#
# Strang-IMEX (Naive9Point / DiagonalMultiStencil):
#   half advection → half C-update → implicit diffusion → half C-update → half advection
#   dt = sigma * adv_CFL
#
# Kurganov:
#   SSP-RK3 (advection + KT diffusion) → implicit C-update
#   dt = sigma * min(adv_CFL, KT_diff_CFL)

# ─────────────────────────────────────────────────────────────────────────────
# Internal single-step for IMEX solvers (Naive9Point, DiagonalMultiStencil)
# ─────────────────────────────────────────────────────────────────────────────

function _step_imex!(
    solver::AbstractFPSolver,  # Naive9Point or DiagonalMultiStencil
    U::AbstractMatrix,
    C10::Float64,
    C11::Float64,
    grid::Grid,
    params::ModelParams,
    t::Float64;
    mass0::Float64,
    fixed::Bool,
    max_dt::Float64,
    adv_cache,
    rhs::AbstractMatrix,
    s1::AbstractMatrix,
    s2::AbstractMatrix,
)
    abs(params.r - grid.r) < 1e-12 ||
        error("Grid ratio mismatch: params.r=$(params.r), grid.r=$(grid.r).")

    c = _eval_coeffs(grid, C10, C11, params)
    dt_adv = _adv_cfl(c.Fx, c.Fy, grid)
    dt     = min(params.sigma * dt_adv, max_dt)
    isfinite(dt) && dt > 0 || error("IMEX: invalid dt=$(dt) at t=$(t).")

    # Step 1: half advection
    U_star = similar(U)
    adv_rhs_n! = (r,V) -> (fill!(r,0.0); _add_mp5_advection_rhs!(r,V,c.Fx,c.Fy,grid,adv_cache))
    _ssprk3_step!(U_star, U, dt/2, rhs, s1, s2, adv_rhs_n!)

    # Step 2: half C-update
    ab_star = compute_absorption(U_star, grid, params)
    C10_h, C11_h = step_concentrations(C10, C11, dt/2, ab_star, params; fixed=fixed)

    # Step 3: implicit diffusion at midpoint coefficients
    c_h = _eval_coeffs(grid, C10_h, C11_h, params)
    L   = build_diffusion_matrix(solver, c_h.Dxx, c_h.Dyy, c_h.Dxy, grid)
    n   = grid.Nx * grid.Ny
    A   = sparse(I, n, n) - dt * L
    U_new = similar(U)
    _unpack_rowmajor!(U_new, A \ _pack_rowmajor(U_star))
    all(isfinite, U_new) || error("IMEX: NaN/Inf at t=$(t), dt=$(dt).")

    # Step 4: half C-update
    ab_new = compute_absorption(U_new, grid, params)
    C10_n, C11_n = step_concentrations(C10_h, C11_h, dt/2, ab_new, params; fixed=fixed)

    # Step 5: half advection with updated velocities
    c_n = _eval_coeffs(grid, C10_n, C11_n, params)
    U_final = similar(U)
    adv_rhs_np1! = (r,V) -> (fill!(r,0.0); _add_mp5_advection_rhs!(r,V,c_n.Fx,c_n.Fy,grid,adv_cache))
    _ssprk3_step!(U_final, U_new, dt/2, rhs, s1, s2, adv_rhs_np1!)

    mass_new = total_mass(U_final, grid)
    U .= U_final
    diag = material_diagnostics(U, grid, C10_n, C11_n)
    return (dt=dt, t_new=t+dt, C10=C10_n, C11=C11_n,
            mass=diag.mass, Y_mat=diag.Y_mat, X_mat=diag.X_mat,
            rel_mass_drift=abs(mass_new-mass0)/max(abs(mass0),1e-30),
            dt_adv=dt_adv)
end

# ─────────────────────────────────────────────────────────────────────────────
# Internal single-step for Kurganov
# ─────────────────────────────────────────────────────────────────────────────

function _step_kurganov!(
    ::Kurganov,
    U::AbstractMatrix,
    C10::Float64,
    C11::Float64,
    grid::Grid,
    params::ModelParams,
    t::Float64;
    mass0::Float64,
    fixed::Bool,
    max_dt::Float64,
    adv_cache,
    rhs::AbstractMatrix,
    s1::AbstractMatrix,
    s2::AbstractMatrix,
    ux::AbstractMatrix,
    uy::AbstractMatrix,
)
    c = _eval_coeffs(grid, C10, C11, params)
    dt_adv  = _adv_cfl(c.Fx, c.Fy, grid)
    dt_diff = _kt_diff_cfl(c.Dxx, c.Dyy, grid)
    dt = min(params.sigma * min(dt_adv, dt_diff), max_dt)
    isfinite(dt) && dt > 0 || error("KT: invalid dt=$(dt) at t=$(t).")

    Fx=c.Fx; Fy=c.Fy; Dxx=c.Dxx; Dyy=c.Dyy; Dxy=c.Dxy
    kt_rhs! = (r,V) -> _kt_spatial_rhs!(r,V,ux,uy,Fx,Fy,Dxx,Dyy,Dxy,grid,adv_cache)
    U_new = similar(U)
    _ssprk3_step!(U_new, U, dt, rhs, s1, s2, kt_rhs!)
    all(isfinite, U_new) || error("KT: NaN/Inf at t=$(t), dt=$(dt).")

    ab = compute_absorption(U_new, grid, params)
    C10_n, C11_n = step_concentrations(C10, C11, dt, ab, params; fixed=fixed)

    mass_new = total_mass(U_new, grid)
    U .= U_new
    diag = material_diagnostics(U, grid, C10_n, C11_n)
    return (dt=dt, t_new=t+dt, C10=C10_n, C11=C11_n,
            mass=diag.mass, Y_mat=diag.Y_mat, X_mat=diag.X_mat,
            rel_mass_drift=abs(mass_new-mass0)/max(abs(mass0),1e-30),
            dt_adv=dt_adv, dt_diff=dt_diff)
end

# ─────────────────────────────────────────────────────────────────────────────
# Public run_simulation
# ─────────────────────────────────────────────────────────────────────────────

"""

Run the solver from t=0 to t=T_end.

`solver` can be any of: `Naive9Point()`, `DiagonalMultiStencil(P)`, `Kurganov()`.

Returns a named tuple:
  U, C10, C11, t, times, mass_rel_hist, min_u_hist, C10_hist, C11_hist,
  Y_rel_hist, X_budget_rel_hist, exploded
"""
function run_simulation(
    solver::AbstractFPSolver,
    grid::Grid,
    params::ModelParams,
    U0::AbstractMatrix,
    T_end::Float64;
    max_steps::Int  = 5_000_000,
    fixed_mobile::Bool = false,
)
    U   = copy(U0)
    C10 = params.C10; C11 = params.C11
    t   = 0.0

    mass0 = total_mass(U, grid)
    diag0 = material_diagnostics(U, grid, C10, C11)
    X0=diag0.X_mat; Y0=diag0.Y_mat

    times             = Float64[0.0]
    mass_rel_hist     = Float64[0.0]
    min_u_hist        = Float64[minimum(U)]
    C10_hist          = Float64[C10]
    C11_hist          = Float64[C11]
    Y_rel_hist        = Float64[0.0]
    X_budget_rel_hist = Float64[0.0]

    # Pre-allocate shared buffers
    adv_cache = _build_mp5_cache(grid.Nx, grid.Ny)
    rhs = zeros(Float64, grid.Nx, grid.Ny)
    s1  = zeros(Float64, grid.Nx, grid.Ny)
    s2  = zeros(Float64, grid.Nx, grid.Ny)
    ux  = zeros(Float64, grid.Nx, grid.Ny)   # only for Kurganov
    uy  = zeros(Float64, grid.Nx, grid.Ny)

    exploded = false
    positivity_warned = false

    for _ in 1:max_steps
        try
            step_out = if solver isa Kurganov
                _step_kurganov!(solver, U, C10, C11, grid, params, t;
                    mass0=mass0, fixed=fixed_mobile, max_dt=T_end-t,
                    adv_cache=adv_cache, rhs=rhs, s1=s1, s2=s2, ux=ux, uy=uy)
            else
                _step_imex!(solver, U, C10, C11, grid, params, t;
                    mass0=mass0, fixed=fixed_mobile, max_dt=T_end-t,
                    adv_cache=adv_cache, rhs=rhs, s1=s1, s2=s2)
            end

            C10 = step_out.C10; C11 = step_out.C11; t = step_out.t_new
            push!(times,             t)
            push!(mass_rel_hist,     step_out.rel_mass_drift)
            _min_u = minimum(U)
            push!(min_u_hist,        _min_u)
            if _min_u < 0.0 && !(solver isa Naive9Point) && !positivity_warned
                positivity_warned = true
                @printf("\n")
                @printf("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                @printf("  !! POSITIVITY VIOLATION — THIS SHOULD BE MATHEMATICALLY    !!\n")
                @printf("  !! IMPOSSIBLE FOR %s                !!\n", rpad(string(typeof(solver)), 28))
                @printf("  !! t=%.6f   min(U)=%.6e   step=%d          !!\n",
                        t, _min_u, length(times))
                @printf("  !! Check stencil admissibility, grid ratio r=%.4f,        !!\n", grid.r)
                @printf("  !! mass conservation (L symmetry), and CFL condition.      !!\n")
                @printf("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                @printf("\n")
            end
            push!(C10_hist,          C10)
            push!(C11_hist,          C11)
            push!(Y_rel_hist,        abs(step_out.Y_mat - Y0)/max(abs(Y0),1e-30))
            push!(X_budget_rel_hist, abs(step_out.X_mat - (X0+params.G10*t))/max(abs(X0),1e-30))

            t >= T_end && break

        catch e
            exploded = true
            @printf("  [%s] EXPLODED at t=%.4f: %s\n",
                    typeof(solver), t, sprint(showerror, e))
            break
        end

        if maximum(abs.(U)) > 1e10
            exploded = true
            if !(solver isa Naive9Point)
                @printf("\n")
                @printf("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                @printf("  !! CATASTROPHIC BLOW-UP — THIS SHOULD BE MATHEMATICALLY     !!\n")
                @printf("  !! IMPOSSIBLE FOR %s                !!\n", rpad(string(typeof(solver)), 28))
                @printf("  !! t=%.6f   max|U|=%.6e                           !!\n",
                        t, maximum(abs.(U)))
                @printf("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                @printf("\n")
            else
                @printf("  [%s] BLEW UP at t=%.4f (max|U|=%.2e)\n",
                        typeof(solver), t, maximum(abs.(U)))
            end
            break
        end
    end

    return (
        U=U, C10=C10, C11=C11, t=t,
        mass0=mass0, X0=X0, Y0=Y0,
        times=times,
        mass_rel_hist=mass_rel_hist,
        min_u_hist=min_u_hist,
        C10_hist=C10_hist,
        C11_hist=C11_hist,
        Y_rel_hist=Y_rel_hist,
        X_budget_rel_hist=X_budget_rel_hist,
        exploded=exploded,
    )
end
