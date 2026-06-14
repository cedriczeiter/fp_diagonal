using Plots, Printf, LinearAlgebra, SparseArrays
using DifferentialEquations # Pour le solveur KenCarp4 de BD

gr()

# ============================================================
# INDEX (La grille commence à n=1, m=0)
# ============================================================
@inline function idx(n, m, N)
    return n + N * m + 2
end

# ============================================================
# PARAMÈTRES ÉNERGÉTIQUES ET TAUX DE RÉACTION
# ============================================================
function E10(n, m)
    x = n + m
    if x < 1 return 0.0 end
    return 24.47 - 34.0 * (cbrt(x)^2 - cbrt(x - 1)^2)
end

function E11(n, m)
    x = n + m
    if x < 1 return 0.0 end
    return 27.47 - 34.0 * (cbrt(x)^2 - cbrt(x - 1)^2)
end

function b10(n, m)
    x = n + m
    return sqrt(max(x, 0.0))
end

function b11(n, m, epsilon)
    x = n + m
    return epsilon * sqrt(max(x, 0.0))
end

function a10(n, m)
    return b10(n, m) * exp(-E10(n, m))
end

function a11(n, m, epsilon)
    return b11(n, m, epsilon) * exp(-E11(n, m))
end

# ============================================================
# RHS Becker–Döring 2D
# ============================================================
function bd2d!(dC, C, p, t)
    N, M, G10, epsilon, beta10, beta11, alpha10, alpha11 = p

    fill!(dC, zero(eltype(C))) 
    C10 = C[1]
    C11 = C[2]
    dC[1] = G10
    dC[2] = 0.0

    for m in 0:M
        for n in 1:N
            i = idx(n, m, N)

            β10 = beta10(n, m)
            β11 = beta11(n, m, epsilon)
            α10 = alpha10(n, m)
            α11 = alpha11(n, m, epsilon)

            loss = α10 + β10 * C10 + α11 + β11 * C11
            dC[i] -= loss * C[i]

            if n > 1
                j = idx(n-1, m, N)
                dC[i] += beta10(n-1, m) * C10 * C[j]
            end
            if n < N
                j = idx(n+1, m, N)
                dC[i] += alpha10(n+1, m) * C[j]
            end
            if n > 1 && m > 0
                j = idx(n-1, m-1, N)
                dC[i] += beta11(n-1, m-1, epsilon) * C11 * C[j]
            end
            if n < N && m < M
                j = idx(n+1, m+1, N)
                dC[i] += alpha11(n+1, m+1, epsilon) * C[j]
            end

            if n < N
                flux_10 = β10 * C10 * C[i] - alpha10(n+1, m) * C[idx(n+1, m, N)]
                dC[1] -= (n == 1 && m == 0) ? 2.0 * flux_10 : flux_10
            end
            if n < N && m < M
                flux_11 = β11 * C11 * C[i] - alpha11(n+1, m+1, epsilon) * C[idx(n+1, m+1, N)]
                dC[2] -= (n == 1 && m == 0) ? 2.0 * flux_11 : flux_11
            end
        end
    end
end

# ============================================================
# JACOBIEN EXACT
# ============================================================
function jacobien_bd2d!(J, C, p, t)
    N, M, G10, epsilon, beta10, beta11, alpha10, alpha11 = p
    
    C10 = C[1]
    C11 = C[2]
    
    fill!(J, 0.0)
    
    for m in 0:M
        for n in 1:N
            i = idx(n, m, N)
            
            β10 = beta10(n, m)
            β11 = beta11(n, m, epsilon)
            α10 = alpha10(n, m)
            α11 = alpha11(n, m, epsilon)
            
            J[i, i] = -(α10 + β10 * C10 + α11 + β11 * C11)
            J[i, 1] = -β10 * C[i]  
            J[i, 2] = -β11 * C[i]  
            
            if n > 1
                j = idx(n-1, m, N)
                J[i, j] += beta10(n-1, m) * C10
                J[i, 1] += beta10(n-1, m) * C[j]
            end
            if n < N
                j = idx(n+1, m, N)
                J[i, j] += alpha10(n+1, m)
            end
            if n > 1 && m > 0
                j = idx(n-1, m-1, N)
                J[i, j] += beta11(n-1, m-1, epsilon) * C11
                J[i, 2] += beta11(n-1, m-1, epsilon) * C[j]
            end
            if n < N && m < M
                j = idx(n+1, m+1, N)
                J[i, j] += alpha11(n+1, m+1, epsilon)
            end
            
            if n < N
                i_suiv = idx(n+1, m, N)
                fac = (n == 1 && m == 0) ? 2.0 : 1.0
                J[1, i]      -= fac * β10
                J[1, i_suiv] += fac * alpha10(n+1, m)
                J[1, 1]      -= fac * β10 * C[i]
            end
            
            if n < N && m < M
                i_diag = idx(n+1, m+1, N)
                fac = (n == 1 && m == 0) ? 2.0 : 1.0
                J[2, i]      -= fac * β11
                J[2, i_diag] += fac * alpha11(n+1, m+1, epsilon)
                J[2, 2]      -= fac * β11 * C[i]
            end
        end
    end
    return nothing
end

# Inclusion du module Fokker-Planck
include(joinpath(@__DIR__, "../src/FokkerPlanck.jl"))
using .FokkerPlanck

# ============================================================
# CONFIGURATION REQUALIFIÉE POUR LES 3 RUNS
# ============================================================
N_bd = 190
M_bd = 190
Nx = 190
Ny = 191 
T_end = 1000.0

grid_fp = FokkerPlanck.make_grid(xmin=2.0, xmax=180.0, Nx=Nx, ymin=0.0, ymax=180.0, Ny=Ny)

# On cible exactement 10 points de mesure temporelle (de t=0 à t=1000)
t_mesures = range(0.0, T_end, length=10)

epsilon = 0.3
C10_init = 1.0e-2   
C11_init = 1.0e-2   
G10 = 1e-17        

params_fp = FokkerPlanck.ModelParams(epsilon, C10_init, C11_init, G10; G11=0.0, r=grid_fp.r, sigma=0.9)

# Calcul unique du paramètre multi-stencil P
kin = FokkerPlanck.compute_kinetics(grid_fp.x, params_fp)
D   = FokkerPlanck.compute_diffusion_tensor(kin.d1, kin.d2)
P   = FokkerPlanck.compute_P_min(D.Dxx, D.Dyy, D.Dxy, grid_fp.r)

# Structure pour stocker les trajectoires de la MAE Relative des 3 conditions initiales
mae_history = Dict{String, Vector{Float64}}()

# Définition des scénarios de conditions initiales (n0, m0, amplitude)
scenarios = [
    ("Scénario Centré", 25.0, 25.0, 1.0e-5),
    ("Scénario Riche Solvant", 45.0, 15.0, 1.0e-5),
    ("Scénario Riche Soluté", 15.0, 45.0, 1.0e-5),
    ("Scénario Centré grande amplitude", 25.0, 25.0, 1.0e-1),
    ("Scénario Centré basse amplitude", 25.0, 25.0, 1.0e-9),
    ("Scénario Riche Solvant grande amplitude", 45.0, 15.0,1.0e-1 ),
    ("Scénario Riche Solvant basse amplitude", 45.0, 15.0, 1.0e-9),
    ("Scénario Riche Soluté grande amplitude", 15.0, 45.0, 1.0e-1),
    ("Scénario Riche Soluté basse amplitude", 15.0, 45.0, 1.0e-9)
]

# ============================================================
# BOUCLE PRINCIPALE SUR LES SCENARIOS
# ============================================================
for (nom, n0, m0, amp) in scenarios
    println("\n========================================================")
    println("Exécution : ", nom)
    println("========================================================")
    
    # 1. Construction de la condition initiale FP pour ce scénario
    U0_fp = fill(1e-15, Nx, Ny)
    σn, σm = 4.0, 4.0
    for j in 1:Ny, i in 1:Nx
        n_p = grid_fp.x[i]
        m_p = grid_fp.y[j]
        gauss = amp * exp(-((n_p - n0)^2) / (2 * σn^2) - ((m_p - m0)^2) / (2 * σm^2))
        U0_fp[i, j] += gauss
    end

    # 2. Construction de la condition initiale BD correspondante
    nb_total = N_bd * (M_bd + 1) + 2
    C0_bd = zeros(nb_total)
    C0_bd[1] = C10_init 
    C0_bd[2] = C11_init 
    for m in 0:M_bd, n in 1:N_bd
        C0_bd[idx(n, m, N_bd)] = U0_fp[n, m+1]
    end

    # 3. Résolution complète de Becker-Döring (sauvegarde uniquement aux 10 points)
    p_bd = (N=N_bd, M=M_bd, G10=G10, epsilon=epsilon, beta10=b10, beta11=b11, alpha10=a10, alpha11=a11)
    f_ode = ODEFunction(bd2d!; jac = jacobien_bd2d!, jac_prototype = sparse(zeros(nb_total, nb_total)))
    prob_bd = ODEProblem(f_ode, C0_bd, (0.0, T_end), p_bd)
    sol_bd = solve(prob_bd, KenCarp4(), reltol=1e-5, abstol=1e-20, saveat=t_mesures)

    mae_points = Float64[]
    
    for (k, t) in enumerate(t_mesures)
        
        # Résolution FP ponctuelle
        res_fp = FokkerPlanck.run_simulation(FokkerPlanck.DiagonalMultiStencil(P), grid_fp, params_fp, U0_fp, t)
        U_fp_t = res_fp.U
        
        # Extraction BD à l'instant t
        C_bd_t = sol_bd.u[k]
        U_bd_projected = similar(U_fp_t)
        for j in 1:Ny, i in 1:Nx
            U_bd_projected[i, j] = C_bd_t[idx(i, j - 1, N_bd)]
        end

        mae = sum(abs.(U_fp_t .- U_bd_projected)) / length(U_fp_t)
        push!(mae_points, mae)
        
    end
    println()
    
    mae_history[nom] = mae_points
end

# ============================================================
# 5. GÉNÉRATION ET SAUVEGARDE DU GRAPHIOUE MAE SIMPLE
# ============================================================
println("\n--- Génération du graphique de la MAE (échelle log) ---")

p_mae_log = plot(
    title = "Évolution de la MAE (échelle logarithmique)",
    xlabel = "Temps (s)",
    ylabel = "Écart Absolu Moyen",
    yscale = :log10,          # 👈 échelle logarithmique
    linewidth = 2.5,
    marker = :circle,
    grid = true,
    legend = :topright
)

for (nom, points) in mae_history
    plot!(p_mae_log, t_mesures, points, label = nom, lw = 2.5)
end

display(p_mae_log)

savefig(p_mae_log, "mae_simulation_9cas_log.png")
println("Le graphique a été sauvegardé avec succès sous le nom 'mae_simulation_9cas_log.png'.")