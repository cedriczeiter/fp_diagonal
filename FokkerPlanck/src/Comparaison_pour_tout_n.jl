using Plots, Printf, LinearAlgebra, SparseArrays
using DifferentialEquations

gr()

# ============================================================
# FONCTIONS INDICES ET PHYSIQUE
# ============================================================
@inline function idx(n, m, N)
    return n + N * m + 2
end

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

# RHS Becker–Döring 2D
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

# Inclusion du module Fokker-Planck
include(joinpath(@__DIR__, "../src/FokkerPlanck.jl"))
using .FokkerPlanck

# ============================================================
# PARAMÈTRES ET STRUCTURE DE LA BOUCLE GLOBALE
# ============================================================
epsilon = 0.3
C10_init = 1.5e-2   
C11_init = 1.0e-2   
G10 = 1e-17         
T_end = 1000.0       

plage_tailles = [5,20,40,60,80,100,120,140,160,180,190]
historique_mae = Float64[]
historique_max = Float64[]

println("=== DÉBUT DE LA COMPARAISON PAR TAILLE DE GRILLE ===")

for k in plage_tailles
    @printf("\n--> Simulation pour une taille de grille N = M = %d\n", k)
    
    # 1. Configuration des dimensions pour l'étape k
    N_bd = k
    M_bd = k
    Nx = k
    Ny = k + 1
    
    grid_fp = FokkerPlanck.make_grid(xmin=2.0, xmax=Float64(k), Nx=Nx, ymin=0.0, ymax=Float64(k), Ny=Ny)
    params_fp = FokkerPlanck.ModelParams(epsilon, C10_init, C11_init, G10; G11=0.0, r=grid_fp.r, sigma=0.9)
    
    # 2. Condition initiale (Gaussienne centrée de manière proportionnelle à la taille k)
    amplitude_max = 1e-5
    U0_fp = fill(1e-15, Nx, Ny)
    
    # On place la cloche au premier quart de la grille pour qu'elle ait la place d'évoluer
    n0_phys = max(3.0, k * 0.2)
    m0_phys = max(3.0, k * 0.2)
    σn = max(1.0, k * 0.05)
    σm = max(1.0, k * 0.05)
    
    for j in 1:Ny, i in 1:Nx
        n_phys = grid_fp.x[i]
        m_phys = grid_fp.y[j]
        gauss = amplitude_max * exp(-((n_phys - n0_phys)^2) / (2 * σn^2) - ((m_phys - m0_phys)^2) / (2 * σm^2))
        U0_fp[i, j] += gauss
    end
    
    nb_total = N_bd * (M_bd + 2) # Taille ajustée selon l'indexation Becker-Döring
    C0_bd = zeros(nb_total)
    C0_bd[1] = C10_init
    C0_bd[2] = C11_init
    for m in 0:M_bd, n in 1:N_bd
        C0_bd[idx(n, m, N_bd)] = U0_fp[n, m+1]
    end
    
    # 3. Résolution Becker-Döring pour la taille k
    p_bd = (N=N_bd, M=M_bd, G10=G10, epsilon=epsilon, beta10=b10, beta11=b11, alpha10=a10, alpha11=a11)
    prob_bd = ODEProblem(bd2d!, C0_bd, (0.0, T_end), p_bd)
    sol_bd = solve(prob_bd, KenCarp4(), reltol=1e-4, abstol=1e-12, save_everystep=false)
    
    # 4. Résolution Fokker-Planck pour la taille k
    kin = FokkerPlanck.compute_kinetics(grid_fp.x, params_fp)
    D   = FokkerPlanck.compute_diffusion_tensor(kin.d1, kin.d2)
    P_val = FokkerPlanck.compute_P_min(D.Dxx, D.Dyy, D.Dxy, grid_fp.r)
    
    res_fp = FokkerPlanck.run_simulation(
        FokkerPlanck.DiagonalMultiStencil(P_val), 
        grid_fp, params_fp, U0_fp, T_end; 
        max_steps=10_000_000
    )
    
    # ============================================================
    # 5. CALCUL DE LA MAE ET DE L'ÉCART MAX SUR TOUTE LA GRILLE k
    # ============================================================
    U_fp_final = res_fp.U
    C_bd_final = sol_bd.u[end]
    
    # Reconstruction de la structure de grille BD pour comparaison matricielle directe
    U_bd_projected = zeros(Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        U_bd_projected[i, j] = C_bd_final[idx(i, j-1, N_bd)]
    end
    
    # Écarts absolus
    ecarts_matrice = abs.(U_fp_final .- U_bd_projected)
    
    mae_courante = sum(ecarts_matrice) / length(ecarts_matrice)
    max_courant  = maximum(ecarts_matrice)
    
    push!(historique_mae, mae_courante)
    push!(historique_max, max_courant)
    
    @printf("   -> MAE calculée : %.4e | Max : %.4e\n", mae_courante, max_courant)
end

# ============================================================
# 6. TRACÉ DES GRAPHIOUES D'ÉVOLUTION DE L'ERREUR (MAE & MAX)
# ============================================================
println("\n--- Génération des graphiques d'erreurs globaux ---")

# Courbe de l'erreur moyenne (MAE)
p_mae = plot(
    plage_tailles, historique_mae,
    label = "MAE (Écart Moyen Absolu)",
    linewidth = 2.5, color = :green,
    xlabel = "Taille maximale de la grille (k)",
    ylabel = "Erreur (Échelle Log10)",
    title = "Évolution de la MAE vs Taille de grille",
    yaxis = :log10
)

# Courbe de l'erreur maximale constatée
p_max = plot(
    plage_tailles, historique_max,
    label = "Écart Maximum constaté",
    linewidth = 2.5, color = :red,
    xlabel = "Taille maximale de la grille (k)",
    ylabel = "Erreur (Échelle Log10)",
    title = "Évolution de l'Erreur Max vs Taille de grille",
    yaxis = :log10
)

# Assemblage des deux graphiques côte à côte
p_comparaison_grilles = plot(p_mae, p_max, layout = (1, 2), size = (1100, 480))

display(p_comparaison_grilles)

# --- LIGNES DE SAUVEGARDE ---
savefig(p_comparaison_grilles, "comparaison_erreurs_grilles.png")
println("Le graphique combiné a été enregistré sous 'comparaison_erreurs_grilles.png'")