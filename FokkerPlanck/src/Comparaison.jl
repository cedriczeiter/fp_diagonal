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
import .FokkerPlanck: ModelParams 

# ============================================================
# 1. CONFIGURATION DES GRILLES ÉQUIVALENTES 
# ============================================================
N_bd = 190
M_bd = 190
Nx = 190
Ny = 191 

grid_fp = FokkerPlanck.make_grid(xmin=2.0, xmax=180.0, Nx=Nx, ymin=0.0, ymax=180.0, Ny=Ny)

# Concentrations initiales stables et raisonnables
epsilon = 0.3
C10_init = 1.5e-2   
C11_init = 1.0e-2   
G10 = 1e-17         # Conservé à 10^-17
T_end = 1000.0        

t_sauvegarde = range(0.0, T_end, length=40)

params_fp = FokkerPlanck.ModelParams(epsilon, C10_init, C11_init, G10; G11=0.0, r=grid_fp.r, sigma=0.9)

# ============================================================
# 2. CONFIGURATION ET CALCUL DU STENCIL P 
# ============================================================
println("--- Préparation du solveur Multi-Stencil ---")
kin = FokkerPlanck.compute_kinetics(grid_fp.x, params_fp)
D   = FokkerPlanck.compute_diffusion_tensor(kin.d1, kin.d2)
P   = FokkerPlanck.compute_P_min(D.Dxx, D.Dyy, D.Dxy, grid_fp.r)

@printf("Paramètre Multi-Stencil sélectionné automatiquement : P = %d\n", P)

# ============================================================
# 3. INITIALISATION DE LA GAUSSIENNE PETITE ET SÉCURISÉE
# ============================================================
println("\n--- Initialisation de la condition initiale (Gaussienne faible + Floor) ---")

amplitude_max = 1e-5 # Petite valeur demandée
U0_fp = fill(1e-15, Nx, Ny) # Floor de sécurité contre log10.(0)

# Positionnement idéal : loin de l'origine et des parois pour éviter les krachs IMEX
n0_phys = 25.0
m0_phys = 25.0
σn = 4.0
σm = 4.0

for j in 1:Ny, i in 1:Nx
    n_phys = grid_fp.x[i]
    m_phys = grid_fp.y[j]
    
    # Calcul de la cloche
    gauss = amplitude_max * exp(-((n_phys - n0_phys)^2) / (2 * σn^2) - ((m_phys - m0_phys)^2) / (2 * σm^2))
    U0_fp[i, j] += gauss
end

# Préparation de Becker-Döring
nb_total = N_bd * (M_bd + 1) + 2
C0_bd = zeros(nb_total)
C0_bd[1] = C10_init 
C0_bd[2] = C11_init 

# Duplication stricte dans la grille discrète BD
for m in 0:M_bd, n in 1:N_bd
    C0_bd[idx(n, m, N_bd)] = U0_fp[n, m+1]
end

# ============================================================
# 4. RESOLUTIONS EN AMONT
# ============================================================
println("\n--- Résolution globale de Becker-Döring ---")
p_bd = (N=N_bd, M=M_bd, G10=G10, epsilon=epsilon, beta10=b10, beta11=b11, alpha10=a10, alpha11=a11)
f_ode = ODEFunction(bd2d!; jac = jacobien_bd2d!, jac_prototype = sparse(zeros(nb_total, nb_total)))
prob_bd = ODEProblem(f_ode, C0_bd, (0.0, T_end), p_bd)

sol_bd = solve(prob_bd, KenCarp4(), reltol=1e-5, abstol=1e-20, saveat=t_sauvegarde)

println("\n--- Recueil des snapshots de Fokker-Planck ---")
history_fp = Vector{Matrix{Float64}}()
for t in t_sauvegarde
    print("\rCalcul FP pour t = ", @sprintf("%.2f s", t))
    res_fp_t = FokkerPlanck.run_simulation(FokkerPlanck.DiagonalMultiStencil(P), grid_fp, params_fp, U0_fp, t)
    push!(history_fp, copy(res_fp_t.U))
end
println()

# ============================================================
# 5. GÉNÉRATION DE L'ANIMATION LOGARITHMIQUE TRÈS PRÉCISE
# ============================================================
println("\n--- Création de l'animation Logarithmique (Haute Visibilité) ---")

anim = @animate for (k, t) in enumerate(t_sauvegarde)
    print("\rRendu de la frame : $k / $(length(t_sauvegarde))")
    
    # --- PARTIE A : Fokker-Planck ---
    U_fp = history_fp[k] 
    Z_fp = log10.(max.(U_fp, 1e-15))
    
    # Échelle calée sur l'amplitude maximale (1e-5 -> -5) et le bruit de fond (-12)
    p1 = heatmap(grid_fp.x, grid_fp.y, Z_fp',
                 xlabel="n", ylabel="m",
                 title=@sprintf("Fokker-Planck (Log10) | t = %.2fs", t),
                 aspectratio=1, color=:viridis, clim=(-12, -4))
                 
    # --- PARTIE B : Becker-Döring ---
    C_t_bd = sol_bd.u[k]
    U_bd = zeros(N_bd, M_bd + 1)
    for m in 0:M_bd, n in 1:N_bd
        U_bd[n, m+1] = C_t_bd[idx(n, m, N_bd)]
    end
    Z_bd = log10.(max.(U_bd, 1e-15))
    
    p2 = heatmap(1:N_bd, 0:M_bd, Z_bd',
                 xlabel="n", ylabel="m",
                 title=@sprintf("Becker-Döring (Log10) | t = %.2fs", t),
                 aspectratio=1, color=:viridis, clim=(-12, -4))
                 
    plot(p1, p2, layout=(1, 2), size=(900, 400))
end
println()

gif(anim, "evolution_gaussienne_190_900sec.gif", fps=10)
println("Félicitations ! L'animation stable est disponible dans 'evolution_gaussienne_log.gif'.")


using Printf

println("\n--- Calcul de l'écart moyen en concentration à T_end ---")

# 1. Récupération des matrices finales (dernier snapshot)
U_fp_final = history_fp[end]
C_bd_final = sol_bd.u[end]

# 2. Reconstruction de Becker-Döring sur la géométrie Fokker-Planck
U_bd_projected = zeros(Nx, Ny)
for j in 1:Ny, i in 1:Nx
    n_discret = i         # i=1 correspond à n=1 (ou n=2 selon votre décalage xmin)
    m_discret = j - 1     # j=1 correspond à m=0
    U_bd_projected[i, j] = C_bd_final[idx(n_discret, m_discret, N_bd)]
end

# 3. Calcul de l'écart absolu en chaque point
ecarts = abs.(U_fp_final .- U_bd_projected)

# 4. Moyenne globale (MAE) sur toute la grille
ecart_moyen = sum(ecarts) / length(ecarts)

# 5. Optionnel : Écart maximum pour compléter l'analyse
ecart_max = maximum(ecarts)

@printf("Écart moyen absolu (MAE) : %.4e\n", ecart_moyen)
@printf("Écart maximum constaté   : %.4e\n", ecart_max)


println("\n--- Génération du GIF de la Heatmap de l'Écart ---")

# On prépare l'animation
anim_ecart = @animate for (k, t) in enumerate(t_sauvegarde)
    print("\rCalcul de la frame : $k / $(length(t_sauvegarde)) (t = $(@sprintf("%.2f", t)) s)")
    
    # 1. Extraction des données à l'instant t
    U_fp_t = history_fp[k]
    C_bd_t = sol_bd.u[k]
    
    # 2. Reconstruction / Projection de Becker-Döring sur la géométrie FP
    U_bd_projected = zeros(Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        n_discret = i
        m_discret = j - 1
        U_bd_projected[i, j] = C_bd_t[idx(n_discret, m_discret, N_bd)]
    end
    
    # 3. Calcul de la matrice d'écart absolu à l'instant t
    Ecart_Matrice = abs.(U_fp_t .- U_bd_projected)
    
    # 4. Dessin de la Heatmap pour l'instant t
    heatmap(
        grid_fp.x, 
        grid_fp.y, 
        Ecart_Matrice', 
        xlabel = "Taille d'amas n",
        ylabel = "Taille d'amas m",
        title = @sprintf("Évolution de l'écart absolu |FP - BD| (t = %.2f s)", t),
        color = :inferno,
        clim = (0.0, 1.6e-6),       # Échelle fixe pour éviter que les couleurs sautent d'une image à l'autre
        aspectratio = :equal,
        colorbar_title = "\n Amplitude de l'écart",
        size = (650, 550)
    )
end
println() # Saut de ligne après la boucle

# Sauvegarde sous forme de fichier GIF
gif(anim_ecart, "evolution_ecart_190_900sec.gif", fps = 8)
println("Succès ! Le fichier 'evolution_ecart_2d.gif' a été créé sur votre bureau/dossier courant.")