using DifferentialEquations
using LinearAlgebra
using Plots
using SparseArrays

gr()

# ============================================================
# INDEX (La grille commence à n=1, m=0)
# ============================================================
@inline function idx(n, m, N)
    # n ∈ [1, N], m ∈ [0, M]
    # +2 pour laisser les deux premières places du vecteur à C10 et C11
    return n + N * m + 2
end

# ============================================================
# PARAMÈTRES ÉNERGÉTIQUES ET TAUX DE RÉACTION
# ============================================================

# Énergies d'activation E(1,0) et E(1,1) en fonction de la taille totale x = n + m
function E10(n, m)
    x = n + m
    if x < 1
        return 0.0
    end
    # Utilisation de cbrt(x)^2 pour simuler x^(2/3) de façon stable pour l'autodiff
    return 24.47 - 34.0 * (cbrt(x)^2 - cbrt(x - 1)^2)
end

function E11(n, m)
    x = n + m
    if x < 1
        return 0.0
    end
    return 27.47 - 34.0 * (cbrt(x)^2 - cbrt(x - 1)^2)
end

# Taux de condensation (Bêta)
# Équation (15) : β(1,0) = x^(1/2)
function b10(n, m)
    x = n + m
    return sqrt(max(x, 0.0))
end

# Équation (16) : β(1,1) = ϵ * x^(1/2)
# On passe epsilon en argument de la fonction pour pouvoir le faire varier
function b11(n, m, epsilon)
    x = n + m
    return epsilon * sqrt(max(x, 0.0))
end

# Taux d'évaporation (Alpha)
# Équation (17) : α(1,0) = β(1,0) * exp(-E(1,0))
function a10(n, m)
    return b10(n, m) * exp(-E10(n, m))
end

# Équation (18) : α(1,1) = β(1,1) * exp(-E(1,1))
function a11(n, m, epsilon)
    return b11(n, m, epsilon) * exp(-E11(n, m))
end
# ============================================================
# RHS Becker–Döring 2D
# ============================================================
function bd2d!(dC, C, p, t)
    # On extrait 'epsilon' du tuple p
    N, M, G10, epsilon, beta10, beta11, alpha10, alpha11 = p

    fill!(dC, zero(eltype(C))) 
    C10 = C[1]
    C11 = C[2]
    dC[1] = G10
    dC[2] = 0.0

    for m in 0:M
        for n in 1:N
            i = idx(n, m, N)

            # --- Utilisation des fonctions avec les bons arguments ---
            # Attention : beta11 et alpha11 ont besoin du paramètre 'epsilon'
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

            # Consommation des monomères (appliquer la même logique avec fac et epsilon)
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
# JACOBIEN EXACT FAIT À LA MAIN
# ============================================================
function jacobien_bd2d!(J, C, p, t)
    # AJOUT de epsilon ici pour respecter l'ordre du tuple p !
    N, M, G10, epsilon, beta10, beta11, alpha10, alpha11 = p
    
    C10 = C[1]
    C11 = C[2]
    
    fill!(J, 0.0)
    
    for m in 0:M
        for n in 1:N
            i = idx(n, m, N)
            
            # On évalue les taux locaux (en passant epsilon à beta11 et alpha11)
            β10 = beta10(n, m)
            β11 = beta11(n, m, epsilon)
            α10 = alpha10(n, m)
            α11 = alpha11(n, m, epsilon)
            
            # 1. Dérivées par rapport aux amas (Lignes i >= 3)
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
                J[i, j] += beta11(n-1, m-1, epsilon) * C11  # Ajout epsilon
                J[i, 2] += beta11(n-1, m-1, epsilon) * C[j]  # Ajout epsilon
            end
            if n < N && m < M
                j = idx(n+1, m+1, N)
                J[i, j] += alpha11(n+1, m+1, epsilon)        # Ajout epsilon
            end
            
            # 2. Dérivées des Monomères (Lignes 1 et 2)
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
                J[2, i_diag] += fac * alpha11(n+1, m+1, epsilon)  # Ajout epsilon
                J[2, 2]      -= fac * β11 * C[i]
            end
        end
    end
    return nothing
end