using DifferentialEquations
using LinearAlgebra
using Plots
using SparseArrays

gr()

# ============================================================
# INDEX
# ============================================================
@inline function idx(n, m, N)
    return n + 1 + (N + 1) * m
end

@inline function idnm(i, N)
    n = (i - 1) % (N + 1)
    m = (i - 1) ÷ (N + 1)
    return n, m
end

# ============================================================
# RHS Becker–Döring 2D
# ============================================================
# On type 'p' de façon lâche ou via un NamedTuple pour éviter les problèmes de performance
function bd2d!(dC, C, p, t)
    # Extraction des paramètres
    N, M, C10, C11, beta10, beta11, alpha10, alpha11 = p

    # CORRECTION CRITIQUE : zero(eltype(C)) au lieu de 0.0 pour ForwardDiff / Rodas5
    fill!(dC, zero(eltype(C))) 

    for m in 0:M
        for n in 0:N

            i = idx(n, m, N)

            # perte locale
            loss = alpha10(n,m) + beta10(n,m)*C10 +
                   alpha11(n,m) + beta11(n,m)*C11

            dC[i] -= loss * C[i]

            # gain from (n-1,m)
            if n > 0
                j = idx(n-1, m, N)
                dC[i] += beta10(n-1,m) * C10 * C[j]
            end

            # gain from (n+1,m)
            if n < N
                j = idx(n+1, m, N)
                dC[i] += alpha10(n+1,m) * C[j]
            end

            # diagonal couplings
            if n > 0 && m > 0
                j = idx(n-1, m-1, N)
                dC[i] += beta11(n-1,m-1) * C11 * C[j]
            end

            if n < N && m < M
                j = idx(n+1, m+1, N)
                dC[i] += alpha11(n+1,m+1) * C[j]
            end
        end
    end
end


# ============================================================
# JACOBIEN A LA MAIN
# ============================================================

function jacobien_bd2d!(J, C, p, t)
    N, M, G10, epsilon, beta10, beta11, alpha10, alpha11 = p
    
    C10 = C[1]
    C11 = C[2]
    
    # On commence par vider la matrice du pas précédent
    fill!(J, 0.0)
    
    # --- Dérivées par rapport à la génération et aux monomères ---
    # d(dC1/dt)/dC1 ...
    
    # --- Remplissage pour les amas (n, m) ---
    for m in 0:M
        for n in 0:N
            i = idx(n, m, N)
            
            # Exemple : Dérivée du flux f_10 = beta10*C10*C[i] - alpha10*C[idx(n+1,m)]
            if n < N
                i_suivant = idx(n+1, m, N)
                
                # Effet sur l'amas actuel (i)
                J[i, i]         -= beta10(n,m) * C10
                J[i, i_suivant] += alpha10(n+1, m)
                J[i, 1]         -= beta10(n,m) * C[i] # Dérivée / C10
                
                # Effet sur l'amas suivant (i_suivant)
                J[i_suivant, i]         += beta10(n,m) * C10
                J[i_suivant, i_suivant] -= alpha10(n+1, m)
                J[i_suivant, 1]         += beta10(n,m) * C[i]
            end
            
            # (Tu répètes la même logique pour les flux croisés en m avec beta11)
        end
    end
end

# ============================================================
# PARAMÈTRES
# ============================================================
N = 90
M = 90

# CORRECTION 1 : Le nombre total inclut la grille d'amas ET les 2 monomères
nb_amas = (N + 1) * (M + 1)
nb_total = nb_amas + 2 

# Position du centre
n0 = div(N, 4)
m0 = div(M, 4)

# Étalement de la cloche
σn = 0.5  
σm = 0.5  

# CORRECTION 2 : On initialise le vecteur à la BONNE taille globale (6563 cases)
C0 = zeros(nb_total)

# Initialisation des monomères (Optionnel mais recommandé pour lancer la physique)
C0[1] = 1e-4  # C(1,0)
C0[2] = 1e-4  # C(0,1)

# Remplissage de la cloche ultra-serrée
for m in 0:M, n in 0:N
    i = idx(n, m, N)
    if 3 <= i <= length(C0)
        C0[i] = exp(-((n-n0)^2)/(2*σn^2) - ((m-m0)^2)/(2*σm^2))
    end
end

# Normalisation à la valeur totale souhaitée
C_totale = 0.1
# CORRECTION 3 : Sécurité pour éviter une division par zéro si la somme est nulle
somme_brute = sum(C0[3:end])
if somme_brute > 0
    C0[3:end] .= (C0[3:end] ./ somme_brute) .* C_totale
end

# paramètres
C10 = 0.05  
C11 = 0.02
# Définition propre des fonctions
b10(n,m) = sqrt(n+1)
b11(n,m) = 0.3*sqrt(n+1)
a10(n,m) = 1e-2
a11(n,m) = 5e-3

p = (N=N, M=M, C10=C10, C11=C11, beta10=b10, beta11=b11, alpha10=a10, alpha11=a11)

# ============================================================
# PROBLÈME ODE
# ============================================================
# On déclare que notre système possède une fonction de flux ET un Jacobien analytique
f_ode = ODEFunction(bd2d_realiste!; 
                    jac = jacobien_bd2d!, 
                    jac_prototype = sparse(zeros(nb_total, nb_total)))

prob = ODEProblem(f_ode, C0, (0.0, 50.0), p)
sol = solve(prob, Rosenbrock23(), reltol=1e-5)


# ============================================================
# RECONSTRUCTION MATRICE
# ============================================================
function reshapeC(u, N, M)
    C = zeros(N+1, M+1)
    for m in 0:M, n in 0:N
        i = idx(n,m,N)
        C[n+1, m+1] = u[i]
    end
    return C
end

# snapshots
snapshots = [reshapeC(sol(t), N, M) for t in range(0, 100, length=40)]
times = range(0, 100, length=40)

# ============================================================
# PLOTS
# ============================================================
function plotheatmap(C; t=0.0, logscale=false)
    # CORRECTION VISU : Pas de transposée si on veut n en X et m en Y 
    # car heatmap(x, y, Z) prend Z où les colonnes sont X et les lignes sont Y.
    Z = copy(C) 
    if logscale
        Z .= log10.(Z .+ 1e-16)
    end

    # On passe explicitement les axes 0:N et 0:M pour correspondre aux labels
    heatmap(0:N, 0:M, Z', xlabel="n", ylabel="m", title="t = $t", aspectratio=1)
end

function animatesolution(snapshots, times)
    anim = @animate for (C, t) in zip(snapshots, times)
        plotheatmap(C, t=t, logscale=true)
    end
    gif(anim, "bd2d_solveur150.gif", fps=5)
end

function plotclusters(snapshots, times, indices)
    p_curve = plot(xlabel="t", ylabel="C(n,m)", title="Évolution des amas solveur")

    for (n,m) in indices
        vals = [S[n+1,m+1] for S in snapshots]
        plot!(p_curve, times, vals, label="($n,$m)")
    end

    display(p_curve)
end

# ============================================================
# VISU
# ============================================================
Cfinal = reshapeC(sol(20.0), N, M)

display(plotheatmap(Cfinal, t=20.0, logscale=false))
display(plotheatmap(Cfinal, t=20.0, logscale=true))

plotclusters(snapshots, times, [(10,10), (15,15), (20,20)])
animatesolution(snapshots, times)

println("Animation sauvegardée dans bd2d_solveur.gif")