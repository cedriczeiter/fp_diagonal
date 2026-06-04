using SparseArrays
using LinearAlgebra
using Plots

gr()

@inline function idx(n::Int, m::Int, N::Int)
    return n + 1 + (N + 1) * m
end

function assemble_BD2D_matrix(
    C_old, dt, C10, C11,
    beta10, beta11,
    alpha10, alpha11
)
    N = size(C_old, 1) - 1
    M = size(C_old, 2) - 1
    nb = (N + 1) * (M + 1)

    rows = Int[]
    cols = Int[]
    vals = Float64[]
    b = zeros(Float64, nb)

    for m in 0:M
        for n in 0:N
            i = idx(n, m, N)
            b[i] = C_old[n+1, m+1]

            diag =
                1.0 +
                dt * alpha10(n, m) +
                dt * beta10(n, m) * C10 +
                dt * alpha11(n, m) +
                dt * beta11(n, m) * C11

            push!(rows, i); push!(cols, i); push!(vals, diag)

            if n >= 1
                j = idx(n-1, m, N)
                push!(rows, i); push!(cols, j)
                push!(vals, -dt * beta10(n-1, m) * C10)
            end

            if n <= N-1
                j = idx(n+1, m, N)
                push!(rows, i); push!(cols, j)
                push!(vals, -dt * alpha10(n+1, m))
            end

            if n >= 1 && m >= 1
                j = idx(n-1, m-1, N)
                push!(rows, i); push!(cols, j)
                push!(vals, -dt * beta11(n-1, m-1) * C11)
            end

            if n <= N-1 && m <= M-1
                j = idx(n+1, m+1, N)
                push!(rows, i); push!(cols, j)
                push!(vals, -dt * alpha11(n+1, m+1))
            end
        end
    end

    A = sparse(rows, cols, vals, nb, nb)
    return A, b
end

function step_BD2D!(
    C_new, C_old, dt,
    C10, C11,
    beta10, beta11,
    alpha10, alpha11
)
    N = size(C_old, 1) - 1
    M = size(C_old, 2) - 1

    A, b = assemble_BD2D_matrix(
        C_old, dt, C10, C11,
        beta10, beta11,
        alpha10, alpha11
    )

    U_new = A \ b

    for m in 0:M
        for n in 0:N
            C_new[n+1, m+1] = U_new[idx(n, m, N)]
        end
    end
end

function solve_BD2D_history(
    C0;
    dt,
    nt,
    save_every,
    C10,
    C11,
    beta10,
    beta11,
    alpha10,
    alpha11
)
    C_old = copy(C0)
    C_new = similar(C_old)

    snapshots = [copy(C_old)]
    times = [0.0]

    for k in 1:nt
        step_BD2D!(
            C_new, C_old, dt,
            C10, C11,
            beta10, beta11,
            alpha10, alpha11
        )

        C_old, C_new = C_new, C_old

        if k % save_every == 0
            push!(snapshots, copy(C_old))
            push!(times, k * dt)
        end
    end

    return C_old, snapshots, times
end

function plot_heatmap(C; t=0.0, logscale=false)
    Z = copy(C')
    if logscale
        Z .= log10.(Z .+ 1e-16)
    end

    heatmap(
        Z,
        xlabel="n",
        ylabel="m",
        title="t = $t",
        aspect_ratio=1
    )
end

function animate_solution(snapshots, times)
    anim = @animate for (C, t) in zip(snapshots, times)
        plot_heatmap(C, t=t, logscale=true)
    end
    gif(anim, "bd2d.gif", fps=5)
end

function plot_clusters(snapshots, times, indices)
    p = plot(xlabel="t", ylabel="C_{n,m}(t)", title="Évolution de quelques amas")
    for (n, m) in indices
        vals = [S[n+1, m+1] for S in snapshots]
        plot!(p, times, vals, label="($n,$m)")
    end
    display(p)
end

# ============================================================
# PARAMÈTRES
# ============================================================

N = 40
M = 40

n0 = N ÷ 3
m0 = M ÷ 3
σn = max(2.0, N / 10)
σm = max(2.0, M / 10)

C0 = zeros(Float64, N+1, M+1)
for m in 0:M
    for n in 0:N
        C0[n+1, m+1] = exp(
            -((n - n0)^2) / (2σn^2)
            -((m - m0)^2) / (2σm^2)
        )
    end
end
C0 ./= sum(C0)

C10 = 1e-3
C11 = 5e-4

beta10(n,m) = sqrt(n + 1)
beta11(n,m) = 0.3 * sqrt(n + 1)

alpha10(n,m) = 1e-2
alpha11(n,m) = 5e-3

dt = 0.2
nt = 100
save_every = 5

C_final, snapshots, times = solve_BD2D_history(
    C0;
    dt=dt,
    nt=nt,
    save_every=save_every,
    C10=C10,
    C11=C11,
    beta10=beta10,
    beta11=beta11,
    alpha10=alpha10,
    alpha11=alpha11
)

display(plot_heatmap(C_final, t=times[end], logscale=false))
display(plot_heatmap(C_final, t=times[end], logscale=true))
plot_clusters(snapshots, times, [(10,10), (15,15), (20,20)])
animate_solution(snapshots, times)
println("Animation sauvegardée dans bd2d.gif")