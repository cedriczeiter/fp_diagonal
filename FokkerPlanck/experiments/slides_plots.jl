if !isdefined(Main, :FokkerPlanck)
    include(joinpath(@__DIR__, "../src/FokkerPlanck.jl"))
end
using .FokkerPlanck, Plots
import Plots: mm

const OUTDIR = joinpath(@__DIR__, "../output")
mkpath(OUTDIR)

# Slide friendly defaults
default(
    titlefontsize=13,
    guidefontsize=12,
    tickfontsize=10,
    legendfontsize=10,
    linewidth=2.0,
    left_margin=8mm,
    bottom_margin=6mm,
    right_margin=4mm,
    top_margin=4mm,
)

function run_solver_set(solvers, grid, params, U0, T_end)
    results = []
    for (label, solver, color) in solvers
        res = run_simulation(solver, grid, params, copy(U0), T_end; fixed_mobile=true)
        push!(results, (label=label, res=res, color=color))
    end
    return results
end

function plot_min_c(results; title="", outpath)
    p = plot(
        title=title,
        xlabel="t",
        ylabel="min C",
        legend=:bottomleft,
        size=(680, 420),
    )
    for r in results
        if r.label != "Diagonal IMEX"
            plot!(p, r.res.times, r.res.min_u_hist,
                  label=r.label, color=r.color, ls=:solid, lw=2.0)
        end
    end
    for r in results
        if r.label == "Diagonal IMEX"
            plot!(p, r.res.times, r.res.min_u_hist,
                  label=r.label, color=r.color, ls=:solid, lw=6.0)
        end
    end
    hline!(p, [0.0], ls=:dash, color=:black, lw=1, label="")
    savefig(p, outpath)
end

function plot_profiles(results, grid; outpath)
    pos_vals = vcat([max.(r.res.U[:], 0.0) for r in results]...)
    pos_min = 0.0
    pos_max = maximum(pos_vals)

    panels = []
    for r in results
        U = r.res.U
        U_pos = copy(U)
        U_pos[U_pos .< 0.0] .= NaN
        neg_mask = Float64.(U .< 0.0)
        neg_mask[neg_mask .== 0.0] .= NaN

        p = heatmap(
            grid.x,
            grid.y,
            U_pos',
            title=r.label,
            xlabel="x",
            ylabel="y",
            color=:viridis,
            clims=(pos_min, pos_max),
            colorbar=true,
        )
        heatmap!(
            p,
            grid.x,
            grid.y,
            neg_mask',
            color=[:white, :red],
            clims=(0.0, 1.0),
            colorbar=false,
            alpha=0.9,
        )
        push!(panels, p)
    end

    savefig(plot(panels..., layout=(1, length(panels)), size=(1020, 310),
                 plot_title=""), outpath)
end

# Experiment 1 — r=1.0 (square cells), sigma=0.9
# At r=1.0 the SE/NW off-diagonals of Naive9Point are maximally large AND Kurganov's
# cross-diffusion CFL is not accounted for → both fail at comparable magnitude (~1e-6).
# DiagonalMultiStencil(1) uses only the (1,1) stencil (Ay=0 at r=r_lo=1), stays ≥ 0.
grid = make_grid(xmin=2.0, xmax=90.0, Nx=177, ymin=2.0, ymax=36.0, Ny=69)
params = ModelParams(1.0, 1.0, 1.0, 1e-18; r=grid.r, sigma=0.9)
X_MID = 0.5 * (grid.x[1] + grid.x[end])
U0 = Float64[grid.x[i] <= X_MID ? 1.0 : 0.0 for i in 1:grid.Nx, _ in 1:grid.Ny]
T_end = 5.0

solvers1 = [
    ("Naive 9-point",  Naive9Point(),           RGB(0.45, 0.45, 0.45)),
    ("Diagonal IMEX",  DiagonalMultiStencil(1), RGB(0/255, 148/255, 68/255)),
    ("Kurganov-Tadmor", Kurganov(),              RGB(200/255, 0/255, 30/255)),
]

res1 = run_solver_set(solvers1, grid, params, U0, T_end)
plot_min_c(res1, outpath=joinpath(OUTDIR, "slides_exp1.pdf"))

# Two profile snapshots: t=0.10 (failures just starting) and t=0.25 (near peak)
res1_t010 = run_solver_set(solvers1, grid, params, U0, 0.10)
plot_profiles(res1_t010, grid,
    outpath=joinpath(OUTDIR, "slides_exp1_profiles_t010.pdf"))

res1_t025 = run_solver_set(solvers1, grid, params, U0, 0.25)
plot_profiles(res1_t025, grid,
    outpath=joinpath(OUTDIR, "slides_exp1_profiles_t025.pdf"))

println("Saved slides_exp1.pdf, slides_exp1_profiles_t010.pdf, slides_exp1_profiles_t025.pdf")
