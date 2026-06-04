"""
Run all experiment scripts at once
"""

using Printf

# Load FokkerPlanck once; each experiment file guards with isdefined.
include(joinpath(@__DIR__, "../src/FokkerPlanck.jl"))
using .FokkerPlanck

const EXPERIMENTS = @__DIR__

function run_experiment(file, label)
    @printf("\n%s\n%s\n%s\n", "="^60, label, "="^60)
    t0 = time()
    include(joinpath(EXPERIMENTS, file))
    @printf("\n  done in %.1f s\n", time() - t0)
end

run_experiment("comparison_cases.jl",   "COMPARISON CASES (3 solver × 3 coupling regimes)")
run_experiment("long_runs.jl",          "LONG RUNS (3 experiments to large T)")
run_experiment("multistencil_cases.jl", "MULTI-STENCIL CASES (P-sweep, convergence, point mass)")

println("\n" * "="^60)
println("All experiments complete. PDFs in FokkerPlanckClean/output/")
println("="^60)
