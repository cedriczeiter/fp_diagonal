module FokkerPlanck

using LinearAlgebra
using SparseArrays
using Printf

include("helpers.jl")
include("grid.jl")
include("physics.jl")
include("initial_conditions.jl")
include("diagnostics.jl")

# AbstractFPSolver must be defined before solver files
abstract type AbstractFPSolver end

include("solvers/naive9point.jl")
include("solvers/multistencil.jl")
include("solvers/kurganov.jl")
include("timestepper.jl")           # run_simulation + step dispatch

export Grid, make_grid
export ModelParams
export Naive9Point, DiagonalMultiStencil, Kurganov
export diagonal_solver, compute_P_min, compute_alpha_max
export run_simulation
export make_gaussian, normalize_to_mass!
export total_mass, material_diagnostics
export compute_kinetics, compute_diffusion_tensor, compute_velocities
export build_diffusion_matrix
export _pack_rowmajor, _unpack_rowmajor!

end # module FokkerPlanck
