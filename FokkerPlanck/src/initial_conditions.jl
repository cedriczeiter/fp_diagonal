"""
Unnormalised Gaussian initial condition on the grid.
"""
function make_gaussian(
    grid::Grid;
    cx::Float64 = 0.5*(grid.x[1]+grid.x[end]),
    cy::Float64 = 0.5*(grid.y[1]+grid.y[end]),
    sigma::Float64 = 0.2 * (grid.x[end]-grid.x[1]),
)
    return [exp(-((x-cx)^2 + (y-cy)^2) / (2*sigma^2))
            for x in grid.x, y in grid.y]
end

"""
Scale U in-place so that integral U dx dy = target.
"""
function normalize_to_mass!(U::AbstractMatrix, grid::Grid, target::Float64=1.0)
    m = total_mass(U, grid)
    m > 0 || error("Cannot normalise: total mass is non-positive.")
    U .*= target / m
    return U
end
