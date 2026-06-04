struct Grid
    Nx::Int
    Ny::Int
    x::Vector{Float64}          # cell centres, length Nx
    y::Vector{Float64}          # cell centres, length Ny
    dx::Float64                 # Δx_0 (base / minimum x-spacing)
    dy::Float64                 # Δy_0 (base / minimum y-spacing)
    r::Float64                  # dx/dy = r_0 (base aspect ratio)
    alpha::Float64              # geometric growth factor α  (0.0 = uniform)
    dx_cells::Vector{Float64}   # per-column widths  dx_cells[i] = dx*(1+α)^(i-1)
    dy_cells::Vector{Float64}   # per-row    heights dy_cells[j] = dy*(1+α)^(j-1)
end

"""
Build a node-centred grid over [xmin,xmax] × [ymin,ymax].

For `alpha = 0` (default) the grid is uniform and behaviour is identical to the
original implementation.  For `alpha > 0` cell widths grow geometrically:

    Δx_i = Δx_0 (1+α)^(i-1),   Δy_j = Δy_0 (1+α)^(j-1),   i,j ≥ 1.

The base spacings Δx_0, Δy_0 are chosen so that the cells tile the domain
exactly.  `r = dx/dy` stores the base aspect ratio r_0 = Δx_0/Δy_0.

The x-domain must satisfy xmin >= 2 (energy formulas in physics.jl require x >= 2).
"""
function make_grid(;
    xmin::Float64 = 2.0,
    xmax::Float64 = 10.0,
    Nx::Int,
    ymin::Float64 = 0.0,
    ymax::Float64 = 1.0,
    Ny::Int,
    alpha::Float64 = 0.0,
)
    xmin >= 2.0 || error("xmin=$(xmin) < 2: energy formulas require x >= 2.")
    xmax > xmin  || error("xmax must be > xmin.")
    ymax > ymin  || error("ymax must be > ymin.")
    Nx >= 2 && Ny >= 2 || error("Need Nx >= 2 and Ny >= 2.")
    alpha >= 0.0 || error("alpha must be >= 0 (negative growth not supported).")

    if alpha == 0.0
        x  = collect(range(xmin, xmax, length=Nx))
        y  = collect(range(ymin, ymax, length=Ny))
        dx = x[2] - x[1]
        dy = y[2] - y[1]
        return Grid(Nx, Ny, x, y, dx, dy, dx/dy, 0.0, fill(dx, Nx), fill(dy, Ny))
    else
        # Δx_0 = (xmax-xmin)*α / ((1+α)^Nx - 1)  so that sum_i Δx_i = xmax-xmin
        dx0 = (xmax - xmin) * alpha / ((1 + alpha)^Nx - 1)
        dy0 = (ymax - ymin) * alpha / ((1 + alpha)^Ny - 1)

        dx_cells = [dx0 * (1 + alpha)^(i-1) for i in 1:Nx]
        dy_cells = [dy0 * (1 + alpha)^(j-1) for j in 1:Ny]

        # cell centres = left edge of cell + half cell width
        x = zeros(Nx)
        pos = xmin
        for i in 1:Nx
            x[i] = pos + dx_cells[i] / 2
            pos  += dx_cells[i]
        end

        y = zeros(Ny)
        pos = ymin
        for j in 1:Ny
            y[j] = pos + dy_cells[j] / 2
            pos  += dy_cells[j]
        end

        return Grid(Nx, Ny, x, y, dx0, dy0, dx0/dy0, alpha, dx_cells, dy_cells)
    end
end
