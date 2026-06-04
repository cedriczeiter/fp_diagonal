"""
Integrate U over the grid using per-cell areas (correct for both uniform and
geometric grids).
"""
function total_mass(U::AbstractMatrix, grid::Grid)
    s = 0.0
    for i in 1:grid.Nx, j in 1:grid.Ny
        s += U[i,j] * grid.dx_cells[i] * grid.dy_cells[j]
    end
    return s
end

"""
Compute conserved material quantities:
- mass   = integral U dx dy
- Y_mat  = C11 + integral y U dx dy       (y counts type-11 units per cluster; C11 is free type-11)
- X_mat  = C10 + C11 + integral x U dx dy (x counts total monomers per cluster; both free species contribute)
"""
function material_diagnostics(
    U::AbstractMatrix,
    grid::Grid,
    C10::Float64,
    C11::Float64,
)
    mass  = 0.0
    x_mom = 0.0
    y_mom = 0.0
    for i in 1:grid.Nx, j in 1:grid.Ny
        cell_area = grid.dx_cells[i] * grid.dy_cells[j]
        mass  += U[i,j] * cell_area
        x_mom += grid.x[i] * U[i,j] * cell_area
        y_mom += grid.y[j] * U[i,j] * cell_area
    end
    return (mass=mass, Y_mat=C11+y_mom, X_mat=C10+C11+x_mom)
end
