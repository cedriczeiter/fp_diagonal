# IMI Project : Becker-Döring & Fokker-Planck Solvers

This repository accompanies the report produced during a Spring 2026 **IMI Project** at École des Ponts, IP Paris, supervised by **Laurent Monasse** (INRIA) and
**Thomas Jourdan** (CEA).

The project studies numerical schemes for the Becker-Döring coagulation model and its
continuous Fokker-Planck mean-field approximation on a 2D grid.
A central and novel contribution is a new **diagonal multi-stencil scheme** that guarantees
positivity of the solution for all parameter regimes.

## Repository structure

| Folder | Description |
|--------|-------------|
| `BeckerDoring/` | Exact solver for the Becker-Döring ODE system |
| `FokkerPlanck/` | Three Julia solvers for the 2D Fokker-Planck PDE, including the new diagonal multi-stencil scheme (see `FokkerPlanck/README.md`) |
| `presentations/` | Various kinds of reports showing progress |

## Team

| Name | Affiliation | Contribution |
|------|-------------|-------------|
| Cédric-Yséry Zeiter | ETH Zürich (exchange at ENPC) | FokkerPlanck implementation & analysis |
| Yassine Guennouni Assimi | ENPC | Becker-Döring implementation & analysis |
| Amitaï Berger | ENPC | Becker-Döring implementation & analysis |
