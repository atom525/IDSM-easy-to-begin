"""
fem.py — P1 finite element common interface (delegates to scikit-fem backend)

All downstream modules (forward_solver, dsm, idsm, idsm_partial) import FEM functions through this file.
Uses scikit-fem implementation (fem_skfem.py) by default, can switch back to manual version via environment variable:

    IDSM_FEM_LEGACY=1 python -m pytest tests/

Public API (signatures unchanged):
  - assemble_stiffness_matrix(mesh, sigma)
  - assemble_mass_matrix(mesh, coeff=None)
  - assemble_boundary_mass_matrix(mesh)
  - assemble_boundary_load(mesh, f_func)
  - assemble_boundary_mean_constraint(mesh)
  - solve_neumann_system(K, b, B)
  - solve_robin_system(mesh, A_op, alpha, v)
  - compute_boundary_normal_flux(mesh, sigma, y)
  - assemble_partial_boundary_mass_matrix(mesh, gamma_d_node_mask)
  - compute_boundary_normal_derivative(mesh, z, sigma_bg=1.0)  [new]
"""

import os

_USE_LEGACY = os.getenv("IDSM_FEM_LEGACY", "0").strip().lower() in ("1", "true")

if _USE_LEGACY:
    from .fem_legacy import *  # noqa: F401, F403
else:
    from .fem_skfem import *  # noqa: F401, F403
