"""
forward_solver.py - EIT forward problem solver

Strict implementation per Ito et al. (2025) Section 4:
  PDE: div(sigma grad y) = 0 in Omega,  sigma dy/dn = f on Gamma
  Constraint: int_Gamma y ds = 0

Solution domain: ellipse Omega = {x1^2 + x2^2/0.64 < 1}
Background conductivity: sigma_0 = 1
Inclusion: u = sigma - sigma_0

Noise model (Paper 1 Section 4, FreeFEM Example1.edp L235-238):
  yd(x) = y*(x) + eps*delta(x)*|y_empty(x) - y*(x)|
  delta(x) ~ Uniform(-1, 1), eps = relative noise level
"""

import numpy as np
from .mesh import EllipticMesh
from .fem import (
    assemble_stiffness_matrix,
    assemble_mass_matrix,
    assemble_boundary_load,
    assemble_boundary_mean_constraint,
    solve_neumann_system,
)


# ============================================================
# Inclusion definition functions
# ============================================================

def square_inclusion(x, y, center, half_width):
    """Characteristic function for square inclusion.

    Ref: FreeFEM Example1.edp L22-23:
      func cIndicator = max(0.2000001 - max(abs(x-0.4), abs(y-0.2)),
                            0.2000001 - max(abs(x+0.5), abs(y+0.2)));

    Parameters
    ----------
    x, y : array or scalar -- coordinates
    center : tuple (cx, cy)
    half_width : float

    Returns
    -------
    mask : bool array, True inside the inclusion
    """
    cx, cy = center
    return (np.abs(x - cx) < half_width) & (np.abs(y - cy) < half_width)


def circle_inclusion(x, y, center, radius):
    """Characteristic function for circular inclusion."""
    cx, cy = center
    return (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2


def make_conductivity_example1(mesh):
    """Create true conductivity for Example 1 (EIT).

    FreeFEM Example1.edp L13-14:
      cA = 1.0 (background), cB = 0.01 (inclusion, near-insulator).
      Two square inclusions:
        - center (0.4, 0.2), half-width 0.2
        - center (-0.5, -0.2), half-width 0.2
    """
    cx, cy = mesh.centroids[:, 0], mesh.centroids[:, 1]

    sigma_background = 1.0
    sigma_inclusion = 0.01

    in_inclusion1 = square_inclusion(cx, cy, (0.4, 0.2), 0.2)
    in_inclusion2 = square_inclusion(cx, cy, (-0.5, -0.2), 0.2)

    sigma = np.full(mesh.n_triangles, sigma_background)
    sigma[in_inclusion1 | in_inclusion2] = sigma_inclusion

    u = sigma - sigma_background
    return sigma, u


def make_conductivity_conductive(mesh):
    """Create conductive inclusion example (sigma > sigma_0).

    Same geometry as Example 1 (two squares), but sigma = 3.0 (conductive),
    i.e., u = sigma - sigma_0 = +2.0.
    """
    cx, cy = mesh.centroids[:, 0], mesh.centroids[:, 1]

    sigma_background = 1.0
    sigma_inclusion = 3.0

    in_inclusion1 = square_inclusion(cx, cy, (0.4, 0.2), 0.2)
    in_inclusion2 = square_inclusion(cx, cy, (-0.5, -0.2), 0.2)

    sigma = np.full(mesh.n_triangles, sigma_background)
    sigma[in_inclusion1 | in_inclusion2] = sigma_inclusion

    u = sigma - sigma_background
    return sigma, u


def make_conductivity_single(mesh):
    """Create single circular inclusion example (insulating type).

    Single circular inclusion, center (0.3, 0.0), radius 0.25, sigma = 0.3.
    For single vs multiple inclusion comparison experiments.
    """
    cx, cy = mesh.centroids[:, 0], mesh.centroids[:, 1]

    sigma_background = 1.0
    sigma_inclusion = 0.3

    in_inclusion = circle_inclusion(cx, cy, (0.3, 0.0), 0.25)

    sigma = np.full(mesh.n_triangles, sigma_background)
    sigma[in_inclusion] = sigma_inclusion

    u = sigma - sigma_background
    return sigma, u


# ============================================================
# Forward problem solver
# ============================================================

def solve_forward(mesh, sigma, f_func):
    """Solve EIT forward problem: div(sigma grad y) = 0 in Omega, sigma dy/dn = f on Gamma, int_Gamma y ds = 0."""
    K = assemble_stiffness_matrix(mesh, sigma)
    b = assemble_boundary_load(mesh, f_func)
    B = assemble_boundary_mean_constraint(mesh)

    y = solve_neumann_system(K, b, B)
    return y


def solve_forward_general(mesh, sigma, potential_coeff, f_func, is_boundary_source=True):
    """Solve generalized elliptic forward problem with zero-order term: -div(sigma grad y) + u_p*y = f.

    Weak form: int_Omega sigma grad(y).grad(v) dx + int_Omega u_p*y*v dx = int_Gamma f*v ds
    Used for DOT (Example 3) and other problems with potential term.
    """
    K = assemble_stiffness_matrix(mesh, sigma)

    if potential_coeff is not None:
        M = assemble_mass_matrix(mesh, potential_coeff)
        A = K + M
    else:
        A = K

    if is_boundary_source:
        b = assemble_boundary_load(mesh, f_func)
    else:
        b = _assemble_domain_load(mesh, f_func)

    B = assemble_boundary_mean_constraint(mesh)
    y = solve_neumann_system(A, b, B)
    return y


def _assemble_domain_load(mesh, f_func):
    """Assemble domain source load vector b_i = int_Omega f(x) phi_i dx.

    Uses centroid quadrature (1-point Gauss): int_{T_e} f phi_i dx ~ |T_e|/3 * f(centroid)
    """
    n = mesh.n_points
    b = np.zeros(n)

    cx, cy = mesh.centroids[:, 0], mesh.centroids[:, 1]
    f_vals = f_func(cx, cy)

    for i in range(3):
        np.add.at(b, mesh.triangles[:, i], mesh.areas * f_vals / 3.0)

    return b


# ============================================================
# Cauchy data generation
# ============================================================

def generate_cauchy_data(mesh, sigma_true, source_funcs, noise_level=0.0, rng=None):
    """Generate noisy Cauchy data pairs.

    For each source f_l:
      1. Solve forward problem with inclusion: y_Omega = solve(sigma_true, f_l)
      2. Solve background forward problem: y_empty = solve(sigma_0=1, f_l)
      3. Add noise (Paper 1 Section 4, FreeFEM Example1.edp L235-238):
         yd(x) = y_Omega(x) + eps*delta(x)*|y_Omega(x) - y_empty(x)|
         delta(x) ~ Uniform(-1, 1)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    sigma_background = np.ones(mesh.n_triangles)

    y_omega_list = []
    y_empty_list = []
    y_data_list = []

    for f_func in source_funcs:
        y_omega = solve_forward(mesh, sigma_true, f_func)
        y_empty = solve_forward(mesh, sigma_background, f_func)

        if noise_level > 0:
            delta = 2.0 * rng.random(mesh.n_points) - 1.0
            scattering = np.abs(y_omega - y_empty)
            y_data = y_omega + noise_level * delta * scattering
        else:
            y_data = y_omega.copy()

        y_omega_list.append(y_omega)
        y_empty_list.append(y_empty)
        y_data_list.append(y_data)

    return {
        'y_omega': y_omega_list,
        'y_empty': y_empty_list,
        'y_data': y_data_list,
        'sources': source_funcs,
    }


# ============================================================
# Additional example geometries (Phase 4)
# ============================================================

def make_double_example2(mesh):
    """Create true inclusions for Example 2 (double type, recover conductivity + potential simultaneously).

    FreeFEM Example2.edp:
      type = "double", coef = "unkown"
      sigma_0 = 1.0 (cA), sigma_range = 0.01 (cB), sigma_inclusion = 0.3 (cU)
      v_0 = 1.0 (vA), v_range = 10.0 (vB), v_inclusion = 6.0 (vU)

      Conductivity inclusions (2 squares, same as Example 1):
        - center (0.4, 0.2), half-width 0.2
        - center (-0.5, -0.2), half-width 0.2
      Potential inclusions (2 squares, different locations):
        - center (-0.4, 0.1), half-width 0.2
        - center (0.5, -0.1), half-width 0.2

    Returns
    -------
    sigma, potential, u_sigma, u_potential
    """
    cx, cy = mesh.centroids[:, 0], mesh.centroids[:, 1]

    sigma_bg = 1.0
    sigma_inclusion = 0.3
    potential_bg = 1.0
    potential_inclusion = 6.0

    # Conductivity inclusions (same as Example 1)
    in_c1 = square_inclusion(cx, cy, (0.4, 0.2), 0.2)
    in_c2 = square_inclusion(cx, cy, (-0.5, -0.2), 0.2)

    # Potential inclusions (different locations)
    in_v1 = square_inclusion(cx, cy, (-0.4, 0.1), 0.2)
    in_v2 = square_inclusion(cx, cy, (0.5, -0.1), 0.2)

    sigma = np.full(mesh.n_triangles, sigma_bg)
    sigma[in_c1 | in_c2] = sigma_inclusion

    potential = np.full(mesh.n_triangles, potential_bg)
    potential[in_v1 | in_v2] = potential_inclusion

    u_sigma = sigma - sigma_bg
    u_potential = potential - potential_bg
    return sigma, potential, u_sigma, u_potential


def make_potential_example3(mesh):
    """Create true inclusions for Example 3 (potential-only type, DOT).

    FreeFEM Example3.edp:
      type = "potential", vA = 1e-10, vB = 10.0, vU = 6 (unknown mode)
      sigma_0 = 1 (constant), no conductivity inclusions
      Potential inclusions v:
        - center (-0.6, 0.1), half-width 0.15
        - center (0.5, -0.1), half-width 0.2
    """
    cx, cy = mesh.centroids[:, 0], mesh.centroids[:, 1]

    sigma = np.ones(mesh.n_triangles)
    v_bg = 1e-10
    v_inclusion = 6.0

    in_v1 = square_inclusion(cx, cy, (-0.6, 0.1), 0.15)
    in_v2 = square_inclusion(cx, cy, (0.5, -0.1), 0.2)

    v_coeff = np.full(mesh.n_triangles, v_bg)
    v_coeff[in_v1 | in_v2] = v_inclusion

    u_v = v_coeff - v_bg
    return sigma, v_coeff, u_v


def generate_cauchy_data_general(mesh, sigma_true, potential_true,
                                  source_funcs, noise_level=0.0, rng=None):
    """Generate Cauchy data for the generalized model with zero-order term (DOT).

    -div(sigma grad y) + v*y = 0 in Omega,  sigma dy/dn = f on Gamma

    Same structure as generate_cauchy_data, but uses solve_forward_general
    to solve PDE with potential term.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    sigma_bg = np.ones(mesh.n_triangles)
    potential_bg_val = potential_true.min()
    potential_bg = np.full(mesh.n_triangles, potential_bg_val)

    y_omega_list = []
    y_empty_list = []
    y_data_list = []

    for f_func in source_funcs:
        y_omega = solve_forward_general(mesh, sigma_true, potential_true, f_func)
        y_empty = solve_forward_general(mesh, sigma_bg, potential_bg, f_func)

        if noise_level > 0:
            delta = 2.0 * rng.random(mesh.n_points) - 1.0
            scattering = np.abs(y_omega - y_empty)
            y_data = y_omega + noise_level * delta * scattering
        else:
            y_data = y_omega.copy()

        y_omega_list.append(y_omega)
        y_empty_list.append(y_empty)
        y_data_list.append(y_data)

    return {
        'y_omega': y_omega_list,
        'y_empty': y_empty_list,
        'y_data': y_data_list,
        'sources': source_funcs,
    }
