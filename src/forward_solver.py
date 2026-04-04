"""
forward_solver.py - EIT Forward Problem Solver

Implements the forward solver strictly per Ito et al. (2025) Section 4:
  PDE: ∇·(σ∇y) = 0 in Ω,  σ ∂y/∂n = f on Γ
  Constraint: ∫_Γ y ds = 0

Domain: Ellipse Ω = {x₁² + x₂²/0.64 < 1}
Background: σ₀ = 1
Inclusion: u = σ − σ₀

Noise model (Paper 1 Section 4, also FreeFEM Example1.edp L235-238):
  yd(x) = y*(x) + ε·δ(x)·|y_∅(x) − y*(x)|
  δ(x) ~ Uniform(−1, 1), ε = relative noise level
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
    """Characteristic function of a square inclusion.

    Reference: FreeFEM Example1.edp L22-23:
      func cIndicator = max(0.2000001 - max(abs(x-0.4), abs(y-0.2)),
                            0.2000001 - max(abs(x+0.5), abs(y+0.2)));

    Parameters
    ----------
    x, y : array or scalar — coordinates
    center : tuple (cx, cy)
    half_width : float

    Returns
    -------
    mask : bool array, True inside the inclusion
    """
    cx, cy = center
    return (np.abs(x - cx) < half_width) & (np.abs(y - cy) < half_width)


def circle_inclusion(x, y, center, radius):
    """Characteristic function of a circular inclusion.

    Parameters
    ----------
    x, y : array or scalar — coordinates
    center : tuple (cx, cy)
    radius : float

    Returns
    -------
    mask : bool array
    """
    cx, cy = center
    return (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2


def make_conductivity_example1(mesh):
    """Create the true conductivity for Example 1 (EIT).

    Paper 1 Section 4.1:
      σ₀ = 1 (background)
      σ = 0.3 inside inclusions (i.e., u = −0.7)
      Two square inclusions:
        - center (0.4, 0.2), half-width 0.2
        - center (−0.5, −0.2), half-width 0.2

    Reference: FreeFEM Example1.edp L22-23.

    Parameters
    ----------
    mesh : EllipticMesh

    Returns
    -------
    sigma : array (M,) — conductivity per triangle (P0)
    u : array (M,) — inclusion u = σ − σ₀
    """
    cx, cy = mesh.centroids[:, 0], mesh.centroids[:, 1]

    sigma_background = 1.0
    sigma_inclusion = 0.3

    in_inclusion1 = square_inclusion(cx, cy, (0.4, 0.2), 0.2)
    in_inclusion2 = square_inclusion(cx, cy, (-0.5, -0.2), 0.2)

    sigma = np.full(mesh.n_triangles, sigma_background)
    sigma[in_inclusion1 | in_inclusion2] = sigma_inclusion

    u = sigma - sigma_background
    return sigma, u


def make_conductivity_conductive(mesh):
    """Create a conductive inclusion example (σ > σ₀).

    Same geometry as Example 1 (two squares), but with σ = 3.0
    inside inclusions (conductive), so u = σ − σ₀ = +2.0.

    Parameters
    ----------
    mesh : EllipticMesh

    Returns
    -------
    sigma : array (M,) — conductivity per triangle (P0)
    u : array (M,) — inclusion u = σ − σ₀
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
    """Create a single circular inclusion example (insulating).

    One circular inclusion centered at (0.3, 0.0) with radius 0.25,
    σ = 0.3 inside. Used for single vs multiple inclusion comparison.

    Parameters
    ----------
    mesh : EllipticMesh

    Returns
    -------
    sigma : array (M,) — conductivity per triangle (P0)
    u : array (M,) — inclusion u = σ − σ₀
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
# Forward problem solvers
# ============================================================

def solve_forward(mesh, sigma, f_func):
    """Solve the EIT forward problem.

    ∇·(σ∇y) = 0 in Ω,  σ ∂y/∂n = f on Γ,  ∫_Γ y ds = 0

    Parameters
    ----------
    mesh : EllipticMesh
    sigma : array (M,) or scalar — conductivity
    f_func : callable — boundary source f(x, y)

    Returns
    -------
    y : array (N,) — full-domain FEM solution
    """
    K = assemble_stiffness_matrix(mesh, sigma)
    b = assemble_boundary_load(mesh, f_func)
    B = assemble_boundary_mean_constraint(mesh)

    y = solve_neumann_system(K, b, B)
    return y


def solve_forward_general(mesh, sigma, potential_coeff, f_func, is_boundary_source=True):
    """Solve a generalized elliptic forward problem with a zeroth-order term.

    −∇·(σ∇y) + u_p·y = f

    Weak form: ∫_Ω σ∇y·∇v dx + ∫_Ω u_p·y·v dx = ∫_Γ f·v ds (or ∫_Ω f·v dx)

    Used for DOT (Example 3), CE (Example 4), etc.

    Parameters
    ----------
    mesh : EllipticMesh
    sigma : array (M,)
    potential_coeff : array (M,) or None — zeroth-order coefficient
    f_func : callable
    is_boundary_source : bool — True for boundary source, False for domain source

    Returns
    -------
    y : array (N,)
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
    """Assemble the domain source load vector b_i = ∫_Ω f(x) φ_i dx.

    Uses centroid quadrature (1-point Gauss): ∫_{T_e} f φ_i dx ≈ |T_e|/3 * f(centroid)
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

    For each source f_ℓ:
      1. Solve forward problem with inclusions: y_Ω = solve(σ_true, f_ℓ)
      2. Solve background forward problem: y_∅ = solve(σ₀=1, f_ℓ)
      3. Add noise (Paper 1 Section 4, FreeFEM Example1.edp L235-238):
         yd(x) = y_Ω(x) + ε·δ(x)·|y_Ω(x) − y_∅(x)|
         δ(x) ~ Uniform(−1, 1)

    Parameters
    ----------
    mesh : EllipticMesh
    sigma_true : array (M,) — true conductivity
    source_funcs : list of callable — boundary source functions
    noise_level : float — relative noise ε
    rng : numpy Generator (optional, for reproducibility)

    Returns
    -------
    data : dict with keys:
        'y_omega' : list of (N,) — solutions with inclusions
        'y_empty' : list of (N,) — background solutions
        'y_data'  : list of (N,) — noisy measurements
        'sources' : same as source_funcs
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

def make_potential_example3(mesh):
    """Create the true inclusion for Example 3 (potential-only, DOT).

    FreeFEM Example3.edp:
      type = "potential", vA = 1e-10, vB = 10.0, vU = 6 (unknown mode)
      σ₀ = 1 (constant), no conductivity inclusions
      Potential inclusions v:
        - center (−0.6, 0.1), half-width 0.15
        - center (0.5, −0.1), half-width 0.2

    Parameters
    ----------
    mesh : EllipticMesh

    Returns
    -------
    sigma : array (M,) — constant conductivity = 1.0
    v_coeff : array (M,) — potential coefficient (P0)
    u_v : array (M,) — potential inclusion u_v = v_coeff − v_bg
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
    """Generate Cauchy data for the generalized model with zeroth-order term (DOT).

    −∇·(σ∇y) + v·y = 0 in Ω,  σ ∂y/∂n = f on Γ

    Same structure as generate_cauchy_data but uses solve_forward_general
    for PDEs with a potential term.

    Parameters
    ----------
    mesh : EllipticMesh
    sigma_true : array (M,)
    potential_true : array (M,)
    source_funcs : list of callable
    noise_level : float
    rng : numpy Generator

    Returns
    -------
    data : dict with y_omega, y_empty, y_data, sources
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
