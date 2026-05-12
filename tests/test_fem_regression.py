"""Regression tests: numerical consistency between fem_skfem and fem_legacy.

For each FEM assembly/solve function, call both backends with the same input
and assert results agree to machine precision.
"""

import numpy as np
import pytest

from src.mesh import generate_elliptic_mesh
from src import fem_skfem as skfem
from src import fem_legacy as legacy


@pytest.fixture
def mesh():
    return generate_elliptic_mesh(n_boundary=80)


@pytest.fixture
def sigma(mesh):
    rng = np.random.default_rng(42)
    return 0.5 + rng.random(mesh.n_triangles)  # non-uniform P0 coefficient


def test_stiffness_matrix_agreement(mesh, sigma):
    """Stiffness matrix K: skfem vs legacy agreement."""
    K_sk = skfem.assemble_stiffness_matrix(mesh, sigma)
    K_lg = legacy.assemble_stiffness_matrix(mesh, sigma)
    err = np.max(np.abs(K_sk.toarray() - K_lg.toarray()))
    assert err < 1e-12, f"Stiffness matrix discrepancy {err:.2e} exceeds threshold"


def test_stiffness_uniform_sigma_agreement(mesh):
    """Stiffness matrix with uniform sigma=1.0: agreement."""
    K_sk = skfem.assemble_stiffness_matrix(mesh, 1.0)
    K_lg = legacy.assemble_stiffness_matrix(mesh, 1.0)
    err = np.max(np.abs(K_sk.toarray() - K_lg.toarray()))
    assert err < 1e-13


def test_mass_matrix_agreement(mesh):
    """Mass matrix M (default coeff=1): agreement."""
    M_sk = skfem.assemble_mass_matrix(mesh)
    M_lg = legacy.assemble_mass_matrix(mesh)
    err = np.max(np.abs(M_sk.toarray() - M_lg.toarray()))
    assert err < 1e-14


def test_mass_matrix_variable_coeff_agreement(mesh, sigma):
    """Mass matrix M (non-uniform P0 coefficient): agreement."""
    M_sk = skfem.assemble_mass_matrix(mesh, sigma)
    M_lg = legacy.assemble_mass_matrix(mesh, sigma)
    err = np.max(np.abs(M_sk.toarray() - M_lg.toarray()))
    assert err < 1e-14


def test_boundary_mass_matrix_agreement(mesh):
    """Boundary mass matrix M_Gamma: agreement."""
    Mb_sk = skfem.assemble_boundary_mass_matrix(mesh)
    Mb_lg = legacy.assemble_boundary_mass_matrix(mesh)
    err = np.max(np.abs(Mb_sk.toarray() - Mb_lg.toarray()))
    assert err < 1e-14


def test_boundary_load_agreement(mesh):
    """Boundary load vector b = integral_Gamma f phi_i ds: agreement.

    Note: for nonlinear functions (e.g. x^2), skfem uses higher-order quadrature
    while legacy uses P1 trapezoidal rule, giving O(h^2) discrepancy.
    Linear functions (x, y) should agree exactly.
    """
    # Linear functions: exact agreement
    for f in [lambda x, y: x, lambda x, y: y]:
        b_sk = skfem.assemble_boundary_load(mesh, f)
        b_lg = legacy.assemble_boundary_load(mesh, f)
        err = np.max(np.abs(b_sk - b_lg))
        assert err < 1e-13, f"Linear boundary load discrepancy {err:.2e}"

    # Nonlinear functions: allow O(h^2) quadrature difference
    b_sk = skfem.assemble_boundary_load(mesh, lambda x, y: x**2 + y)
    b_lg = legacy.assemble_boundary_load(mesh, lambda x, y: x**2 + y)
    rel_err = np.max(np.abs(b_sk - b_lg)) / (np.max(np.abs(b_lg)) + 1e-30)
    assert rel_err < 0.01, f"Nonlinear boundary load rel. error {rel_err:.2e} exceeds 1%"


def test_boundary_mean_constraint_agreement(mesh):
    """Boundary mean constraint B: agreement."""
    B_sk = skfem.assemble_boundary_mean_constraint(mesh)
    B_lg = legacy.assemble_boundary_mean_constraint(mesh)
    err = np.max(np.abs(B_sk - B_lg))
    assert err < 1e-14


def test_neumann_solve_agreement(mesh):
    """Neumann saddle-point system solve: agreement."""
    K = skfem.assemble_stiffness_matrix(mesh, 1.0)
    b = skfem.assemble_boundary_load(mesh, lambda x, y: x)
    B = skfem.assemble_boundary_mean_constraint(mesh)
    y_sk = skfem.solve_neumann_system(K, b, B)

    K_lg = legacy.assemble_stiffness_matrix(mesh, 1.0)
    b_lg = legacy.assemble_boundary_load(mesh, lambda x, y: x)
    B_lg = legacy.assemble_boundary_mean_constraint(mesh)
    y_lg = legacy.solve_neumann_system(K_lg, b_lg, B_lg)

    err = np.max(np.abs(y_sk - y_lg))
    assert err < 1e-10, f"Neumann solution discrepancy {err:.2e}"


def test_robin_solve_agreement(mesh):
    """Robin BVP solve: agreement."""
    A_sk = skfem.assemble_stiffness_matrix(mesh, 1.0)
    A_lg = legacy.assemble_stiffness_matrix(mesh, 1.0)
    v = np.random.default_rng(42).standard_normal(mesh.n_points)

    z_sk = skfem.solve_robin_system(mesh, A_sk, 1.0, v)
    z_lg = legacy.solve_robin_system(mesh, A_lg, 1.0, v)

    err = np.max(np.abs(z_sk - z_lg))
    assert err < 1e-10, f"Robin solution discrepancy {err:.2e}"


def test_boundary_normal_flux_agreement(mesh):
    """Boundary normal flux sigma * dy/dn: agreement."""
    sigma = np.ones(mesh.n_triangles)
    K = skfem.assemble_stiffness_matrix(mesh, 1.0)
    b = skfem.assemble_boundary_load(mesh, lambda x, y: x)
    B = skfem.assemble_boundary_mean_constraint(mesh)
    y = skfem.solve_neumann_system(K, b, B)

    flux_sk = skfem.compute_boundary_normal_flux(mesh, sigma, y)
    flux_lg = legacy.compute_boundary_normal_flux(mesh, sigma, y)

    bdry = mesh.boundary_nodes
    err = np.max(np.abs(flux_sk[bdry] - flux_lg[bdry]))
    assert err < 1e-10, f"Normal flux discrepancy {err:.2e}"


def test_partial_boundary_mass_agreement(mesh):
    """Partial boundary mass matrices M_D, M_N: agreement."""
    mask = np.zeros(mesh.n_points, dtype=bool)
    mask[mesh.boundary_nodes[:len(mesh.boundary_nodes) // 2]] = True

    MD_sk, MN_sk = skfem.assemble_partial_boundary_mass_matrix(mesh, mask)
    MD_lg, MN_lg = legacy.assemble_partial_boundary_mass_matrix(mesh, mask)

    err_D = np.max(np.abs(MD_sk.toarray() - MD_lg.toarray()))
    err_N = np.max(np.abs(MN_sk.toarray() - MN_lg.toarray()))
    assert err_D < 1e-14, f"M_D discrepancy {err_D:.2e}"
    assert err_N < 1e-14, f"M_N discrepancy {err_N:.2e}"
