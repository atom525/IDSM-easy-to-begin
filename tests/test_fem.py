"""Tests for fem.py — P1 finite element assembly and solvers."""

import numpy as np
import pytest

from IDSM.src.fem import (
    assemble_boundary_load,
    assemble_boundary_mass_matrix,
    assemble_boundary_mean_constraint,
    assemble_mass_matrix,
    assemble_stiffness_matrix,
    solve_neumann_system,
)
from IDSM.src.mesh import generate_elliptic_mesh


@pytest.fixture
def mesh():
    return generate_elliptic_mesh(n_boundary=80)


def test_stiffness_matrix_symmetry(mesh):
    K = assemble_stiffness_matrix(mesh, 1.0)
    diff = (K - K.T).tocoo()
    if diff.nnz > 0:
        assert np.max(np.abs(diff.data)) < 1e-12


def test_stiffness_null_space_is_constants(mesh):
    """K*1 should be approximately zero (constants in null space)."""
    K = assemble_stiffness_matrix(mesh, 1.0)
    ones = np.ones(mesh.n_points)
    assert np.linalg.norm(K.dot(ones)) < 1e-10


def test_mass_matrix_positive_diagonal(mesh):
    M = assemble_mass_matrix(mesh, 1.0)
    assert np.all(M.diagonal() > 0)


def test_mass_matrix_row_sums_approximate_area(mesh):
    """Sum of all mass matrix entries should equal domain area."""
    M = assemble_mass_matrix(mesh, 1.0)
    total = float(M.sum())
    target = float(np.sum(mesh.areas))
    assert abs(total - target) / target < 0.01


def test_boundary_mass_positive(mesh):
    M_bdry = assemble_boundary_mass_matrix(mesh)
    diag = M_bdry.diagonal()
    bdry_set = set(int(n) for n in mesh.boundary_nodes)
    for i in bdry_set:
        assert diag[i] > 0


def test_neumann_compatibility_f1(mesh):
    """f1 = x1 satisfies compatibility by symmetry of ellipse."""
    b = assemble_boundary_load(mesh, lambda x, y: x)
    assert abs(np.sum(b)) < 1e-12


def test_neumann_compatibility_f2(mesh):
    """f2 = x2 satisfies compatibility by symmetry of ellipse."""
    b = assemble_boundary_load(mesh, lambda x, y: y)
    assert abs(np.sum(b)) < 1e-12


def test_neumann_solution_finite_and_satisfies_constraint(mesh):
    K = assemble_stiffness_matrix(mesh, 1.0)
    b = assemble_boundary_load(mesh, lambda x, y: x)
    B = assemble_boundary_mean_constraint(mesh)
    y = solve_neumann_system(K, b, B)
    assert np.all(np.isfinite(y))
    assert abs(np.dot(B, y)) < 1e-8


def test_stiffness_with_variable_sigma(mesh):
    """Variable sigma should still produce symmetric K."""
    sigma = np.random.default_rng(42).uniform(0.5, 2.0, mesh.n_triangles)
    K = assemble_stiffness_matrix(mesh, sigma)
    diff = (K - K.T).tocoo()
    if diff.nnz > 0:
        assert np.max(np.abs(diff.data)) < 1e-12
