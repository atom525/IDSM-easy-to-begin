"""Tests for forward_solver.py — EIT forward problem and data generation."""

import numpy as np
import pytest

from IDSM.src.forward_solver import (
    generate_cauchy_data,
    make_conductivity_example1,
    solve_forward,
)
from IDSM.src.mesh import generate_elliptic_mesh


@pytest.fixture
def mesh():
    return generate_elliptic_mesh(n_boundary=80)


def test_background_solution_is_finite(mesh):
    y = solve_forward(mesh, 1.0, lambda x, y: x)
    assert y.shape[0] == mesh.n_points
    assert np.all(np.isfinite(y))


def test_background_solution_satisfies_gauge(mesh):
    """Solution with sigma=1 should satisfy ∫_Γ y ds = 0."""
    from IDSM.src.fem import assemble_boundary_mean_constraint
    y = solve_forward(mesh, 1.0, lambda x, y: x)
    B = assemble_boundary_mean_constraint(mesh)
    assert abs(np.dot(B, y)) < 1e-8


def test_inclusion_changes_boundary_data(mesh):
    """Including inclusions should change the boundary response."""
    sigma_true, _ = make_conductivity_example1(mesh)
    sigma_bg = np.ones(mesh.n_triangles)
    f = lambda x, y: x
    y_true = solve_forward(mesh, sigma_true, f)
    y_bg = solve_forward(mesh, sigma_bg, f)
    bdry = mesh.boundary_nodes
    diff = np.linalg.norm(y_true[bdry] - y_bg[bdry])
    assert diff > 0.01, "Inclusions should produce detectable scattering"


def test_noise_model_zero_noise_gives_exact_data(mesh):
    sigma_true, _ = make_conductivity_example1(mesh)
    data = generate_cauchy_data(mesh, sigma_true, [lambda x, y: x], noise_level=0.0)
    assert np.allclose(data['y_data'][0], data['y_omega'][0])


def test_noise_model_positive_noise_changes_data(mesh):
    sigma_true, _ = make_conductivity_example1(mesh)
    data = generate_cauchy_data(
        mesh, sigma_true, [lambda x, y: x],
        noise_level=0.2, rng=np.random.default_rng(42),
    )
    diff = np.linalg.norm(data['y_data'][0] - data['y_omega'][0])
    assert diff > 0.0


def test_noise_model_bounded_by_epsilon(mesh):
    """Noise should be bounded: |y_d - y*| <= eps * |y_empty - y*|."""
    sigma_true, _ = make_conductivity_example1(mesh)
    eps = 0.3
    data = generate_cauchy_data(
        mesh, sigma_true, [lambda x, y: x],
        noise_level=eps, rng=np.random.default_rng(42),
    )
    y_data = data['y_data'][0]
    y_omega = data['y_omega'][0]
    y_empty = data['y_empty'][0]
    bdry = mesh.boundary_nodes
    noise = np.abs(y_data[bdry] - y_omega[bdry])
    signal = np.abs(y_empty[bdry] - y_omega[bdry])
    max_ratio = np.max(noise / (signal + 1e-30))
    assert max_ratio <= eps + 1e-10


def test_conductivity_example1_has_two_inclusions(mesh):
    sigma, u = make_conductivity_example1(mesh)
    assert np.sum(u != 0) > 0
    assert np.min(sigma) < 1.0
    assert np.max(sigma) <= 1.0 + 1e-10
