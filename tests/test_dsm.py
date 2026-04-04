"""Tests for dsm.py -- classical Direct Sampling Method."""

import numpy as np
import pytest

from IDSM.src.dsm import (
    compute_dsm_indicator,
    discretize_laplace_beltrami,
    compute_scattering_data,
)
from IDSM.src.forward_solver import generate_cauchy_data, make_conductivity_example1
from IDSM.src.mesh import generate_elliptic_mesh


@pytest.fixture
def mesh():
    return generate_elliptic_mesh(n_boundary=80)


def test_lb_eigenvalues_nonneg_and_sorted(mesh):
    op = discretize_laplace_beltrami(mesh, gamma=0.5, n_eigenvalues=30)
    lam = op.eigenvalues
    assert np.all(lam >= -1e-10)
    assert np.all(np.diff(lam) >= -1e-10)


def test_lb_first_eigenvalue_near_zero(mesh):
    op = discretize_laplace_beltrami(mesh, gamma=0.5, n_eigenvalues=10)
    assert abs(op.eigenvalues[0]) < 1e-8


def test_lb_eigenvalues_come_in_pairs(mesh):
    op = discretize_laplace_beltrami(mesh, gamma=0.5, n_eigenvalues=20)
    lam = op.eigenvalues
    for k in range(1, min(5, len(lam) // 2)):
        ratio = lam[2 * k] / max(lam[2 * k - 1], 1e-30)
        assert abs(ratio - 1.0) < 0.1, f"Pair {k}: ratio={ratio}"


def test_dsm_indicator_nonnegative(mesh):
    sigma_true, _ = make_conductivity_example1(mesh)
    data = generate_cauchy_data(mesh, sigma_true, [lambda x, y: x], noise_level=0.0)
    result = compute_dsm_indicator(mesh, data, n_grid=61, n_eigenvalues=20)
    vals = result["indicator"][result["mask"]]
    assert np.all(vals >= -1e-12)


def test_dsm_indicator_larger_at_inclusion(mesh):
    sigma_true, _ = make_conductivity_example1(mesh)
    data = generate_cauchy_data(
        mesh, sigma_true, [lambda x, y: x, lambda x, y: y], noise_level=0.0
    )
    result = compute_dsm_indicator(mesh, data, n_grid=101, n_eigenvalues=40)
    indicator = result["indicator"]
    gx = result["grid_x"]
    gy = result["grid_y"]

    ix_inc = np.argmin(np.abs(gx - 0.4))
    iy_inc = np.argmin(np.abs(gy - 0.2))
    ix_far = np.argmin(np.abs(gx - 0.0))
    iy_far = np.argmin(np.abs(gy - 0.0))

    val_inc = indicator[ix_inc, iy_inc]
    val_far = indicator[ix_far, iy_far]
    assert val_inc > val_far


def test_scattering_data_structure():
    mesh = generate_elliptic_mesh(n_boundary=64)
    sigma_true, _ = make_conductivity_example1(mesh)
    data = generate_cauchy_data(mesh, sigma_true, [lambda x, y: x], noise_level=0.0)
    scatter = compute_scattering_data(data)
    assert len(scatter) == 1
    assert scatter[0].shape[0] == mesh.n_points
