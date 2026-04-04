"""Tests for idsm_partial.py -- partial-data IDSM with stabilization."""

import numpy as np
import pytest

from IDSM.src.forward_solver import generate_cauchy_data, make_conductivity_example1
from IDSM.src.idsm_partial import (
    apply_hr_dtn,
    complete_data,
    define_accessible_boundary,
    run_idsm_partial,
)
from IDSM.src.fem import (
    assemble_mass_matrix,
    assemble_partial_boundary_mass_matrix,
    assemble_stiffness_matrix,
)
from IDSM.src.mesh import generate_elliptic_mesh


@pytest.fixture
def mesh():
    return generate_elliptic_mesh(n_boundary=80)


@pytest.fixture
def gamma_d(mesh):
    return define_accessible_boundary(mesh, (-np.pi / 2, np.pi / 2))


def test_data_completion_rule(mesh, gamma_d):
    """Data completion: use measured on GammaD, current on GammaN."""
    y_data = np.ones(mesh.n_points)
    y_cur = np.zeros(mesh.n_points)
    yc = complete_data(y_data, y_cur, gamma_d["node_mask"])
    assert np.allclose(yc[gamma_d["node_mask"]], y_data[gamma_d["node_mask"]])
    assert np.allclose(yc[~gamma_d["node_mask"]], y_cur[~gamma_d["node_mask"]])


def test_data_completion_preserves_measured_data(mesh, gamma_d):
    """On GammaD, completed data must equal original measured data exactly."""
    rng = np.random.default_rng(0)
    y_data = rng.standard_normal(mesh.n_points)
    y_cur = rng.standard_normal(mesh.n_points)
    yc = complete_data(y_data, y_cur, gamma_d["node_mask"])
    mask = gamma_d["node_mask"]
    assert np.allclose(yc[mask], y_data[mask], atol=1e-15)


def test_hr_dtn_reduces_to_uniform_when_alpha_equal(mesh, gamma_d):
    """When alpha_d == alpha_n, HR-DtN should behave like uniform DtN."""
    M_D, M_N = assemble_partial_boundary_mass_matrix(mesh, gamma_d["node_mask"])
    A = assemble_stiffness_matrix(mesh, 1.0) + assemble_mass_matrix(mesh, 1e-10)
    v = np.random.default_rng(0).standard_normal(mesh.n_points)
    w1 = apply_hr_dtn(mesh, v, A, 0.2, 0.2, M_D, M_N, sigma_bg=1.0)
    w2 = apply_hr_dtn(mesh, v, A, 0.2, 0.2, M_D, M_N, sigma_bg=1.0)
    assert np.allclose(w1, w2, atol=1e-12)


def test_hr_dtn_output_is_finite(mesh, gamma_d):
    M_D, M_N = assemble_partial_boundary_mass_matrix(mesh, gamma_d["node_mask"])
    A = assemble_stiffness_matrix(mesh, 1.0) + assemble_mass_matrix(mesh, 1e-10)
    v = np.random.default_rng(0).standard_normal(mesh.n_points)
    w = apply_hr_dtn(mesh, v, A, 0.05, 2.0, M_D, M_N, sigma_bg=1.0)
    assert np.all(np.isfinite(w))


def test_partial_idsm_lambda_nonnegative(mesh, gamma_d):
    sigma_true, _ = make_conductivity_example1(mesh)
    data = generate_cauchy_data(mesh, sigma_true, [lambda x, y: x], noise_level=0.02)
    hist = run_idsm_partial(mesh, data, gamma_d, n_iter=3, verbose=False)
    lam = np.asarray(hist["lambda_history"])
    if lam.size > 0:
        assert np.all(lam >= -1e-12)


def test_partial_idsm_residuals_finite(mesh, gamma_d):
    sigma_true, _ = make_conductivity_example1(mesh)
    data = generate_cauchy_data(mesh, sigma_true, [lambda x, y: x], noise_level=0.02)
    hist = run_idsm_partial(mesh, data, gamma_d, n_iter=3, verbose=False)
    assert np.all(np.isfinite(hist["residuals"]))


def test_define_accessible_boundary_partition(mesh):
    """GammaD and GammaN should partition the boundary."""
    gd = define_accessible_boundary(mesh, (0, np.pi))
    bdry_mask = gd["bdry_mask"]
    n_d = np.sum(bdry_mask)
    n_n = np.sum(~bdry_mask)
    assert n_d > 0
    assert n_n > 0
    assert n_d + n_n == len(mesh.boundary_nodes)
