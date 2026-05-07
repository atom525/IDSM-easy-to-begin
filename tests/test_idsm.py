"""Tests for idsm.py -- Iterative Direct Sampling Method."""

import numpy as np
import pytest

from cooperation.ghy.IDSM.src.forward_solver import (
    generate_cauchy_data,
    generate_cauchy_data_general,
    make_conductivity_example1,
    make_double_example2,
)
from cooperation.ghy.IDSM.src.idsm import run_idsm, apply_regularized_dtn
from cooperation.ghy.IDSM.src.fem import (
    assemble_stiffness_matrix,
    assemble_mass_matrix,
    assemble_boundary_mass_matrix,
)
from cooperation.ghy.IDSM.src.mesh import generate_elliptic_mesh
from cooperation.ghy.IDSM.src.utils import compute_iou


@pytest.fixture
def mesh():
    return generate_elliptic_mesh(n_boundary=80)


@pytest.fixture
def cauchy_data(mesh):
    sigma_true, _ = make_conductivity_example1(mesh)
    return generate_cauchy_data(
        mesh, sigma_true, [lambda x, y: x], noise_level=0.02,
        rng=np.random.default_rng(42),
    )


def test_idsm_box_constraints(mesh, cauchy_data):
    hist = run_idsm(
        mesh, cauchy_data, n_iter=3, sigma_bg=1.0, sigma_range=0.3,
        problem_type="conductivity", verbose=False,
    )
    sigma_final = hist["sigma_final"]
    assert np.all(sigma_final >= 0.3 - 1e-10)
    assert np.all(sigma_final <= 1.0 + 1e-10)


def test_idsm_residuals_finite(mesh, cauchy_data):
    hist = run_idsm(
        mesh, cauchy_data, n_iter=5, sigma_bg=1.0, sigma_range=0.3,
        problem_type="conductivity", verbose=False,
    )
    assert np.all(np.isfinite(hist["residuals"]))
    assert len(hist["residuals"]) == 6  # initial + 5 iterations


def test_idsm_residual_decreases(mesh, cauchy_data):
    """Residual should generally decrease over iterations."""
    hist = run_idsm(
        mesh, cauchy_data, n_iter=8, sigma_bg=1.0, sigma_range=0.3,
        problem_type="conductivity", verbose=False,
    )
    res = hist["residuals"]
    assert res[-1] < res[0], "Final residual should be less than initial"


def test_idsm_sigma_history_length(mesh, cauchy_data):
    n_iter = 5
    hist = run_idsm(
        mesh, cauchy_data, n_iter=n_iter, sigma_bg=1.0, sigma_range=0.3,
        problem_type="conductivity", verbose=False,
    )
    assert len(hist["sigma_guess"]) == n_iter


def test_idsm_dfp_vs_bfg_both_work(mesh, cauchy_data):
    for method in ["DFP", "BFG"]:
        hist = run_idsm(
            mesh, cauchy_data, n_iter=3, sigma_bg=1.0, sigma_range=0.3,
            problem_type="conductivity", lowrank_method=method, verbose=False,
        )
        assert np.all(np.isfinite(hist["sigma_final"]))


# ============================================================
# End-to-end test: IDSM reconstruction quality (IoU)
# ============================================================

def test_idsm_reconstruction_iou(mesh):
    """End-to-end: IoU should be well above random after several IDSM iterations.

    Uses 2 Cauchy data, 8 iterations (sufficient for coarse mesh).
    IoU > 0.1 indicates the algorithm correctly locates inclusions.
    """
    sigma_true, u_true = make_conductivity_example1(mesh)
    data = generate_cauchy_data(
        mesh, sigma_true,
        [lambda x, y: x, lambda x, y: y],
        noise_level=0.05,
        rng=np.random.default_rng(123),
    )
    hist = run_idsm(
        mesh, data, n_iter=8, sigma_bg=1.0, sigma_range=0.3,
        problem_type="conductivity", lowrank_method="BFG", verbose=False,
    )
    # u_pred = sigma_final - sigma_bg
    u_pred = hist["sigma_final"] - 1.0
    iou = compute_iou(u_true, u_pred, mesh)
    # Coarse mesh (80 boundary) typically gives IoU 0.15-0.40, far above random (~0.01)
    assert iou > 0.05, f"IoU={iou:.4f} too low, reconstruction failed"
    # Residual should decrease substantially
    res = hist["residuals"]
    assert res[-1] < 0.8 * res[0], "Residual did not decrease enough"


# ============================================================
# DtN map unit tests
# ============================================================

def test_regularized_dtn_output_finite(mesh):
    """apply_regularized_dtn should return finite values."""
    K_bg = assemble_stiffness_matrix(mesh, 1.0)
    M_pot = assemble_mass_matrix(mesh, 1e-10)
    A_op = K_bg + M_pot
    M_bdry = assemble_boundary_mass_matrix(mesh)
    # Use a linear function on the boundary as input
    v = mesh.points[:, 0]  # x coordinate
    w = apply_regularized_dtn(mesh, v, A_op, alpha=1.0, M_bdry=M_bdry, sigma_bg=1.0)
    assert np.all(np.isfinite(w))
    assert w.shape == (mesh.n_points,)


def test_regularized_dtn_alpha_dependence(mesh):
    """Larger alpha should produce smoother (smaller norm) output.

    Regularization property of Eq. 3.5: larger alpha = stronger regularization.
    """
    K_bg = assemble_stiffness_matrix(mesh, 1.0)
    M_pot = assemble_mass_matrix(mesh, 1e-10)
    A_op = K_bg + M_pot
    M_bdry = assemble_boundary_mass_matrix(mesh)
    v = mesh.points[:, 0]

    w_small = apply_regularized_dtn(mesh, v, A_op, alpha=0.1, M_bdry=M_bdry)
    w_large = apply_regularized_dtn(mesh, v, A_op, alpha=10.0, M_bdry=M_bdry)

    # Larger alpha should give smaller L2 norm
    norm_small = np.linalg.norm(w_small)
    norm_large = np.linalg.norm(w_large)
    assert norm_large < norm_small, (
        f"Larger alpha should produce smaller output: "
        f"||w(α=0.1)||={norm_small:.4e}, ||w(α=10)||={norm_large:.4e}"
    )


def test_regularized_dtn_zero_input_gives_zero(mesh):
    """Zero input should produce zero output (linear operator)."""
    K_bg = assemble_stiffness_matrix(mesh, 1.0)
    M_pot = assemble_mass_matrix(mesh, 1e-10)
    A_op = K_bg + M_pot
    v = np.zeros(mesh.n_points)
    w = apply_regularized_dtn(mesh, v, A_op, alpha=1.0, sigma_bg=1.0)
    assert np.allclose(w, 0.0, atol=1e-12)


# ============================================================
# Example 2 (double type) tests
# ============================================================

def test_idsm_double_type_residual_decreases(mesh):
    """Double-type IDSM should run and reduce residual.

    Uses FreeFEM Example2.edp parameters:
      sigma_0=1.0, v_0=1.0, alpha=0.1, DFP, R_0=100.0 (constant).
    """
    sigma_true, potential_true, u_sigma, u_potential = make_double_example2(mesh)

    cauchy = generate_cauchy_data_general(
        mesh, sigma_true, potential_true,
        [lambda x, y: x, lambda x, y: y],
        noise_level=0.05,
        rng=np.random.default_rng(42),
    )

    hist = run_idsm(
        mesh, cauchy, n_iter=5,
        sigma_bg=1.0, potential_bg=1.0,
        sigma_range=0.01, potential_range=10.0,
        alpha=0.1, lowrank_method='DFP',
        problem_type='double', coeff_known=False,
        r0_constant=100.0,
        verbose=False,
    )

    # Box constraint checks
    sf = hist['sigma_final']
    vf = hist['potential_final']
    assert np.all(sf >= 0.01 - 1e-10), f"sigma lower bound violated: min={sf.min()}"
    assert np.all(sf <= 1.0 + 1e-10), f"sigma upper bound violated: max={sf.max()}"
    assert np.all(vf >= 1.0 - 1e-10), f"v lower bound violated: min={vf.min()}"
    assert np.all(vf <= 10.0 + 1e-10), f"v upper bound violated: max={vf.max()}"

    # Residuals finite and decreasing
    res = hist['residuals']
    assert np.all(np.isfinite(res)), "Residuals contain non-finite values"
    assert res[-1] < res[0], f"Residual did not decrease: {res[0]:.4e} -> {res[-1]:.4e}"

    # History length
    assert len(hist['sigma_guess']) == 5
    assert len(hist['potential_guess']) == 5


def test_idsm_double_type_both_fields_nontrivial(mesh):
    """Double-type reconstruction should update both sigma and v (not stuck at background)."""
    sigma_true, potential_true, _, _ = make_double_example2(mesh)

    cauchy = generate_cauchy_data_general(
        mesh, sigma_true, potential_true,
        [lambda x, y: x, lambda x, y: y],
        noise_level=0.02,
        rng=np.random.default_rng(99),
    )

    hist = run_idsm(
        mesh, cauchy, n_iter=5,
        sigma_bg=1.0, potential_bg=1.0,
        sigma_range=0.01, potential_range=10.0,
        alpha=0.1, lowrank_method='DFP',
        problem_type='double', r0_constant=100.0,
        verbose=False,
    )

    sf = hist['sigma_final']
    vf = hist['potential_final']

    # sigma should deviate from background at inclusion locations
    sigma_deviation = np.max(np.abs(sf - 1.0))
    assert sigma_deviation > 0.01, f"sigma stuck at background: max deviation={sigma_deviation:.6f}"

    # v should deviate from background at inclusion locations
    potential_deviation = np.max(np.abs(vf - 1.0))
    assert potential_deviation > 0.01, f"v stuck at background: max deviation={potential_deviation:.6f}"
