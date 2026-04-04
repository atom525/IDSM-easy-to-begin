"""Tests for idsm.py -- Iterative Direct Sampling Method."""

import numpy as np
import pytest

from IDSM.src.forward_solver import generate_cauchy_data, make_conductivity_example1
from IDSM.src.idsm import run_idsm
from IDSM.src.mesh import generate_elliptic_mesh


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
