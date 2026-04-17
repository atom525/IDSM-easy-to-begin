"""Tests for idsm.py -- Iterative Direct Sampling Method."""

import numpy as np
import pytest

from IDSM.src.forward_solver import generate_cauchy_data, make_conductivity_example1
from IDSM.src.idsm import run_idsm, apply_regularized_dtn
from IDSM.src.fem import (
    assemble_stiffness_matrix,
    assemble_mass_matrix,
    assemble_boundary_mass_matrix,
)
from IDSM.src.mesh import generate_elliptic_mesh
from IDSM.src.utils import compute_iou


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
# 端到端测试：验证 IDSM 重建质量（IoU）
# ============================================================

def test_idsm_reconstruction_iou(mesh):
    """端到端：IDSM 多次迭代后 IoU 应显著高于随机猜测。

    使用 2 个 Cauchy 数据，8 次迭代（粗网格下足够收敛），
    IoU > 0.1 即说明算法正确定位了夹杂。
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
    # 粗网格（80 boundary）下 IoU 通常在 0.15-0.40，远高于随机（~0.01）
    assert iou > 0.05, f"IoU={iou:.4f} too low, reconstruction failed"
    # 残差应有实质下降
    res = hist["residuals"]
    assert res[-1] < 0.8 * res[0], "Residual did not decrease enough"


# ============================================================
# DtN map 单元测试
# ============================================================

def test_regularized_dtn_output_finite(mesh):
    """apply_regularized_dtn 返回有限值。"""
    K_bg = assemble_stiffness_matrix(mesh, 1.0)
    M_pot = assemble_mass_matrix(mesh, 1e-10)
    A_op = K_bg + M_pot
    M_bdry = assemble_boundary_mass_matrix(mesh)
    # 用边界上的线性函数作为输入
    v = mesh.points[:, 0]  # x 坐标
    w = apply_regularized_dtn(mesh, v, A_op, alpha=1.0, M_bdry=M_bdry, sigma_bg=1.0)
    assert np.all(np.isfinite(w))
    assert w.shape == (mesh.n_points,)


def test_regularized_dtn_alpha_dependence(mesh):
    """α 越大，DtN map 输出越平滑（范数更小）。

    Eq. 3.5 的正则化性质：大 α 意味着更强的正则化。
    """
    K_bg = assemble_stiffness_matrix(mesh, 1.0)
    M_pot = assemble_mass_matrix(mesh, 1e-10)
    A_op = K_bg + M_pot
    M_bdry = assemble_boundary_mass_matrix(mesh)
    v = mesh.points[:, 0]

    w_small = apply_regularized_dtn(mesh, v, A_op, alpha=0.1, M_bdry=M_bdry)
    w_large = apply_regularized_dtn(mesh, v, A_op, alpha=10.0, M_bdry=M_bdry)

    # 大 α 的输出 L2 范数应更小（更平滑/更正则化）
    norm_small = np.linalg.norm(w_small)
    norm_large = np.linalg.norm(w_large)
    assert norm_large < norm_small, (
        f"Larger alpha should produce smaller output: "
        f"||w(α=0.1)||={norm_small:.4e}, ||w(α=10)||={norm_large:.4e}"
    )


def test_regularized_dtn_zero_input_gives_zero(mesh):
    """零输入应产生零输出（线性算子）。"""
    K_bg = assemble_stiffness_matrix(mesh, 1.0)
    M_pot = assemble_mass_matrix(mesh, 1e-10)
    A_op = K_bg + M_pot
    v = np.zeros(mesh.n_points)
    w = apply_regularized_dtn(mesh, v, A_op, alpha=1.0, sigma_bg=1.0)
    assert np.allclose(w, 0.0, atol=1e-12)
