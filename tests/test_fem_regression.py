"""回归测试：fem_skfem vs fem_legacy 数值一致性验证。

对每个 FEM 装配/求解函数，用相同输入调用两种后端，
断言结果在机器精度范围内一致。
"""

import numpy as np
import pytest

from IDSM.src.mesh import generate_elliptic_mesh
from IDSM.src import fem_skfem as skfem
from IDSM.src import fem_legacy as legacy


@pytest.fixture
def mesh():
    return generate_elliptic_mesh(n_boundary=80)


@pytest.fixture
def sigma(mesh):
    rng = np.random.default_rng(42)
    return 0.5 + rng.random(mesh.n_triangles)  # 非均匀 P0 系数


def test_stiffness_matrix_agreement(mesh, sigma):
    """刚度矩阵 K: skfem vs legacy 数值一致。"""
    K_sk = skfem.assemble_stiffness_matrix(mesh, sigma)
    K_lg = legacy.assemble_stiffness_matrix(mesh, sigma)
    err = np.max(np.abs(K_sk.toarray() - K_lg.toarray()))
    assert err < 1e-12, f"刚度矩阵差异 {err:.2e} 超出阈值"


def test_stiffness_uniform_sigma_agreement(mesh):
    """均匀 sigma=1.0 的刚度矩阵一致性。"""
    K_sk = skfem.assemble_stiffness_matrix(mesh, 1.0)
    K_lg = legacy.assemble_stiffness_matrix(mesh, 1.0)
    err = np.max(np.abs(K_sk.toarray() - K_lg.toarray()))
    assert err < 1e-13


def test_mass_matrix_agreement(mesh):
    """质量矩阵 M（默认 coeff=1）: 一致性。"""
    M_sk = skfem.assemble_mass_matrix(mesh)
    M_lg = legacy.assemble_mass_matrix(mesh)
    err = np.max(np.abs(M_sk.toarray() - M_lg.toarray()))
    assert err < 1e-14


def test_mass_matrix_variable_coeff_agreement(mesh, sigma):
    """质量矩阵 M（非均匀 P0 系数）: 一致性。"""
    M_sk = skfem.assemble_mass_matrix(mesh, sigma)
    M_lg = legacy.assemble_mass_matrix(mesh, sigma)
    err = np.max(np.abs(M_sk.toarray() - M_lg.toarray()))
    assert err < 1e-14


def test_boundary_mass_matrix_agreement(mesh):
    """边界质量矩阵 M_Γ: 一致性。"""
    Mb_sk = skfem.assemble_boundary_mass_matrix(mesh)
    Mb_lg = legacy.assemble_boundary_mass_matrix(mesh)
    err = np.max(np.abs(Mb_sk.toarray() - Mb_lg.toarray()))
    assert err < 1e-14


def test_boundary_load_agreement(mesh):
    """边界载荷向量 b = ∫_Γ f φ_i ds: 一致性。

    注意：对于非线性函数（如 x²），skfem 使用更高阶 quadrature
    而 legacy 使用 P1 梯形规则，两者在非线性函数上有 O(h²) 差异。
    线性函数（x, y）应精确一致。
    """
    # 线性函数：两种后端精确一致
    for f in [lambda x, y: x, lambda x, y: y]:
        b_sk = skfem.assemble_boundary_load(mesh, f)
        b_lg = legacy.assemble_boundary_load(mesh, f)
        err = np.max(np.abs(b_sk - b_lg))
        assert err < 1e-13, f"线性函数边界载荷差异 {err:.2e}"

    # 非线性函数：允许 O(h²) quadrature 差异
    b_sk = skfem.assemble_boundary_load(mesh, lambda x, y: x**2 + y)
    b_lg = legacy.assemble_boundary_load(mesh, lambda x, y: x**2 + y)
    rel_err = np.max(np.abs(b_sk - b_lg)) / (np.max(np.abs(b_lg)) + 1e-30)
    assert rel_err < 0.01, f"非线性函数边界载荷相对差异 {rel_err:.2e} 超出 1%"


def test_boundary_mean_constraint_agreement(mesh):
    """边界均值约束 B: 一致性。"""
    B_sk = skfem.assemble_boundary_mean_constraint(mesh)
    B_lg = legacy.assemble_boundary_mean_constraint(mesh)
    err = np.max(np.abs(B_sk - B_lg))
    assert err < 1e-14


def test_neumann_solve_agreement(mesh):
    """Neumann 鞍点系统求解: 一致性。"""
    K = skfem.assemble_stiffness_matrix(mesh, 1.0)
    b = skfem.assemble_boundary_load(mesh, lambda x, y: x)
    B = skfem.assemble_boundary_mean_constraint(mesh)
    y_sk = skfem.solve_neumann_system(K, b, B)

    K_lg = legacy.assemble_stiffness_matrix(mesh, 1.0)
    b_lg = legacy.assemble_boundary_load(mesh, lambda x, y: x)
    B_lg = legacy.assemble_boundary_mean_constraint(mesh)
    y_lg = legacy.solve_neumann_system(K_lg, b_lg, B_lg)

    err = np.max(np.abs(y_sk - y_lg))
    assert err < 1e-10, f"Neumann 解差异 {err:.2e}"


def test_robin_solve_agreement(mesh):
    """Robin BVP 求解: 一致性。"""
    A_sk = skfem.assemble_stiffness_matrix(mesh, 1.0)
    A_lg = legacy.assemble_stiffness_matrix(mesh, 1.0)
    v = np.random.default_rng(42).standard_normal(mesh.n_points)

    z_sk = skfem.solve_robin_system(mesh, A_sk, 1.0, v)
    z_lg = legacy.solve_robin_system(mesh, A_lg, 1.0, v)

    err = np.max(np.abs(z_sk - z_lg))
    assert err < 1e-10, f"Robin 解差异 {err:.2e}"


def test_boundary_normal_flux_agreement(mesh):
    """边界法向通量 σ∂y/∂n: 一致性。"""
    sigma = np.ones(mesh.n_triangles)
    K = skfem.assemble_stiffness_matrix(mesh, 1.0)
    b = skfem.assemble_boundary_load(mesh, lambda x, y: x)
    B = skfem.assemble_boundary_mean_constraint(mesh)
    y = skfem.solve_neumann_system(K, b, B)

    flux_sk = skfem.compute_boundary_normal_flux(mesh, sigma, y)
    flux_lg = legacy.compute_boundary_normal_flux(mesh, sigma, y)

    bdry = mesh.boundary_nodes
    err = np.max(np.abs(flux_sk[bdry] - flux_lg[bdry]))
    assert err < 1e-10, f"法向通量差异 {err:.2e}"


def test_partial_boundary_mass_agreement(mesh):
    """部分边界质量矩阵 M_D, M_N: 一致性。"""
    mask = np.zeros(mesh.n_points, dtype=bool)
    mask[mesh.boundary_nodes[:len(mesh.boundary_nodes) // 2]] = True

    MD_sk, MN_sk = skfem.assemble_partial_boundary_mass_matrix(mesh, mask)
    MD_lg, MN_lg = legacy.assemble_partial_boundary_mass_matrix(mesh, mask)

    err_D = np.max(np.abs(MD_sk.toarray() - MD_lg.toarray()))
    err_N = np.max(np.abs(MN_sk.toarray() - MN_lg.toarray()))
    assert err_D < 1e-14, f"M_D 差异 {err_D:.2e}"
    assert err_N < 1e-14, f"M_N 差异 {err_N:.2e}"
