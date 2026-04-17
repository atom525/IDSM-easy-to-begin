"""
forward_solver.py - EIT 正问题求解器

严格按照 Ito et al. (2025) Section 4 实现：
  PDE: ∇·(σ∇y) = 0 in Ω,  σ ∂y/∂n = f on Γ
  约束: ∫_Γ y ds = 0

求解域: 椭圆 Ω = {x₁² + x₂²/0.64 < 1}
背景电导率: σ₀ = 1
夹杂体: u = σ − σ₀

噪声模型 (Paper 1 Section 4, FreeFEM Example1.edp L235-238):
  yd(x) = y*(x) + ε·δ(x)·|y_∅(x) − y*(x)|
  δ(x) ~ Uniform(−1, 1), ε = 相对噪声水平
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
# 夹杂体定义函数
# ============================================================

def square_inclusion(x, y, center, half_width):
    """方形夹杂体的特征函数。

    参考 FreeFEM Example1.edp L22-23:
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
    """圆形夹杂体的特征函数。"""
    cx, cy = center
    return (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2


def make_conductivity_example1(mesh):
    """创建 Example 1 (EIT) 的真实电导率。

    Paper 1 Section 4.1:
      σ₀ = 1（背景），σ = 0.3（夹杂体内，即 u = −0.7）
      两个方形夹杂体:
        - 中心 (0.4, 0.2), 半宽 0.2
        - 中心 (−0.5, −0.2), 半宽 0.2

    参考: FreeFEM Example1.edp L22-23.
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
    """创建导电型夹杂体示例 (σ > σ₀)。

    与 Example 1 相同几何形状（两个方形），但 σ = 3.0（导电型），
    即 u = σ − σ₀ = +2.0。
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
    """创建单圆形夹杂体示例（绝缘型）。

    单个圆形夹杂体，中心 (0.3, 0.0)，半径 0.25，σ = 0.3。
    用于单夹杂 vs 多夹杂对比实验。
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
# 正问题求解器
# ============================================================

def solve_forward(mesh, sigma, f_func):
    """求解 EIT 正问题: ∇·(σ∇y) = 0 in Ω,  σ ∂y/∂n = f on Γ,  ∫_Γ y ds = 0。"""
    K = assemble_stiffness_matrix(mesh, sigma)
    b = assemble_boundary_load(mesh, f_func)
    B = assemble_boundary_mean_constraint(mesh)

    y = solve_neumann_system(K, b, B)
    return y


def solve_forward_general(mesh, sigma, potential_coeff, f_func, is_boundary_source=True):
    """求解含零阶项的广义椭圆正问题: −∇·(σ∇y) + u_p·y = f。

    弱形式: ∫_Ω σ∇y·∇v dx + ∫_Ω u_p·y·v dx = ∫_Γ f·v ds
    用于 DOT (Example 3) 等含 potential 项的问题。
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
    """组装域源载荷向量 b_i = ∫_Ω f(x) φ_i dx。

    使用重心积分（1 点 Gauss）: ∫_{T_e} f φ_i dx ≈ |T_e|/3 * f(重心)
    """
    n = mesh.n_points
    b = np.zeros(n)

    cx, cy = mesh.centroids[:, 0], mesh.centroids[:, 1]
    f_vals = f_func(cx, cy)

    for i in range(3):
        np.add.at(b, mesh.triangles[:, i], mesh.areas * f_vals / 3.0)

    return b


# ============================================================
# Cauchy 数据生成
# ============================================================

def generate_cauchy_data(mesh, sigma_true, source_funcs, noise_level=0.0, rng=None):
    """生成带噪声的 Cauchy 数据对。

    对每个源 f_ℓ:
      1. 求解含夹杂体的正问题: y_Ω = solve(σ_true, f_ℓ)
      2. 求解背景正问题: y_∅ = solve(σ₀=1, f_ℓ)
      3. 添加噪声 (Paper 1 Section 4, FreeFEM Example1.edp L235-238):
         yd(x) = y_Ω(x) + ε·δ(x)·|y_Ω(x) − y_∅(x)|
         δ(x) ~ Uniform(−1, 1)
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
# 附加示例几何（Phase 4）
# ============================================================

def make_potential_example3(mesh):
    """创建 Example 3 (仅 potential 型, DOT) 的真实夹杂体。

    FreeFEM Example3.edp:
      type = "potential", vA = 1e-10, vB = 10.0, vU = 6（未知模式）
      σ₀ = 1（常数），无电导率夹杂体
      Potential 夹杂体 v:
        - 中心 (−0.6, 0.1), 半宽 0.15
        - 中心 (0.5, −0.1), 半宽 0.2
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
    """为含零阶项的广义模型 (DOT) 生成 Cauchy 数据。

    −∇·(σ∇y) + v·y = 0 in Ω,  σ ∂y/∂n = f on Γ

    与 generate_cauchy_data 结构相同，但使用 solve_forward_general
    求解含 potential 项的 PDE。
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
