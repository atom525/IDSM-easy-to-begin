"""Stable partial-data IDSM (Paper 3 oriented implementation).

This module implements the key components described in:
Jin, Wang, Zou (2026), "A stable iterative direct sampling method for elliptic
inverse problems with partial Cauchy data", notably Eq. (4.1)-(4.16) and
Algorithm 5.1 in a discrete FEM setting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.sparse.linalg import spsolve
import warnings

from .fem import (
    assemble_mass_matrix,
    assemble_partial_boundary_mass_matrix,
    assemble_stiffness_matrix,
)
from .forward_solver import solve_forward, solve_forward_general
from .idsm import compute_ellipse_normal_derivative, compute_p0_gradient
from .mesh import coarse_to_fine_p0, fine_to_coarse_p0, generate_coarse_mesh
from .utils import distance_to_boundary
from .config import RuntimeConfig


def define_accessible_boundary(mesh, theta_range, a=1.0, b=0.8):
    """定义椭圆边界上的可观测部分 Γ_D。

    给定椭圆参数角范围 [θ_start, θ_end]，标记边界节点中落在
    该角度范围内的节点为可观测节点。支持跨越 ±π 的角度范围。

    Paper 3 Section 2: Γ = Γ_D ∪ Γ_N，Γ_D 为 Cauchy 数据可用的部分。

    Parameters
    ----------
    mesh : EllipticMesh
    theta_range : tuple (θ_start, θ_end) — 椭圆参数角范围（弧度）
    a, b : float — 椭圆半轴（默认 a=1.0, b=0.8）

    Returns
    -------
    dict: 'node_mask' (全域 bool), 'bdry_mask' (边界 bool)
    """
    theta_start, theta_end = theta_range
    node_mask = np.zeros(mesh.n_points, dtype=bool)
    for idx in mesh.boundary_nodes:
        x, y = mesh.points[idx]
        theta = np.arctan2(y / b, x / a)
        if theta_start <= theta_end:
            in_range = theta_start <= theta <= theta_end
        else:
            in_range = (theta >= theta_start) or (theta <= theta_end)
        node_mask[idx] = in_range
    return {"node_mask": node_mask, "bdry_mask": node_mask[mesh.boundary_nodes]}


def complete_data(y_data, y_current, gamma_d_node_mask):
    """Paper 3 Eq. (4.1): y_tilde_d = T_D y* + T_N y(u_k)."""
    y_complete = y_current.copy()
    y_complete[gamma_d_node_mask] = y_data[gamma_d_node_mask]
    return y_complete


def apply_hr_dtn(
    mesh,
    v,
    A_op,
    alpha_d,
    alpha_n,
    M_bdry_D,
    M_bdry_N,
    sigma_bg=1.0,
    a=1.0,
    b=0.8,
):
    """Apply heterogeneous regularized DtN map via two HR-Robin solves."""
    system_matrix = A_op + (1.0 / alpha_d) * M_bdry_D + (1.0 / alpha_n) * M_bdry_N
    M_rhs = (1.0 / alpha_d) * M_bdry_D + (1.0 / alpha_n) * M_bdry_N

    rhs1 = M_rhs.dot(v)
    z = spsolve(system_matrix, rhs1)

    g = compute_ellipse_normal_derivative(mesh, z, sigma_bg=sigma_bg, a=a, b=b)

    rhs2 = M_rhs.dot(g)
    w = spsolve(system_matrix, rhs2)
    return w


def _l_p_norm(values, areas, p):
    values = np.asarray(values, dtype=np.float64)
    if p == np.inf:
        return float(np.max(np.abs(values)))
    p = float(p)
    return float(np.sum(areas * np.abs(values) ** p) ** (1.0 / p))


def _duality(zeta, eta, areas):
    return float(np.sum(areas * np.asarray(zeta) * np.asarray(eta)))


def compute_heterogeneous_D(mesh, gamma_d_node_mask, alpha_d, alpha_n, gamma=4.0, epsilon=0.02):
    """Compute heterogeneous R₀ diagonal，与全数据 initialize_r0_diagonal 结构一致。

    Paper 3 Eq. (4.5) 定义了 heterogeneous Sobolev 范数 ‖Φ_z‖_{H¹_{α_D}(Γ)}。
    其离散实现复用全数据 FreeFEM 的 diagFunc 公式（∫_Γ 1/r² ds'），
    但按 α/(1+α) 在 Γ_D/Γ_N 上分区加权。

    公式：D(z) = 1 / (Σ_e w_e · L_e/2 · (1/r0² + 1/r1²))^exponent
    其中 w_e = 1/(1+α_e)，α_e = α_d（Γ_D 边）或 α_n（Γ_N 边）。
    exponent = 0.5（匹配 FreeFEM Example1.edp L260-261）。

    权重的作用：Γ_D 上 w=1/(1+α_d)≈0.95（高权重，可信数据），
    Γ_N 上 w=1/(1+α_n)≈0.33（低权重，补全数据不可靠）。
    这使得远离 Γ_D 的采样点 D 值更小，抑制不可靠区域的重建。
    """
    centroids = mesh.centroids
    bdry_edges = mesh.boundary_edges
    p0 = mesh.points[bdry_edges[:, 0]]
    p1 = mesh.points[bdry_edges[:, 1]]
    lengths = mesh.boundary_edge_lengths()

    # 按边分类 Γ_D / Γ_N
    n0 = bdry_edges[:, 0]
    n1 = bdry_edges[:, 1]
    edge_on_d = gamma_d_node_mask[n0] & gamma_d_node_mask[n1]
    alpha_edge = np.where(edge_on_d, alpha_d, alpha_n)

    # 加权因子：1/(1+α)，在 Γ_D 上大，在 Γ_N 上小
    weight = 1.0 / (1.0 + alpha_edge)

    # 距离计算
    diff0 = centroids[:, None, :] - p0[None, :, :]
    diff1 = centroids[:, None, :] - p1[None, :, :]
    r0_sq = np.sum(diff0**2, axis=2) + 1e-20
    r1_sq = np.sum(diff1**2, axis=2) + 1e-20

    # 加权积分：Σ_e w_e · L_e/2 · (1/r0² + 1/r1²)
    # 与 idsm.py initialize_r0_diagonal 相同结构，多了 w_e 权重
    integrand = weight[None, :] * lengths[None, :] * 0.5 * (1.0 / r0_sq + 1.0 / r1_sq)
    integral = np.sum(integrand, axis=1)

    # exponent=0.5 匹配 FreeFEM Example1.edp L260-261
    exponent = 0.5
    D = 1.0 / np.power(integral + 1e-30, exponent)

    # 靠近边界的截断（ε_Ω 截断）
    d2b = distance_to_boundary(mesh, centroids)
    D[d2b < epsilon] = 0.0
    return D


def apply_stabilizer_S(fine_mesh, coarse_mesh, values):
    """Apply S using fine->coarse->fine P0 projection."""
    coarse = fine_to_coarse_p0(fine_mesh, coarse_mesh, values)
    return coarse_to_fine_p0(fine_mesh, coarse_mesh, coarse)


def compute_auxiliary_eta(u_k, r_tilde_zeta, a=0.0, b=1.0):
    """Compute Eq. (4.12)-style auxiliary index in normalized coordinates."""
    eta = u_k.copy()
    at_upper = np.isclose(u_k, b, atol=1e-12)
    at_lower = np.isclose(u_k, a, atol=1e-12)
    mid = ~(at_upper | at_lower)
    eta[at_upper] = np.maximum(b, r_tilde_zeta[at_upper])
    eta[at_lower] = np.minimum(a, r_tilde_zeta[at_lower])
    eta[mid] = u_k[mid]
    return eta


def compute_safeguard_upsilon(zeta, eta_tilde, u_k, r_tilde_zeta, areas):
    """Compute Eq. (4.14) safeguard coefficient."""
    zu = _duality(zeta, u_k, areas)
    zr = _duality(zeta, r_tilde_zeta, areas)
    zt = _duality(zeta, eta_tilde, areas)
    if zu > zr > zt:
        denom = 2.0 * max(zu - zr, 1e-30)
        return np.clip(zu / denom, 0.0, 1.0)
    return 1.0


def compute_damping_factor_dfp(eta_hat, zeta_hat, r_tilde_zeta, areas, C_lambda, p):
    """Compute lambda_{k,p} with DFP-style Eq. (4.11)."""
    p_star = np.inf if p == 1 else p / (p - 1.0)
    omega_vol = float(np.sum(areas))
    scale = C_lambda * (omega_vol ** (0.0 if p_star == np.inf else 2.0 / p_star))
    zeta_eta = max(abs(_duality(zeta_hat, eta_hat, areas)), 1e-30)
    zeta_r = max(abs(_duality(zeta_hat, r_tilde_zeta, areas)), 1e-30)
    f1 = eta_hat / zeta_eta + r_tilde_zeta / zeta_r
    f2 = eta_hat / zeta_eta - r_tilde_zeta / zeta_r
    return scale * _l_p_norm(f1, areas, p) * _l_p_norm(f2, areas, p)


def compute_damping_factor_bfg(eta_hat, zeta_hat, r_tilde_zeta, areas, C_lambda, p):
    """Compute lambda_{k,p} with BFG-style Eq. (4.11)."""
    p_star = np.inf if p == 1 else p / (p - 1.0)
    omega_vol = float(np.sum(areas))
    scale = C_lambda * (omega_vol ** (0.0 if p_star == np.inf else 2.0 / p_star))
    zeta_eta = max(abs(_duality(zeta_hat, eta_hat, areas)), 1e-30)
    delta = eta_hat - r_tilde_zeta
    corr = 2.0 * delta - eta_hat * (_duality(zeta_hat, delta, areas) / zeta_eta)
    return scale * (_l_p_norm(eta_hat, areas, p) / zeta_eta) * _l_p_norm(corr, areas, p)


@dataclass
class StabilizedLowRankResolver:
    """Low-rank resolver with damping/smoothing stabilization."""

    base_diag: np.ndarray
    fine_mesh: object
    coarse_mesh: object
    method: str = "BFG"
    max_store: int = 30
    s_store: List[np.ndarray] = field(default_factory=list)
    y_store: List[np.ndarray] = field(default_factory=list)
    ry_store: List[np.ndarray] = field(default_factory=list)
    C_lambda: float = 1.0

    def _apply_with_store(self, input_vec, s_store, y_store, ry_store):
        result = self.base_diag * input_vec
        n_corr = min(len(s_store), self.max_store)
        if self.method.upper() == "DFP":
            for j in range(n_corr):
                sj = s_store[j]
                yj = y_store[j]
                ryj = ry_store[j]
                s_dot_y = np.dot(sj, yj)
                y_dot_ry = np.dot(yj, ryj)
                if abs(s_dot_y) > 1e-30:
                    result += (np.dot(input_vec, sj) / s_dot_y) * sj
                if abs(y_dot_ry) > 1e-30:
                    result -= (np.dot(input_vec, ryj) / y_dot_ry) * ryj
        else:
            for j in range(n_corr):
                sj = s_store[j]
                yj = y_store[j]
                ryj = ry_store[j]
                s_dot_y = np.dot(sj, yj)
                if abs(s_dot_y) <= 1e-30:
                    continue
                y_dot_ry = np.dot(yj, ryj)
                coeff = np.dot(input_vec, sj) / s_dot_y
                result += (1.0 + y_dot_ry / s_dot_y) * coeff * sj
                result -= (np.dot(input_vec, ryj) / s_dot_y) * sj
                result -= coeff * ryj
        return result

    def apply(self, vec):
        return self._apply_with_store(vec, self.s_store, self.y_store, self.ry_store)

    def apply_stabilized(self, vec):
        return self._apply_with_store(vec, self.s_store, self.y_store, self.ry_store)

    def stabilize(self, lambda_prev):
        """Eq. (4.10): R̃_{k+1} = S ∘ (1+λ)^{-1} R_k (1+λ)^{-1}

        对所有存储的低秩向量施加阻尼 (1+λ)^{-1} 和平滑算子 S。
        """
        damp = 1.0 / (1.0 + max(lambda_prev, 0.0))
        m = self.fine_mesh.n_triangles
        for j in range(len(self.s_store)):
            s = self.s_store[j]
            ry = self.ry_store[j]
            y = self.y_store[j]
            # 对 s, ry, y 分块施加 S 算子（粗网格投影 + 恢复）
            s_c = apply_stabilizer_S(self.fine_mesh, self.coarse_mesh, s[:m])
            s_v = apply_stabilizer_S(self.fine_mesh, self.coarse_mesh, s[m:])
            ry_c = apply_stabilizer_S(self.fine_mesh, self.coarse_mesh, ry[:m])
            ry_v = apply_stabilizer_S(self.fine_mesh, self.coarse_mesh, ry[m:])
            y_c = apply_stabilizer_S(self.fine_mesh, self.coarse_mesh, y[:m])
            y_v = apply_stabilizer_S(self.fine_mesh, self.coarse_mesh, y[m:])
            self.s_store[j] = damp * np.concatenate([s_c, s_v])
            self.ry_store[j] = damp * np.concatenate([ry_c, ry_v])
            self.y_store[j] = damp * np.concatenate([y_c, y_v])

    def update_correction(self, s_vec, y_vec, ry_vec):
        idx = len(self.s_store) % self.max_store
        if len(self.s_store) < self.max_store:
            self.s_store.append(s_vec.copy())
            self.y_store.append(y_vec.copy())
            self.ry_store.append(ry_vec.copy())
        else:
            self.s_store[idx] = s_vec.copy()
            self.y_store[idx] = y_vec.copy()
            self.ry_store[idx] = ry_vec.copy()

    def scale_D(self, scale):
        self.base_diag *= float(scale)


def run_idsm_partial(
    mesh,
    cauchy_data,
    gamma_d_info,
    sigma_bg=1.0,
    potential_bg=1e-10,
    sigma_range=0.01,
    potential_range=2e-10,
    alpha_d=0.05,
    alpha_n=2.0,
    n_iter=30,
    lowrank_method="BFG",
    problem_type="conductivity",
    coeff_known=False,
    gamma_D=4.0,
    epsilon_cutoff=0.02,
    p_norm=2.0,
    verbose=True,
    runtime_config=None,
):
    """Run partial-data IDSM with data completion + HR-DtN + stabilization."""
    n_data = len(cauchy_data["y_data"])
    M_tri = mesh.n_triangles

    if runtime_config is None:
        runtime_config = RuntimeConfig.from_env()
    device_info = runtime_config.resolve_device()
    if runtime_config.use_gpu and not device_info["enabled"]:
        warnings.warn(device_info["reason"], RuntimeWarning, stacklevel=2)
    y_data_list = cauchy_data["y_data"]
    y_empty_list = cauchy_data["y_empty"]
    source_funcs = cauchy_data["sources"]

    gamma_d_mask = gamma_d_info["node_mask"]
    M_bdry_D, M_bdry_N = assemble_partial_boundary_mass_matrix(mesh, gamma_d_mask)

    K_bg = assemble_stiffness_matrix(mesh, sigma_bg)
    M_pot = assemble_mass_matrix(mesh, potential_bg)
    A_op = K_bg + M_pot

    sigma_min = min(sigma_bg, sigma_range)
    sigma_max = max(sigma_bg, sigma_range)
    pot_min = min(potential_bg, potential_range)
    pot_max = max(potential_bg, potential_range)

    sigma_guess = np.full(M_tri, sigma_bg, dtype=np.float64)
    potential_guess = np.full(M_tri, potential_bg, dtype=np.float64)

    yU_list = []
    for k in range(n_data):
        if problem_type in ("potential", "double"):
            yU = solve_forward_general(mesh, sigma_guess, potential_guess, source_funcs[k])
        else:
            yU = solve_forward(mesh, sigma_guess, source_funcs[k])
        yU_list.append(yU)

    D = compute_heterogeneous_D(
        mesh,
        gamma_d_node_mask=gamma_d_mask,
        alpha_d=alpha_d,
        alpha_n=alpha_n,
        gamma=gamma_D,
        epsilon=epsilon_cutoff,
    )
    base_diag = np.concatenate([D, np.ones(M_tri)])
    coarse_mesh = generate_coarse_mesh(target_triangles=1770)
    resolver = StabilizedLowRankResolver(
        base_diag=base_diag,
        fine_mesh=mesh,
        coarse_mesh=coarse_mesh,
        method=lowrank_method,
        max_store=n_iter,
    )

    residual0 = 0.0
    for k in range(n_data):
        diff = yU_list[k] - y_data_list[k]
        residual0 += diff.dot(M_bdry_D.dot(diff))
    residual0 = float(np.sqrt(max(residual0, 0.0)))

    history = {
        "sigma_guess": [],
        "potential_guess": [],
        "residuals": [residual0],
        "lambda_history": [],
    }
    if verbose:
        print(f"Initial residual (GammaD): {residual0:.6e}")

    lambda_prev = 0.0
    areas2 = np.concatenate([mesh.areas, mesh.areas])

    for n in range(n_iter):
        w_list = []
        for k in range(n_data):
            y_complete = complete_data(y_data_list[k], yU_list[k], gamma_d_mask)
            scatter = y_complete - y_empty_list[k]
            w_k = apply_hr_dtn(
                mesh,
                scatter,
                A_op,
                alpha_d=alpha_d,
                alpha_n=alpha_n,
                M_bdry_D=M_bdry_D,
                M_bdry_N=M_bdry_N,
                sigma_bg=sigma_bg,
            )
            w_list.append(w_k)

        gradc, gradv = compute_p0_gradient(mesh, w_list, yU_list)
        zeta_k = np.concatenate([gradc, gradv])
        eta_k = resolver.apply(zeta_k)
        gradc = np.maximum(eta_k[:M_tri], 0.0)
        gradv = np.minimum(eta_k[M_tri:], 0.0)

        if not coeff_known:
            if problem_type in ("double", "conductivity"):
                sigma_guess = sigma_bg - gradc * abs(sigma_bg - sigma_range)
            else:
                sigma_guess[:] = sigma_bg
            if problem_type in ("double", "potential"):
                potential_guess = potential_bg - gradv * abs(potential_bg - potential_range)
            else:
                potential_guess[:] = potential_bg

        sigma_guess = np.clip(sigma_guess, sigma_min, sigma_max)
        potential_guess = np.clip(potential_guess, pot_min, pot_max)

        history["sigma_guess"].append(sigma_guess.copy())
        history["potential_guess"].append(potential_guess.copy())

        residual = 0.0
        for k in range(n_data):
            if problem_type in ("potential", "double"):
                yU_list[k] = solve_forward_general(mesh, sigma_guess, potential_guess, source_funcs[k])
            else:
                yU_list[k] = solve_forward(mesh, sigma_guess, source_funcs[k])
            diff = yU_list[k] - y_data_list[k]
            residual += diff.dot(M_bdry_D.dot(diff))
        residual = float(np.sqrt(max(residual, 0.0)))
        history["residuals"].append(residual)
        if verbose:
            print(f"Iter {n:3d}: residual = {residual:.6e}")

        if n == n_iter - 1:
            continue

        # Auxiliary update stage (Algorithm 5.1 steps 16-32).
        w_ax_list = []
        for k in range(n_data):
            scatter_current = yU_list[k] - y_empty_list[k]
            w_ax = apply_hr_dtn(
                mesh,
                scatter_current,
                A_op,
                alpha_d=alpha_d,
                alpha_n=alpha_n,
                M_bdry_D=M_bdry_D,
                M_bdry_N=M_bdry_N,
                sigma_bg=sigma_bg,
            )
            w_ax_list.append(w_ax)
        Axc, Axv = compute_p0_gradient(mesh, w_ax_list, yU_list)
        zeta_hat = np.concatenate([Axc, Axv])

        resolver.stabilize(lambda_prev)
        r_tilde_zeta = resolver.apply_stabilized(zeta_hat)

        cErr = (sigma_bg - sigma_guess) / max(abs(sigma_bg - sigma_range), 1e-30)
        vErr = (potential_bg - potential_guess) / max(abs(potential_bg - potential_range), 1e-30)
        u_norm = np.concatenate([cErr, vErr])

        eta_tilde = compute_auxiliary_eta(u_norm, r_tilde_zeta, a=0.0, b=1.0)
        upsilon = compute_safeguard_upsilon(zeta_hat, eta_tilde, u_norm, r_tilde_zeta, areas2)
        eta_hat = upsilon * eta_tilde + (1.0 - upsilon) * u_norm

        if n == 0:
            if lowrank_method.upper() == "DFP":
                base = compute_damping_factor_dfp(eta_hat, zeta_hat, r_tilde_zeta, areas2, 1.0, p_norm)
            else:
                base = compute_damping_factor_bfg(eta_hat, zeta_hat, r_tilde_zeta, areas2, 1.0, p_norm)
            resolver.C_lambda = 1.0 / max(base, 1e-30)

        if lowrank_method.upper() == "DFP":
            lambda_next = compute_damping_factor_dfp(
                eta_hat, zeta_hat, r_tilde_zeta, areas2, resolver.C_lambda, p_norm
            )
        else:
            lambda_next = compute_damping_factor_bfg(
                eta_hat, zeta_hat, r_tilde_zeta, areas2, resolver.C_lambda, p_norm
            )
        history["lambda_history"].append(float(lambda_next))

        ry_hat = resolver.apply_stabilized(zeta_hat)

        # 首次迭代分块缩放 D（匹配 idsm.py:570-594 和 FreeFEM L431-448）
        # 注意：必须先缩放 D 再存储低秩校正，否则 ry_hat 量级错误
        if n == 0:
            scale_c = 0.0
            scale_v = 0.0
            if problem_type in ('conductivity', 'double'):
                l1_c = np.sum(mesh.areas * np.abs(u_norm[:M_tri]))
                l1_rc = np.sum(mesh.areas * np.abs(ry_hat[:M_tri]))
                if l1_rc > 1e-30:
                    scale_c = l1_c / l1_rc
            if problem_type in ('potential', 'double'):
                l1_v = np.sum(mesh.areas * np.abs(u_norm[M_tri:]))
                l1_rv = np.sum(mesh.areas * np.abs(ry_hat[M_tri:]))
                if l1_rv > 1e-30:
                    scale_v = l1_v / l1_rv
            resolver.base_diag[:M_tri] *= scale_c
            resolver.base_diag[M_tri:] *= scale_v
            # 用缩放后的 R 重算 ry_hat
            ry_hat = resolver.apply_stabilized(zeta_hat)

        resolver.update_correction(eta_hat, zeta_hat, ry_hat)

        lambda_prev = float(lambda_next)

    history["sigma_final"] = sigma_guess
    history["potential_final"] = potential_guess
    return history
