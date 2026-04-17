"""
idsm.py — Iterative Direct Sampling Method (IDSM)

Educational reference implementation of Algorithm 3.2 from Ito, Jin, Wang, Zou
(2025) (Paper 1) for the IDSM iteration scheme in linear elliptic inverse problems.

Core formulas:
  - Regularized DtN map Λ_α(A): Eq. (3.5)/(3.20), via two Robin BVPs
  - Indicator ζ_k = Σ_ℓ B_τ[y_ℓ]* w_ℓ: Eq. (3.17), aggregating multiple Cauchy data sets
  - Low-rank correction R_k: DFP (Eq. 3.14) or BFG (Eq. 3.15) quasi-Newton update
  - Projection P: box constraint, Section 4.1

FreeFEM reference: Example1.edp
  - L252-264: R₀ initialization (diagFunc)
  - L278-315: Rsolver (DFP/BFG low-rank correction)
  - L317-453: IDSM main loop

For EIT (Example 1):
  A = −∇·(σ₀∇·), B[u](y) = −∇·(u∇y), B_τ[y]*w = ∇y·∇w
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import warnings

from .mesh import EllipticMesh
from .fem import (
    assemble_stiffness_matrix,
    assemble_mass_matrix,
    assemble_boundary_mass_matrix,
    assemble_boundary_mean_constraint,
    solve_neumann_system,
    solve_robin_system,
)
from .forward_solver import solve_forward, solve_forward_general
from .config import RuntimeConfig


# ============================================================
# Normal derivative on the elliptic boundary
# ============================================================

def compute_ellipse_normal_derivative(mesh, z, sigma_bg=1.0, a=1.0, b=0.8):
    """计算 σ₀ ∂z/∂n（向后兼容别名，内部委托给通用实现）。

    委托给 fem.compute_boundary_normal_derivative，不再依赖椭圆几何。
    保留此函数名是为了兼容所有现有调用点。

    原始实现使用椭圆解析法向 n̂ = (x₁/a², x₂/b²)/‖·‖，
    通用版本使用基于边几何的外法向，适用于任意 2D 域。

    Reference: FreeFEM Example1.edp L329-330

    Parameters
    ----------
    mesh : EllipticMesh
    z : array (N,) — P1 FEM 解
    sigma_bg : float — 背景电导率 σ₀
    a, b : float — 椭圆半轴（保留参数签名兼容，不再使用）

    Returns
    -------
    flux : array (N,) — σ₀ ∂z/∂n，边界节点有值，内部节点为零
    """
    from .fem import compute_boundary_normal_derivative
    return compute_boundary_normal_derivative(mesh, z, sigma_bg=sigma_bg)


# ============================================================
# Regularized DtN map
# ============================================================

def apply_regularized_dtn(mesh, v, A_op, alpha, M_bdry=None,
                          sigma_bg=1.0, a=1.0, b=0.8):
    """Apply the regularized DtN map Λ_α(A) via two Robin BVPs.

    数学背景（Eq. 3.5）：
      经典 DtN 映射 Λ(A): H^{1/2}(Γ) → H^{-1/2}(Γ) 是无界算子，
      直接计算会放大噪声。正则化版本 Λ_α(A) 由 Robin 问题定义：
        Λ_α(A) v = σ₀ ∂z_α/∂n，其中 Az_α = 0 in Ω, z_α + α σ₀∂z_α/∂n = v on Γ
      当 α → 0 时 Λ_α → Λ（经典 DtN）；α 越大正则化越强，噪声敏感度越低。

    离散实现（Eq. 3.20）：
      Step 1: weak form [A_op + (1/α) M_Γ] z = (1/α) M_Γ v  （第一次 Robin 求解）
      Step 2: g = σ₀ ∂z/∂n on Γ                              （边界法向导数）
      Step 3: [A_op + (1/α) M_Γ] w = (1/α) M_Γ g             （第二次 Robin 求解）

    由 Lemma 3.2，双 Robin 求解计算
      G[u]* Λ_α(A)* Λ_α(A) 的核心部分。

    Parameters
    ----------
    mesh : EllipticMesh
    v : array (N,) — Robin boundary data
    A_op : sparse (N, N) — interior operator
    alpha : float — regularization parameter
    M_bdry : sparse (N, N) or None — precomputed boundary mass matrix
    sigma_bg : float
    a, b : float — ellipse semi-axes

    Returns
    -------
    w : array (N,) — second Robin solve result
    """
    if M_bdry is None:
        M_bdry = assemble_boundary_mass_matrix(mesh)

    # Robin system matrix (shared by both solves)
    system_matrix = A_op + (1.0 / alpha) * M_bdry

    rhs1 = (1.0 / alpha) * M_bdry.dot(v)
    z = spsolve(system_matrix, rhs1)

    g = compute_ellipse_normal_derivative(mesh, z, sigma_bg, a, b)

    rhs2 = (1.0 / alpha) * M_bdry.dot(g)
    w = spsolve(system_matrix, rhs2)

    return w


# ============================================================
# P0 gradient computation
# ============================================================

def compute_p0_gradient(mesh, w_list, y_list):
    """Compute the IDSM gradient: Σ_ℓ B_τ[y_ℓ]* w_ℓ in P0 representation.

    For EIT:
      gradc = Σ_ℓ ∇w_ℓ · ∇y_ℓ          (conductivity gradient)
      gradv = Σ_ℓ w_ℓ · y_ℓ             (potential gradient)

    Paper 1, Eq. (3.17) / FreeFEM L333-334:
      gradc = gradc + Grad(unknown)' * Grad(yU[k]);
      gradv = gradv + unknown * yU[k];

    Parameters
    ----------
    mesh : EllipticMesh
    w_list : list of array (N,) — adjoint solutions
    y_list : list of array (N,) — forward solutions

    Returns
    -------
    gradc : array (M,) — P0 conductivity gradient
    gradv : array (M,) — P0 potential gradient
    """
    M = mesh.n_triangles
    tri = mesh.triangles
    gp = mesh.grad_phi

    gradc = np.zeros(M)
    gradv = np.zeros(M)

    for w, y in zip(w_list, y_list):
        grad_w = (w[tri[:, 0], None] * gp[:, 0, :]
                  + w[tri[:, 1], None] * gp[:, 1, :]
                  + w[tri[:, 2], None] * gp[:, 2, :])

        grad_y = (y[tri[:, 0], None] * gp[:, 0, :]
                  + y[tri[:, 1], None] * gp[:, 1, :]
                  + y[tri[:, 2], None] * gp[:, 2, :])

        gradc += grad_w[:, 0] * grad_y[:, 0] + grad_w[:, 1] * grad_y[:, 1]

        # P0 approximation of w·y: vertex average / 3 (centroid quadrature)
        gradv += (w[tri[:, 0]] * y[tri[:, 0]]
                  + w[tri[:, 1]] * y[tri[:, 1]]
                  + w[tri[:, 2]] * y[tri[:, 2]]) / 3.0

    return gradc, gradv


# ============================================================
# Low-rank preconditioner
# ============================================================

class LowRankPreconditioner:
    """DFP/BFG low-rank preconditioner R_k.

    Paper 1, Eq. (3.14)-(3.15):
      DFP: δR ξ = (ξ·s)/(s·y) s − (ξ·Ry)/(y·Ry) Ry
      BFG: δR ξ = (1 + y·Ry/s·y)(ξ·s/s·y) s − (ξ·Ry/s·y) s − (ξ·s/s·y) Ry

    FreeFEM L278-315 (Rsolver function):
      - First applies diagonal part: result = diagFunc * input
      - Then accumulates low-rank correction terms

    All inner products are plain Euclidean dot products (no area weighting),
    matching FreeFEM's a'*b convention.

    Attributes
    ----------
    diag : array (D,) — diagonal preconditioner R₀
    method : str — 'DFP' or 'BFG'
    max_store : int — maximum stored corrections
    s_store, y_store, ry_store : lists of arrays
    """

    def __init__(self, diag, method='BFG', max_store=22):
        """Initialize the low-rank preconditioner.

        Parameters
        ----------
        diag : array (D,) — diagonal part R₀
        method : str — 'DFP' or 'BFG' (Eq. 3.14 or 3.15)
        max_store : int — maximum stored correction pairs (FreeFEM: storeNum=22)
        """
        self.diag = diag.copy()
        self.method = method
        self.max_store = max_store
        self.s_store = []
        self.y_store = []
        self.ry_store = []
        self.count = 0

    def apply(self, input_vec):
        """Apply R_k to a vector: ``result = R_k · input_vec``.

        Matches the FreeFEM ``Rsolver`` (L278–315): multiply by the diagonal
        ``R₀``, then add the accumulated DFP (Eq. 3.14) or BFG (Eq. 3.15)
        low-rank corrections.

        Parameters
        ----------
        input_vec : array (D,)

        Returns
        -------
        result : array (D,)
        """
        result = self.diag * input_vec

        n_corrections = min(self.count, self.max_store)

        if self.method == 'DFP':
            # FreeFEM L282–294
            for j in range(n_corrections):
                sj = self.s_store[j]
                yj = self.y_store[j]
                ryj = self.ry_store[j]

                s_dot_y = np.dot(sj, yj)
                y_dot_ry = np.dot(yj, ryj)

                if abs(s_dot_y) > 1e-30:
                    result += (np.dot(input_vec, sj) / s_dot_y) * sj
                if abs(y_dot_ry) > 1e-30:
                    result -= (np.dot(input_vec, ryj) / y_dot_ry) * ryj

        elif self.method == 'BFG':
            # FreeFEM L298–313
            for j in range(n_corrections):
                sj = self.s_store[j]
                yj = self.y_store[j]
                ryj = self.ry_store[j]

                s_dot_y = np.dot(sj, yj)
                y_dot_ry = np.dot(yj, ryj)

                if abs(s_dot_y) > 1e-30:
                    coeff = np.dot(input_vec, sj) / s_dot_y
                    result += (1.0 + y_dot_ry / s_dot_y) * coeff * sj
                    result -= (np.dot(input_vec, ryj) / s_dot_y) * sj
                    result -= coeff * ryj

        return result

    def update(self, s, y, ry):
        """Append or overwrite a stored low-rank correction (circular buffer).

        Corresponds to FreeFEM L449–451:
          sStore(:, loopIndex % storeNum) = sk;
          yStore(:, loopIndex % storeNum) = yk;
          ryStore(:, loopIndex % storeNum) = ryk;

        Parameters
        ----------
        s : array (D,) — normalized reconstruction u_{k+1}
        y : array (D,) — A·x_{k+1} (predicted gradient ζ̃_{k+1})
        ry : array (D,) — R_k · y
        """
        idx = self.count % self.max_store
        if self.count < self.max_store:
            self.s_store.append(s.copy())
            self.y_store.append(y.copy())
            self.ry_store.append(ry.copy())
        else:
            self.s_store[idx] = s.copy()
            self.y_store[idx] = y.copy()
            self.ry_store[idx] = ry.copy()
        self.count += 1

    def scale_diagonal(self, scale_c, scale_v, n_cond):
        """Scale the R₀ diagonal blocks after the first iteration’s L¹ scaling.

        Paper 1: scale R₀ using ‖u₁‖_{L¹(Ω)} / ‖R₀ ζ̃₁‖_{L¹(Ω)}.
        FreeFEM L445–446:
          diagFunc(0*P0solve.ndof:1*P0solve.ndof) *= scale1;
          diagFunc(1*P0solve.ndof:2*P0solve.ndof) *= scale2.

        Parameters
        ----------
        scale_c : float — factor for the conductivity (first M P0 DOFs)
        scale_v : float — factor for the potential block
        n_cond : int — number of P0 DOFs in the conductivity block
        """
        self.diag[:n_cond] *= scale_c
        self.diag[n_cond:] *= scale_v


# ============================================================
# R₀ initialization
# ============================================================

def initialize_r0_diagonal(mesh, cond_exponent=0.5, pot_exponent=0.0):
    """初始化 R₀ 对角预条件子。

    Paper 1, Section 4.1：离散 R₀ 基于 ‖∇Φ_x‖_{L²(Γ)} 的构造。

    FreeFEM L252–264 (diagFunc):
      conductivity: diagFunc(i) = 1/((∫_Γ 1/dis^2.0)^cond_exponent)
      potential:    diagFunc(i+M) = 1/((∫_Γ 1/dis^2.0)^pot_exponent)

    其中 dis = |x_i − x'|，x_i 是三角形 i 的质心，x' 在 Γ 上运行。

    Parameters
    ----------
    mesh : EllipticMesh
    cond_exponent : float
        电导率块的指数，默认 0.5（FreeFEM L260–261）。
    pot_exponent : float
        势块的指数，默认 0.0（即恒等，FreeFEM L262–263）。

    Returns
    -------
    diag : array (2*M,) — [conductivity_part, potential_part]
    """
    M = mesh.n_triangles
    centroids = mesh.centroids

    bdry_edges = mesh.boundary_edges
    bdry_lengths = mesh.boundary_edge_lengths()
    p0 = mesh.points[bdry_edges[:, 0]]
    p1 = mesh.points[bdry_edges[:, 1]]

    diff0 = centroids[:, None, :] - p0[None, :, :]
    diff1 = centroids[:, None, :] - p1[None, :, :]

    r0_sq = np.sum(diff0 ** 2, axis=2) + 1e-20
    r1_sq = np.sum(diff1 ** 2, axis=2) + 1e-20

    # 梯形法则在 ∂Ω 上：Σ_e (L_e/2) (1/r0² + 1/r1²)
    integrand = bdry_lengths[None, :] * 0.5 * (1.0 / r0_sq + 1.0 / r1_sq)
    integral = np.sum(integrand, axis=1)

    # 电导率块：1/(∫_Γ 1/dis²)^cond_exponent
    diag_cond = 1.0 / np.power(integral + 1e-30, cond_exponent)
    # 势块：1/(∫_Γ 1/dis²)^pot_exponent
    diag_pot = 1.0 / np.power(integral + 1e-30, pot_exponent)

    return np.concatenate([diag_cond, diag_pot])


# ============================================================
# IDSM main driver
# ============================================================

def run_idsm(mesh, cauchy_data, sigma_bg=1.0, potential_bg=1e-10,
             sigma_range=0.01, potential_range=2e-10,
             alpha=1.0, n_iter=22, lowrank_method='BFG',
             problem_type='conductivity', coeff_known=False,
             cond_exponent=0.5, pot_exponent=0.0,
             verbose=True, runtime_config=None):
    """Run the IDSM iterative scheme (Algorithm 3.2).

    For EIT (linear A, Example 1):
      A = K(σ₀) + M(v₀), B[u](y) = K(u) (admittance), B_τ[y]*w = ∇y·∇w

    Parameters
    ----------
    mesh : EllipticMesh
    cauchy_data : dict from generate_cauchy_data
    sigma_bg : float — background conductivity σ₀
    potential_bg : float — background potential v₀
    sigma_range : float — conductivity search bound
    potential_range : float — potential search bound
    alpha : float — Robin regularization parameter (Paper 1: α=1.0)
    n_iter : int — maximum iterations (FreeFEM: storeNum=22)
    lowrank_method : str — 'DFP' or 'BFG'
    problem_type : str — 'conductivity', 'potential', or 'double'
    coeff_known : bool — True for known-coefficient mapping
    cond_exponent : float — R₀ 对角中 conductivity 块的指数（FreeFEM Example1: 0.5）
    pot_exponent : float — R₀ 对角中 potential 块的指数（FreeFEM Example3: 1.5, Example1: 0.0）
    verbose : bool
    runtime_config : RuntimeConfig or None

    Returns
    -------
    history : dict with keys:
        'sigma_guess' : list of (M,) per-iteration conductivity
        'potential_guess' : list of (M,) per-iteration potential
        'residuals' : list of float
        'sigma_final', 'potential_final' : arrays
    """
    n_data = len(cauchy_data['y_data'])
    M_tri = mesh.n_triangles

    if runtime_config is None:
        runtime_config = RuntimeConfig.from_env()
    device_info = runtime_config.resolve_device()
    if runtime_config.use_gpu and not device_info["enabled"]:
        warnings.warn(device_info["reason"], RuntimeWarning, stacklevel=2)

    y_data_list = cauchy_data['y_data']
    y_empty_list = cauchy_data['y_empty']
    source_funcs = cauchy_data['sources']

    # ================================================================
    # Pre-iteration: assemble operators, compute fixed adjoints
    # ================================================================

    K_bg = assemble_stiffness_matrix(mesh, sigma_bg)
    M_pot = assemble_mass_matrix(mesh, potential_bg)
    A_op = K_bg + M_pot

    M_bdry = assemble_boundary_mass_matrix(mesh)

    # Algorithm 3.2, Step 2: Compute fixed adjoint w_ℓ via double Robin solve
    if verbose:
        print("Pre-iteration: computing fixed adjoints via double Robin...")

    w_fixed = []
    for k in range(n_data):
        # Scattered field y_data − y_empty; dual via double Robin (cf. FreeFEM L322–331)
        scatter = y_data_list[k] - y_empty_list[k]
        w_k = apply_regularized_dtn(mesh, scatter, A_op, alpha, M_bdry,
                                     sigma_bg)
        w_fixed.append(w_k)

    diag = initialize_r0_diagonal(mesh, cond_exponent=cond_exponent,
                                   pot_exponent=pot_exponent)
    R = LowRankPreconditioner(diag, method=lowrank_method, max_store=n_iter)

    sigma_guess = np.full(M_tri, sigma_bg)
    potential_guess = np.full(M_tri, potential_bg)

    yU_list = []
    for k in range(n_data):
        if problem_type in ('potential', 'double'):
            yU = solve_forward_general(mesh, sigma_guess, potential_guess,
                                        source_funcs[k])
        else:
            yU = solve_forward(mesh, sigma_guess, source_funcs[k])
        yU_list.append(yU)

    residual = 0.0
    for k in range(n_data):
        diff = yU_list[k] - y_data_list[k]
        residual += diff.dot(M_bdry.dot(diff))
    residual = np.sqrt(residual)

    if verbose:
        print(f"Initial residual: {residual:.6e}")

    # ================================================================
    # Iterative loop (Algorithm 3.2, Steps 3-11)
    # ================================================================

    history = {
        'sigma_guess': [],
        'potential_guess': [],
        'residuals': [residual],
    }

    sigma_min = min(sigma_bg, sigma_range)
    sigma_max = max(sigma_bg, sigma_range)
    pot_min = min(potential_bg, potential_range)
    pot_max = max(potential_bg, potential_range)

    for n in range(n_iter):
        # Step 4: ζ_k = Σ_ℓ B_τ[y_ℓ]* w_ℓ
        gradc, gradv = compute_p0_gradient(mesh, w_fixed, yU_list)

        # Step 5: η_k = R_k · ζ_k
        Atb = np.concatenate([gradc, gradv])
        RAx = R.apply(Atb)
        gradc = RAx[:M_tri]
        gradv = RAx[M_tri:]

        # Thresholding (FreeFEM L339-340)
        # Direction depends on whether sigma_range < sigma_bg (insulating)
        # or sigma_range > sigma_bg (conductive).
        if sigma_range <= sigma_bg:
            gradc = np.maximum(gradc, 0.0)
        else:
            gradc = np.minimum(gradc, 0.0)
        gradv = np.minimum(gradv, 0.0)

        # Step 6: u_{k+1} = P(η_k) — update guess with box constraint
        if not coeff_known:
            if problem_type in ('conductivity', 'double'):
                sigma_guess = sigma_bg - gradc * abs(sigma_bg - sigma_range)
            else:
                sigma_guess = np.full(M_tri, sigma_bg)

            if problem_type in ('potential', 'double'):
                potential_guess = potential_bg - gradv * abs(potential_bg - potential_range)
            else:
                potential_guess = np.full(M_tri, potential_bg)
        else:
            if problem_type in ('conductivity', 'double'):
                gradc_sort = np.sort(gradc)
                tau1 = gradc_sort[int(len(gradc_sort) * 0.99)]
                tau2 = gradc_sort[int(len(gradc_sort) * 0.01)]
                slope = abs(sigma_range - sigma_bg) / max(abs(tau1 - tau2), 1e-30)
                sigma_guess = 0.5 * sigma_guess + 0.5 * (slope * (tau1 - gradc) + sigma_min)
            else:
                sigma_guess = np.full(M_tri, sigma_bg)

            if problem_type in ('potential', 'double'):
                gradv_sort = np.sort(gradv)
                tau1 = gradv_sort[int(len(gradv_sort) * 0.99)]
                tau2 = gradv_sort[int(len(gradv_sort) * 0.01)]
                slope = abs(potential_range - potential_bg) / max(abs(tau1 - tau2), 1e-30)
                potential_guess = (0.5 * potential_guess
                                   + 0.5 * (slope * (tau1 - gradv) + pot_min))
            else:
                potential_guess = np.full(M_tri, potential_bg)

        # Box constraint (FreeFEM L369-372)
        sigma_guess = np.clip(sigma_guess, sigma_min, sigma_max)
        potential_guess = np.clip(potential_guess, pot_min, pot_max)

        history['sigma_guess'].append(sigma_guess.copy())
        history['potential_guess'].append(potential_guess.copy())

        # Step 7: Solve forward problem with updated guess + compute residual
        residual = 0.0
        for k in range(n_data):
            if problem_type in ('potential', 'double'):
                yU_list[k] = solve_forward_general(mesh, sigma_guess, potential_guess,
                                                    source_funcs[k])
            else:
                yU_list[k] = solve_forward(mesh, sigma_guess, source_funcs[k])
            diff = yU_list[k] - y_data_list[k]
            residual += diff.dot(M_bdry.dot(diff))
        residual = np.sqrt(residual)
        history['residuals'].append(residual)

        if verbose:
            print(f"Iter {n:3d}: residual = {residual:.6e}")

        # Step 8: Auxiliary scatter ζ̃_{k+1} via double Robin
        w_ax_list = []
        for k in range(n_data):
            scatter_current = yU_list[k] - y_empty_list[k]
            w_ax = apply_regularized_dtn(mesh, scatter_current, A_op, alpha,
                                          M_bdry, sigma_bg)
            w_ax_list.append(w_ax)

        Axc, Axv = compute_p0_gradient(mesh, w_ax_list, yU_list)

        # Step 9: Low-rank update vectors
        cErr = (sigma_bg - sigma_guess) / max(abs(sigma_bg - sigma_range), 1e-30)
        vErr = (potential_bg - potential_guess) / max(abs(potential_bg - potential_range), 1e-30)

        yk = np.concatenate([Axc, Axv])
        sk = np.concatenate([cErr, vErr])
        ryk = R.apply(yk)

        # First-iteration scaling (FreeFEM L432-448)
        if n == 0:
            rykc = ryk[:M_tri]
            rykv = ryk[M_tri:]
            areas = mesh.areas

            scale_c = 0.0
            scale_v = 0.0

            if problem_type in ('conductivity', 'double'):
                l1_s = np.sum(areas * np.abs(cErr))
                l1_ry = np.sum(areas * np.abs(rykc))
                if l1_ry > 1e-30:
                    scale_c = l1_s / l1_ry

            if problem_type in ('potential', 'double'):
                l1_s = np.sum(areas * np.abs(vErr))
                l1_ry = np.sum(areas * np.abs(rykv))
                if l1_ry > 1e-30:
                    scale_v = l1_s / l1_ry

            R.scale_diagonal(scale_c, scale_v, M_tri)
            ryk = R.apply(yk)

        # Step 10: Store low-rank correction
        R.update(sk, yk, ryk)

    history['sigma_final'] = sigma_guess
    history['potential_final'] = potential_guess

    return history
