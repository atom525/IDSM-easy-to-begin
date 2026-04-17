"""
dsm.py - 经典直接采样方法 (Classical Direct Sampling Method, DSM)

严格按照 Ito, Jin, Wang, Zou (2025) Section 2.2 实现：
  指标函数 η(x) = ⟨G(·,x), y_d^s⟩_{H^γ(Γ)} / ‖G(·,x)‖_{H^γ(Γ)}    (Eq. 2.8)

其中：
  - G(·,x) 是 Green 函数（Neumann 函数）
  - y_d^s 是散射数据（背景解减测量数据）
  - H^γ(Γ) 是由 (−Δ_Γ)^γ 诱导的边界 Sobolev 空间

分子计算 (Eq. 2.9)：求解辅助 Neumann 问题
  −∇·(σ₀∇ζ) = 0 in Ω,  σ₀ ∂ζ/∂n = (−Δ_Γ)^γ y_d^s on Γ

分母近似 (Eq. 2.10)：d(x,Γ)^γ 或积分近似

参考: FreeFEM Example1.edp L252-264 (diagFunc) 和 L317-335 (迭代 0).
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, eigsh
from .mesh import EllipticMesh, generate_sampling_grid
from .fem import (
    assemble_stiffness_matrix,
    assemble_boundary_mass_matrix,
    assemble_boundary_load,
    assemble_boundary_mean_constraint,
    solve_neumann_system,
)
from .utils import distance_to_boundary


# ============================================================
# Laplace-Beltrami 算子离散化
# ============================================================

class LaplaceBeltramiOperator:
    """边界 Γ 上 (−Δ_Γ)^γ 的离散表示。

    通过一维有限元特征分解构建：
      K_Γ v_i = λ_i M_Γ v_i

    其中 K_Γ 是一维边界刚度矩阵（弧长导数），
    M_Γ 是一维边界质量矩阵。

    算子作用: (−Δ_Γ)^γ f = Σ_{i: λ_i>0} λ_i^γ (f, v_i)_{M_Γ} v_i

    展开系数: c_i = v_i^T M_Γ f （因 V^T M_Γ V = I）

    Attributes
    ----------
    eigenvalues : array (K,) — 排序后的特征值 λ_i
    eigenvectors : array (B, K) — M_Γ-正交归一化的特征向量
    gamma : float — 分数幂 γ
    n_full : int — 域节点总数
    boundary_nodes : array — 边界节点索引
    M_bdry_local : sparse (B, B) — 局部边界质量矩阵
    """

    def __init__(self, eigenvalues, eigenvectors, gamma, n_full,
                 boundary_nodes, M_bdry_local):
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.gamma = gamma
        self.n_full = n_full
        self.boundary_nodes = boundary_nodes
        self.M_bdry_local = M_bdry_local

        self._MV = M_bdry_local.dot(eigenvectors)

        self._lam_gamma = np.zeros(len(eigenvalues))
        max_lam = np.max(eigenvalues) if len(eigenvalues) > 0 else 1.0
        self._zero_threshold = max(1e-8, max_lam * 1e-10)
        for i in range(len(eigenvalues)):
            if eigenvalues[i] > self._zero_threshold:
                self._lam_gamma[i] = eigenvalues[i] ** gamma

    def apply(self, f_full):
        """将 (−Δ_Γ)^γ 作用于全域向量（仅边界部分有效）。"""
        f_bdry = f_full[self.boundary_nodes]
        result_bdry = self.apply_to_boundary(f_bdry)

        result = np.zeros(self.n_full)
        result[self.boundary_nodes] = result_bdry
        return result

    def apply_to_boundary(self, f_bdry):
        """将 (−Δ_Γ)^γ 直接作用于边界向量。

        c_i = (M_Γ v_i)^T f,  结果 = Σ_i λ_i^γ c_i v_i
        """
        coeffs = self._MV.T.dot(f_bdry)
        weighted = self._lam_gamma * coeffs
        result_bdry = self.eigenvectors.dot(weighted)
        return result_bdry


def discretize_laplace_beltrami(mesh, gamma=0.5, n_eigenvalues=None):
    """离散化边界 Γ 上的 Laplace-Beltrami 算子 (−Δ_Γ)^γ。

    步骤：
      1. 组装一维边界刚度矩阵: (K_Γ)_{ij} = ∫_Γ (dφ_i/ds)(dφ_j/ds) ds
      2. 组装一维边界质量矩阵: (M_Γ)_{ij} = ∫_Γ φ_i φ_j ds
      3. 求解广义特征值问题: K_Γ v = λ M_Γ v
      4. 构建 (−Δ_Γ)^γ f = Σ_i λ_i^γ (f, v_i)_{M_Γ} v_i

    对于每条边界边 e=(n0,n1)，长度 L_e：
      (K_Γ)_e = (1/L_e) [[1, −1], [−1, 1]]   （一维刚度）
      (M_Γ)_e = (L_e/6) [[2, 1], [1, 2]]     （一维质量）

    参考: Paper 1, Section 2.2 — (−Δ_Γ)^γ 定义了 H^γ(Γ) 内积。
    """
    bdry_nodes = mesh.boundary_nodes
    n_bdry = len(bdry_nodes)

    if n_eigenvalues is None:
        n_eigenvalues = n_bdry - 1

    node_to_local = {int(node): i for i, node in enumerate(bdry_nodes)}

    edges = mesh.boundary_edges
    lengths = mesh.boundary_edge_lengths()

    rows = []
    cols = []
    k_vals = []
    m_vals = []

    for k in range(len(edges)):
        n0, n1 = int(edges[k, 0]), int(edges[k, 1])
        L = lengths[k]

        if n0 not in node_to_local or n1 not in node_to_local:
            continue

        i0 = node_to_local[n0]
        i1 = node_to_local[n1]

        rows.extend([i0, i0, i1, i1])
        cols.extend([i0, i1, i0, i1])
        k_vals.extend([1.0 / L, -1.0 / L, -1.0 / L, 1.0 / L])
        m_vals.extend([L / 3.0, L / 6.0, L / 6.0, L / 3.0])

    K_bdry = sparse.csr_matrix(
        (np.array(k_vals), (np.array(rows), np.array(cols))),
        shape=(n_bdry, n_bdry)
    )
    M_bdry = sparse.csr_matrix(
        (np.array(m_vals), (np.array(rows), np.array(cols))),
        shape=(n_bdry, n_bdry)
    )

    n_eig = min(n_eigenvalues, n_bdry - 2)
    eigenvalues, eigenvectors = eigsh(K_bdry, k=n_eig, M=M_bdry, which='SM')

    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    for i in range(len(eigenvalues)):
        norm = np.sqrt(eigenvectors[:, i].dot(M_bdry.dot(eigenvectors[:, i])))
        if norm > 1e-14:
            eigenvectors[:, i] /= norm

    return LaplaceBeltramiOperator(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        gamma=gamma,
        n_full=mesh.n_points,
        boundary_nodes=bdry_nodes,
        M_bdry_local=M_bdry,
    )


# ============================================================
# 散射数据
# ============================================================

def compute_scattering_data(cauchy_data):
    """计算散射数据 y_d^s = y_∅ − y_d。

    Paper 1, Section 2.1 (Eq. 2.7 之后):
      y_d^s = T·y_∅ − y_d
    对于 Neumann 边界条件，T 是恒等（trace）算子。
    """
    scatter = []
    for y_empty, y_data in zip(cauchy_data['y_empty'], cauchy_data['y_data']):
        scatter.append(y_empty - y_data)
    return scatter


# ============================================================
# DSM 分子：辅助 PDE 求解
# ============================================================

def compute_dsm_numerator(mesh, scatter_full, lb_operator, sigma_bg=1.0):
    """计算 DSM 分子: ζ(x) = ⟨G(·,x), y_d^s⟩_{H^γ(Γ)}。

    求解辅助 Neumann 问题 (Paper 1, Eq. 2.9):
      −∇·(σ₀∇ζ) = 0 in Ω,  σ₀ ∂ζ/∂n = (−Δ_Γ)^γ y_d^s on Γ
      约束: ∫_Γ ζ ds = 0

    解 ζ(x) 同时给出所有内点 x 处的分子值。
    """
    g_full = lb_operator.apply(scatter_full)

    M_bdry = assemble_boundary_mass_matrix(mesh)
    b = M_bdry.dot(g_full)

    K = assemble_stiffness_matrix(mesh, sigma_bg)
    B = assemble_boundary_mean_constraint(mesh)

    zeta = solve_neumann_system(K, b, B)
    return zeta


# ============================================================
# DSM 分母：归一化因子
# ============================================================

def compute_dsm_denominator_distance(mesh, points, gamma=0.5):
    """分母近似方法 1：基于距离。

    Paper 1, Eq. (2.10)（d=2 情形）:
      ‖G(·,x)‖_{H^γ(Γ)} ≈ C / d(x,Γ)^γ
    """
    dist = distance_to_boundary(mesh, points)
    dist = np.maximum(dist, 1e-12)
    denom = 1.0 / (dist ** gamma)
    return denom


def compute_dsm_denominator_integral(mesh, points):
    """分母近似方法 2：基于积分（FreeFEM 风格）。

    参考 FreeFEM Example1.edp L260-261:
      diagFunc(i) = 1 / ((int1d(Th)(1/dis^2.0))^0.5)

    即 R(x) = 1 / √(∫_Γ 1/|x−x'|² ds(x'))

    这近似 1/‖∇Φ_x‖_{L²(Γ)}，其中 Φ_x = −1/(2π) ln|x−x'| 是基本解。
    """
    bdry_edges = mesh.boundary_edges
    bdry_lengths = mesh.boundary_edge_lengths()

    p0 = mesh.points[bdry_edges[:, 0]]
    p1 = mesh.points[bdry_edges[:, 1]]

    diff0 = points[:, None, :] - p0[None, :, :]
    diff1 = points[:, None, :] - p1[None, :, :]

    r0_sq = np.sum(diff0 ** 2, axis=2) + 1e-20
    r1_sq = np.sum(diff1 ** 2, axis=2) + 1e-20

    integrand = bdry_lengths[None, :] * 0.5 * (1.0 / r0_sq + 1.0 / r1_sq)
    integral = np.sum(integrand, axis=1)

    denom = np.sqrt(integral + 1e-30)
    return denom


# ============================================================
# DSM 指标函数 — 主驱动
# ============================================================

def compute_dsm_indicator(mesh, cauchy_data, gamma=0.5, n_grid=201,
                          denom_method='integral', sigma_bg=1.0,
                          n_eigenvalues=None):
    """计算均匀网格上的 DSM 指标函数 η(x)。

    Paper 1, Eq. (2.8):
      η(x) = ⟨G(·,x), y_d^s⟩_{H^γ(Γ)} / ‖G(·,x)‖_{H^γ(Γ)}

    对 L 组 Cauchy 数据，取绝对值聚合:
      η(x) = (1/L) Σ_ℓ |ζ_ℓ(x)| / R(x)

    流程:
      1. 离散化 (−Δ_Γ)^γ（特征分解）
      2. 计算散射数据 y_d^s
      3. 对每组数据求解辅助 PDE → ζ_ℓ(x)
      4. 在 FEM 网格上聚合分子
      5. 插值到均匀扫描网格
      6. 计算分母 R(x)
      7. η(x) = 分子 / 分母

    Parameters
    ----------
    mesh : EllipticMesh
    cauchy_data : dict from generate_cauchy_data
    gamma : float — H^γ(Γ) parameter (default 0.5)
    n_grid : int — grid points per axis
    denom_method : str — 'integral' or 'distance'
    sigma_bg : float — background conductivity
    n_eigenvalues : int or None

    Returns
    -------
    result : dict with keys:
        'indicator' : array (n_grid, n_grid) — indicator values (NaN outside domain)
        'grid_x', 'grid_y' : arrays
        'mask' : array (n_grid, n_grid), bool
        'zeta_sum' : array (N,) — aggregated numerator on FEM mesh
        'lb_operator' : LaplaceBeltramiOperator
    """
    lb_op = discretize_laplace_beltrami(mesh, gamma=gamma,
                                        n_eigenvalues=n_eigenvalues)

    scatter_list = compute_scattering_data(cauchy_data)
    n_data = len(scatter_list)

    zeta_sum = np.zeros(mesh.n_points)
    for ell in range(n_data):
        zeta_ell = compute_dsm_numerator(
            mesh, scatter_list[ell], lb_op, sigma_bg=sigma_bg
        )
        zeta_sum += np.abs(zeta_ell)

    zeta_sum /= n_data

    grid_points, grid_x, grid_y, mask = generate_sampling_grid(n_grid)

    numerator_grid = _interpolate_p1_to_grid(mesh, zeta_sum, grid_points)

    if denom_method == 'integral':
        denom = compute_dsm_denominator_integral(mesh, grid_points)
    elif denom_method == 'distance':
        denom = compute_dsm_denominator_distance(mesh, grid_points, gamma=gamma)
    else:
        raise ValueError("denom_method must be 'integral' or 'distance'")

    indicator_values = np.abs(numerator_grid) / (denom + 1e-30)

    indicator_2d = np.full((n_grid, n_grid), np.nan)
    indicator_2d[mask] = indicator_values

    return {
        'indicator': indicator_2d,
        'grid_x': grid_x,
        'grid_y': grid_y,
        'mask': mask,
        'zeta_sum': zeta_sum,
        'lb_operator': lb_op,
    }


# ============================================================
# P1 插值到网格点
# ============================================================

def _interpolate_p1_to_grid(mesh, values, grid_points):
    """通过重心坐标将 P1 FEM 解插值到任意网格点。"""
    K = len(grid_points)
    interpolated = np.zeros(K)

    p = mesh.points
    t = mesh.triangles
    centroids = mesh.centroids

    from scipy.spatial import cKDTree
    tree = cKDTree(centroids)

    for k in range(K):
        xq = grid_points[k]
        _, tri_indices = tree.query(xq, k=5)
        if np.isscalar(tri_indices):
            tri_indices = [tri_indices]

        found = False
        for tri_idx in tri_indices:
            v0 = p[t[tri_idx, 0]]
            v1 = p[t[tri_idx, 1]]
            v2 = p[t[tri_idx, 2]]

            d00 = v1[0] - v0[0]
            d01 = v2[0] - v0[0]
            d10 = v1[1] - v0[1]
            d11 = v2[1] - v0[1]

            det = d00 * d11 - d01 * d10
            if abs(det) < 1e-20:
                continue

            lam1 = ((xq[0] - v0[0]) * d11 - (xq[1] - v0[1]) * d01) / det
            lam2 = ((xq[1] - v0[1]) * d00 - (xq[0] - v0[0]) * d10) / det
            lam0 = 1.0 - lam1 - lam2

            tol = -1e-8
            if lam0 >= tol and lam1 >= tol and lam2 >= tol:
                interpolated[k] = (
                    lam0 * values[t[tri_idx, 0]]
                    + lam1 * values[t[tri_idx, 1]]
                    + lam2 * values[t[tri_idx, 2]]
                )
                found = True
                break

        if not found:
            tri_idx = tri_indices[0] if not np.isscalar(tri_indices) else tri_indices
            v0 = p[t[tri_idx, 0]]
            v1 = p[t[tri_idx, 1]]
            v2 = p[t[tri_idx, 2]]
            d00 = v1[0] - v0[0]
            d01 = v2[0] - v0[0]
            d10 = v1[1] - v0[1]
            d11 = v2[1] - v0[1]
            det = d00 * d11 - d01 * d10
            if abs(det) > 1e-20:
                lam1 = ((xq[0] - v0[0]) * d11 - (xq[1] - v0[1]) * d01) / det
                lam2 = ((xq[1] - v0[1]) * d00 - (xq[0] - v0[0]) * d10) / det
                lam0 = 1.0 - lam1 - lam2
                lam0 = max(0, min(1, lam0))
                lam1 = max(0, min(1, lam1))
                lam2 = max(0, min(1, lam2))
                s = lam0 + lam1 + lam2
                if s > 0:
                    lam0 /= s
                    lam1 /= s
                    lam2 /= s
                interpolated[k] = (
                    lam0 * values[t[tri_idx, 0]]
                    + lam1 * values[t[tri_idx, 1]]
                    + lam2 * values[t[tri_idx, 2]]
                )

    return interpolated


# ============================================================
# 可视化辅助
# ============================================================

def plot_dsm_indicator(result, title='DSM Indicator $\\eta(x)$',
                       figsize=(8, 6), cmap='hot', save_path=None,
                       inclusion_boxes=None, vmin=None, vmax=None):
    """可视化 DSM 指标函数。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    indicator = result['indicator']
    gx = result['grid_x']
    gy = result['grid_y']

    if vmin is None:
        vmin = np.nanmin(indicator)
    if vmax is None:
        vmax = np.nanmax(indicator)

    im = ax.imshow(
        indicator, origin='lower', cmap=cmap,
        extent=[gx[0], gx[-1], gy[0], gy[-1]],
        vmin=vmin, vmax=vmax, aspect='equal'
    )
    plt.colorbar(im, ax=ax, label='$\\eta(x)$')

    if inclusion_boxes:
        for box in inclusion_boxes:
            cx, cy = box['center']
            hw = box['half_width']
            color = box.get('color', 'w')
            rect = plt.Rectangle(
                (cx - hw, cy - hw), 2 * hw, 2 * hw,
                linewidth=2, edgecolor=color, facecolor='none',
                linestyle='--'
            )
            ax.add_patch(rect)

    ax.set_title(title)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig
