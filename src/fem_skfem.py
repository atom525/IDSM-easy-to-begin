"""
fem_skfem.py — 基于 scikit-fem (skfem) 的 P1 有限元核心组件

使用 skfem 库实现所有 FEM 装配和求解，替代原手写版本。
对外保持与 fem_legacy.py 完全相同的函数签名和返回类型。

依赖：scikit-fem >= 9.0
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from skfem import (
    MeshTri,
    Basis,
    FacetBasis,
    ElementTriP1,
    ElementTriP0,
    BilinearForm,
    LinearForm,
    Functional,
    asm,
)
from skfem.helpers import dot, grad


# ============================================================
# 内部工具：从 EllipticMesh 构建 skfem 对象
# ============================================================

def _build_skfem_mesh(mesh):
    """从 EllipticMesh 构建 skfem.MeshTri。

    skfem 约定：
      - points: shape (2, N)，即坐标矩阵的转置
      - triangles: shape (3, M)，即连接矩阵的转置，dtype=int32
    """
    p = np.ascontiguousarray(mesh.points.T, dtype=np.float64)  # (2, N)
    t = np.ascontiguousarray(mesh.triangles.T, dtype=np.int32)  # (3, M)
    return MeshTri(p, t)


def _build_basis(mesh):
    """构建 P1 内部基函数对象。"""
    skfem_mesh = _build_skfem_mesh(mesh)
    return Basis(skfem_mesh, ElementTriP1())


def _build_facet_basis(mesh, facets=None):
    """构建边界 facet 基函数对象。

    Parameters
    ----------
    mesh : EllipticMesh
    facets : array or None
        指定的边界 facet 索引。None 表示全部边界 facets。
    """
    skfem_mesh = _build_skfem_mesh(mesh)
    if facets is not None:
        return FacetBasis(skfem_mesh, ElementTriP1(), facets=facets)
    return FacetBasis(skfem_mesh, ElementTriP1())


# ============================================================
# 刚度矩阵
# ============================================================

def assemble_stiffness_matrix(mesh, sigma):
    """装配 P1 刚度矩阵 K。

    K_{ij} = ∫_Ω σ(x) ∇φ_i · ∇φ_j dx

    Parameters
    ----------
    mesh : EllipticMesh
    sigma : array (M,) 或 scalar
        每个三角形上的电导率（P0 表示），或均匀标量。

    Returns
    -------
    K : scipy.sparse.csr_matrix (N, N)
    """
    basis = _build_basis(mesh)

    if np.isscalar(sigma):
        sigma = np.full(mesh.n_triangles, float(sigma))
    sigma = np.asarray(sigma, dtype=np.float64)

    # 将 P0 系数投影到 skfem 的 quadrature 点
    # basis.dx 已包含 Jacobian，所以只需在 form 中提供系数
    @BilinearForm
    def stiffness(u, v, w):
        # w.x[0], w.x[1] 是 quadrature 点坐标
        # 需要用 P0 值：每个 element 一个常数
        return w["sigma"] * dot(grad(u), grad(v))

    # 将 P0 sigma 插值到 ElementTriP0 上
    skfem_mesh = _build_skfem_mesh(mesh)
    p0_basis = Basis(skfem_mesh, ElementTriP0())
    sigma_proj = p0_basis.zeros()
    sigma_proj[:] = sigma

    K = asm(stiffness, basis, sigma=p0_basis.interpolate(sigma_proj))
    return K.tocsr()


# ============================================================
# 质量矩阵
# ============================================================

def assemble_mass_matrix(mesh, coeff=None):
    """装配 P1 质量矩阵 M。

    M_{ij} = ∫_Ω c(x) φ_i φ_j dx

    Parameters
    ----------
    mesh : EllipticMesh
    coeff : array (M,) 或 scalar 或 None
        每个三角形上的系数（P0），默认 1。

    Returns
    -------
    M : scipy.sparse.csr_matrix (N, N)
    """
    basis = _build_basis(mesh)

    if coeff is None:
        coeff_arr = np.ones(mesh.n_triangles)
    elif np.isscalar(coeff):
        coeff_arr = np.full(mesh.n_triangles, float(coeff))
    else:
        coeff_arr = np.asarray(coeff, dtype=np.float64)

    @BilinearForm
    def mass(u, v, w):
        return w["coeff"] * u * v

    skfem_mesh = _build_skfem_mesh(mesh)
    p0_basis = Basis(skfem_mesh, ElementTriP0())
    coeff_proj = p0_basis.zeros()
    coeff_proj[:] = coeff_arr

    M = asm(mass, basis, coeff=p0_basis.interpolate(coeff_proj))
    return M.tocsr()


# ============================================================
# 边界质量矩阵
# ============================================================

def assemble_boundary_mass_matrix(mesh):
    """装配边界质量矩阵 M_Γ。

    (M_Γ)_{ij} = ∫_Γ φ_i φ_j ds

    Returns
    -------
    M_bdry : scipy.sparse.csr_matrix (N, N)
    """
    fbasis = _build_facet_basis(mesh)

    @BilinearForm
    def bdry_mass(u, v, _):
        return u * v

    M_bdry = asm(bdry_mass, fbasis)
    return M_bdry.tocsr()


# ============================================================
# 边界载荷向量
# ============================================================

def assemble_boundary_load(mesh, f_func):
    """装配边界载荷向量。

    b_i = ∫_Γ f(x) φ_i(x) ds

    Parameters
    ----------
    mesh : EllipticMesh
    f_func : callable
        边界源函数 f(x, y) -> scalar。

    Returns
    -------
    b : np.ndarray (N,)
    """
    fbasis = _build_facet_basis(mesh)

    @LinearForm
    def bdry_load(v, w):
        f_vals = f_func(w.x[0], w.x[1])
        return f_vals * v

    b = asm(bdry_load, fbasis)
    return np.asarray(b, dtype=np.float64)


# ============================================================
# 边界均值约束
# ============================================================

def assemble_boundary_mean_constraint(mesh):
    """装配边界均值约束向量 B。

    约束：∫_Γ y ds = 0，即 B^T y = 0
    其中 B_i = ∫_Γ φ_i ds

    Returns
    -------
    B : np.ndarray (N,)
    """
    fbasis = _build_facet_basis(mesh)

    @LinearForm
    def bdry_ones(v, _):
        return 1.0 * v

    B = asm(bdry_ones, fbasis)
    return np.asarray(B, dtype=np.float64)


# ============================================================
# Neumann 求解器
# ============================================================

def solve_neumann_system(K, b, B):
    """通过鞍点系统（Lagrange 乘子法）求解 Neumann 问题。

    [[K,  B],   [y]   [b]
     [B^T, 0]] * [λ] = [0]

    对应 FreeFEM:
      matrix AA = [[A,B],[B',0]];
      xx = AA^-1 * bb;

    Parameters
    ----------
    K : sparse matrix (N, N) — 刚度矩阵
    b : array (N,) — 载荷向量
    B : array (N,) — 约束向量

    Returns
    -------
    y : array (N,) — 满足 ∫_Γ y ds = 0 的解
    """
    # 注意：此函数只做线性代数，不依赖 FEM 库
    # 保持与 legacy 版本完全一致的实现
    n = K.shape[0]

    B_col = sparse.csr_matrix(B.reshape(-1, 1))
    top = sparse.hstack([K, B_col])
    bottom = sparse.hstack([B_col.T, sparse.csr_matrix((1, 1))])
    saddle = sparse.vstack([top, bottom]).tocsr()

    rhs = np.zeros(n + 1)
    rhs[:n] = b

    solution = spsolve(saddle, rhs)
    y = solution[:n]
    return y


# ============================================================
# Robin 求解器
# ============================================================

def solve_robin_system(mesh, A_op, alpha, v):
    """求解 Robin 边值问题（正则化 DtN 映射）。

    Paper 1, Eq. (3.20):
      -Δz = 0 in Ω,   z + α ∂z/∂n = v on Γ

    弱形式：
      ∫_Ω ∇z·∇w dx + (1/α) ∫_Γ z·w ds = (1/α) ∫_Γ v·w ds

    Parameters
    ----------
    mesh : EllipticMesh
    A_op : sparse matrix (N, N) — 内部算子（刚度 + 可选质量）
    alpha : float — 正则化参数
    v : array (N,) — 边界数据（全域向量，仅边界节点有效）

    Returns
    -------
    z : array (N,)
    """
    M_bdry = assemble_boundary_mass_matrix(mesh)
    system_matrix = A_op + (1.0 / alpha) * M_bdry
    rhs = (1.0 / alpha) * M_bdry.dot(v)
    z = spsolve(system_matrix, rhs)
    return z


# ============================================================
# 边界法向通量
# ============================================================

def compute_boundary_normal_flux(mesh, sigma, y):
    """计算边界法向通量 σ ∂y/∂n。

    对每个边界边，从相邻三角形计算 P1 梯度，
    然后与几何外法向取点积。

    Parameters
    ----------
    mesh : EllipticMesh
    sigma : array (M,) — 电导率（P0）
    y : array (N,) — FEM 解

    Returns
    -------
    flux : array (N,) — 边界节点上的法向通量
    """
    # 构建 edge → triangle 映射
    edge_to_tri = {}
    for tri_idx, tri in enumerate(mesh.triangles):
        for i in range(3):
            e = tuple(sorted([int(tri[i]), int(tri[(i + 1) % 3])]))
            edge_to_tri.setdefault(e, []).append(tri_idx)

    p = mesh.points
    n_pts = mesh.n_points
    flux = np.zeros(n_pts)
    weight = np.zeros(n_pts)

    for edge in mesh.boundary_edges:
        e_key = tuple(sorted([int(edge[0]), int(edge[1])]))
        tri_list = edge_to_tri.get(e_key, [])
        if not tri_list:
            continue

        tri_idx = tri_list[0]
        tri = mesh.triangles[tri_idx]

        # P1 梯度 ∇y|_T = Σ_i y[tri[i]] * grad_phi[tri_idx, i, :]
        grad_y = np.zeros(2)
        for i in range(3):
            grad_y += y[tri[i]] * mesh.grad_phi[tri_idx, i, :]

        n0, n1 = int(edge[0]), int(edge[1])
        dx = p[n1, 0] - p[n0, 0]
        dy = p[n1, 1] - p[n0, 1]
        length = np.sqrt(dx ** 2 + dy ** 2)
        # 外法向：边向量旋转 90°
        normal = np.array([dy, -dx]) / length

        # 确保法向朝外（指向远离三角形质心的方向）
        mid = 0.5 * (p[n0] + p[n1])
        centroid = mesh.centroids[tri_idx]
        if np.dot(normal, mid - centroid) < 0:
            normal = -normal

        sigma_val = sigma[tri_idx] if not np.isscalar(sigma) else sigma
        flux_val = sigma_val * np.dot(grad_y, normal)

        flux[n0] += flux_val * length / 2
        flux[n1] += flux_val * length / 2
        weight[n0] += length / 2
        weight[n1] += length / 2

    valid = weight > 0
    flux[valid] /= weight[valid]

    return flux


# ============================================================
# 通用边界法向导数（不依赖特定几何形状）
# ============================================================

def compute_boundary_normal_derivative(mesh, z, sigma_bg=1.0):
    """计算 σ₀ ∂z/∂n（通用版本，适用于任意 2D 域）。

    对每个边界边，从相邻三角形获取 P1 梯度 ∇z，
    用边的几何外法向 n̂ 计算 σ₀ (∇z · n̂)。

    这是 compute_ellipse_normal_derivative 的通用替代版本，
    不依赖椭圆几何 n̂ = (x₁/a², x₂/b²)/‖·‖。

    Parameters
    ----------
    mesh : EllipticMesh
    z : array (N,) — P1 FEM 解
    sigma_bg : float — 背景电导率 σ₀

    Returns
    -------
    flux : array (N,) — σ₀ ∂z/∂n，边界节点有值，内部节点为零
    """
    # 构建 edge → triangle 映射
    edge_to_tri = {}
    for tri_idx in range(mesh.n_triangles):
        tri = mesh.triangles[tri_idx]
        for i in range(3):
            e = tuple(sorted([int(tri[i]), int(tri[(i + 1) % 3])]))
            edge_to_tri.setdefault(e, []).append(tri_idx)

    # 边界节点：收集相邻边界三角形上的 ∇z
    node_grad = {}
    for edge in mesh.boundary_edges:
        e_key = tuple(sorted([int(edge[0]), int(edge[1])]))
        tri_list = edge_to_tri.get(e_key, [])
        if not tri_list:
            continue
        tri_idx = tri_list[0]
        tri = mesh.triangles[tri_idx]

        grad_z = np.zeros(2)
        for i in range(3):
            grad_z += z[tri[i]] * mesh.grad_phi[tri_idx, i, :]

        n0, n1 = int(edge[0]), int(edge[1])
        # 边的外法向
        dx = mesh.points[n1, 0] - mesh.points[n0, 0]
        dy = mesh.points[n1, 1] - mesh.points[n0, 1]
        length = np.sqrt(dx ** 2 + dy ** 2)
        normal = np.array([dy, -dx]) / length

        mid = 0.5 * (mesh.points[n0] + mesh.points[n1])
        centroid = mesh.centroids[tri_idx]
        if np.dot(normal, mid - centroid) < 0:
            normal = -normal

        for n_idx in [n0, n1]:
            node_grad.setdefault(n_idx, []).append(
                (grad_z, normal)
            )

    flux = np.zeros(mesh.n_points)
    for n_idx in mesh.boundary_nodes:
        n_idx = int(n_idx)
        if n_idx not in node_grad:
            continue
        # 取所有相邻边界边上的 ∇z·n̂ 的平均
        vals = [sigma_bg * np.dot(g, n) for g, n in node_grad[n_idx]]
        flux[n_idx] = np.mean(vals)

    return flux


# ============================================================
# 部分边界质量矩阵（Paper 3，部分数据）
# ============================================================

def assemble_partial_boundary_mass_matrix(mesh, gamma_d_node_mask):
    """将边界质量矩阵拆分为 Γ_D 和 Γ_N 两部分。

    用于 Paper 3 的异质正则化 DtN 映射 Λ_{α,D}(A)：
      α_D = α_d · χ_{Γ_D} + α_n · χ_{Γ_N}

    边分类规则：
      - 两端点都在 Γ_D → 贡献到 M_bdry_D
      - 否则 → 贡献到 M_bdry_N

    Parameters
    ----------
    mesh : EllipticMesh
    gamma_d_node_mask : array (N,), bool
        True 表示节点属于可观测边界 Γ_D。

    Returns
    -------
    M_bdry_D : scipy.sparse.csr_matrix (N, N) — Γ_D 边上的边界质量
    M_bdry_N : scipy.sparse.csr_matrix (N, N) — Γ_N 边上的边界质量
    """
    # 将边界边分成 D 和 N 两类
    skfem_mesh = _build_skfem_mesh(mesh)
    n = mesh.n_points
    p = mesh.points

    facets_d = []
    facets_n = []

    # skfem_mesh.facets 是 (2, E_total) 格式的所有边
    # 我们需要找到边界 facets 并按 D/N 分类
    boundary_facets = skfem_mesh.boundary_facets()

    for fidx in boundary_facets:
        n0, n1 = skfem_mesh.facets[0, fidx], skfem_mesh.facets[1, fidx]
        if gamma_d_node_mask[n0] and gamma_d_node_mask[n1]:
            facets_d.append(fidx)
        else:
            facets_n.append(fidx)

    @BilinearForm
    def bdry_mass(u, v, _):
        return u * v

    # 装配 Γ_D 部分
    if facets_d:
        fb_d = FacetBasis(skfem_mesh, ElementTriP1(),
                          facets=np.array(facets_d, dtype=np.int64))
        M_bdry_D = asm(bdry_mass, fb_d).tocsr()
    else:
        M_bdry_D = sparse.csr_matrix((n, n))

    # 装配 Γ_N 部分
    if facets_n:
        fb_n = FacetBasis(skfem_mesh, ElementTriP1(),
                          facets=np.array(facets_n, dtype=np.int64))
        M_bdry_N = asm(bdry_mass, fb_n).tocsr()
    else:
        M_bdry_N = sparse.csr_matrix((n, n))

    return M_bdry_D, M_bdry_N
