"""
fem.py - P1 Finite Element Core Components

Implements the finite element discretization from Ito et al. (2025):
  Weak form: ∫_Ω σ∇y·∇v dx = ∫_Γ f·v ds   (EIT, Eq. 2.2)
  Constraint: ∫_Γ y ds = 0                    (uniqueness)

For P1 (linear) triangular elements:
  - Stiffness matrix K: K_{ij} = ∫_Ω σ ∇φ_i · ∇φ_j dx
  - Mass matrix M: M_{ij} = ∫_Ω u φ_i φ_j dx
  - Boundary mass matrix M_Γ: (M_Γ)_{ij} = ∫_Γ φ_i φ_j ds
  - Boundary load vector b: b_i = ∫_Γ f φ_i ds

Reference: FreeFEM Example1.edp assembly routines.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def assemble_stiffness_matrix(mesh, sigma):
    """Assemble the P1 stiffness matrix K.

    K_{ij} = ∫_Ω σ(x) ∇φ_i · ∇φ_j dx

    For each triangle T_e, the local stiffness matrix is:
      K_e^{ij} = σ_e * (∇φ_i · ∇φ_j) * |T_e|

    where σ_e is the value of σ at the triangle centroid (P0 approximation).

    Parameters
    ----------
    mesh : EllipticMesh
    sigma : array (M,) or scalar
        Conductivity per triangle (P0 representation), or uniform scalar.

    Returns
    -------
    K : sparse matrix (N, N), CSR format
    """
    n = mesh.n_points
    t = mesh.triangles
    areas = mesh.areas
    gp = mesh.grad_phi  # (M, 3, 2)

    if np.isscalar(sigma):
        sigma = np.full(mesh.n_triangles, sigma)
    sigma = np.asarray(sigma, dtype=np.float64)

    rows = []
    cols = []
    vals = []

    for i in range(3):
        for j in range(3):
            dot_ij = gp[:, i, 0] * gp[:, j, 0] + gp[:, i, 1] * gp[:, j, 1]
            local_val = sigma * areas * dot_ij

            rows.append(t[:, i])
            cols.append(t[:, j])
            vals.append(local_val)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)

    K = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return K


def assemble_mass_matrix(mesh, coeff=None):
    """Assemble the P1 mass matrix M.

    M_{ij} = ∫_Ω u(x) φ_i φ_j dx

    For each triangle T_e, the exact P1 mass matrix is:
      M_e^{ij} = u_e * |T_e| * (1/12 if i≠j, 1/6 if i==j)
    i.e., M_e = u_e * |T_e| / 12 * [[2,1,1],[1,2,1],[1,1,2]]

    Parameters
    ----------
    mesh : EllipticMesh
    coeff : array (M,) or scalar or None
        Coefficient per triangle (P0), default 1.

    Returns
    -------
    M : sparse matrix (N, N)
    """
    n = mesh.n_points
    t = mesh.triangles
    areas = mesh.areas

    if coeff is None:
        coeff = np.ones(mesh.n_triangles)
    elif np.isscalar(coeff):
        coeff = np.full(mesh.n_triangles, coeff)
    coeff = np.asarray(coeff, dtype=np.float64)

    local_mass = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]], dtype=np.float64) / 12.0

    rows = []
    cols = []
    vals = []

    for i in range(3):
        for j in range(3):
            local_val = coeff * areas * local_mass[i, j]
            rows.append(t[:, i])
            cols.append(t[:, j])
            vals.append(local_val)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)

    M = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return M


def assemble_boundary_mass_matrix(mesh):
    """Assemble the boundary mass matrix M_Γ.

    (M_Γ)_{ij} = ∫_Γ φ_i φ_j ds

    For each boundary edge e = (n0, n1) of length L_e:
      (M_Γ)_{n0,n0} += L_e / 3
      (M_Γ)_{n0,n1} += L_e / 6
      (M_Γ)_{n1,n0} += L_e / 6
      (M_Γ)_{n1,n1} += L_e / 3

    Returns
    -------
    M_bdry : sparse matrix (N, N)
    """
    n = mesh.n_points
    edges = mesh.boundary_edges
    lengths = mesh.boundary_edge_lengths()

    rows = []
    cols = []
    vals = []

    for k in range(len(edges)):
        n0, n1 = edges[k]
        L = lengths[k]

        rows.extend([n0, n0, n1, n1])
        cols.extend([n0, n1, n0, n1])
        vals.extend([L / 3, L / 6, L / 6, L / 3])

    M_bdry = sparse.csr_matrix(
        (np.array(vals), (np.array(rows), np.array(cols))), shape=(n, n)
    )
    return M_bdry


def assemble_boundary_load(mesh, f_func):
    """Assemble the boundary load vector.

    b_i = ∫_Γ f(x) φ_i(x) ds

    Uses the 1D P1 mass matrix times nodal f values (trapezoidal rule):
      b_{n0} += L_e * [2*f(x_n0) + f(x_n1)] / 6
      b_{n1} += L_e * [f(x_n0) + 2*f(x_n1)] / 6

    Parameters
    ----------
    mesh : EllipticMesh
    f_func : callable
        Boundary source function f(x, y) -> scalar.

    Returns
    -------
    b : array (N,)
    """
    n = mesh.n_points
    edges = mesh.boundary_edges
    lengths = mesh.boundary_edge_lengths()
    p = mesh.points

    b = np.zeros(n)
    for k in range(len(edges)):
        n0, n1 = edges[k]
        x0, y0 = p[n0]
        x1, y1 = p[n1]
        L = lengths[k]

        f0 = f_func(x0, y0)
        f1 = f_func(x1, y1)

        b[n0] += L * (2 * f0 + f1) / 6
        b[n1] += L * (f0 + 2 * f1) / 6

    return b


def assemble_boundary_mean_constraint(mesh):
    """Assemble the boundary mean constraint vector B.

    Constraint: ∫_Γ y ds = 0, i.e., B^T y = 0
    where B_i = ∫_Γ φ_i ds.

    For each boundary edge e = (n0, n1) of length L_e:
      B_{n0} += L_e / 2
      B_{n1} += L_e / 2

    Returns
    -------
    B : array (N,)
    """
    n = mesh.n_points
    edges = mesh.boundary_edges
    lengths = mesh.boundary_edge_lengths()

    B = np.zeros(n)
    for k in range(len(edges)):
        n0, n1 = edges[k]
        L = lengths[k]
        B[n0] += L / 2
        B[n1] += L / 2

    return B


def solve_neumann_system(K, b, B):
    """Solve the Neumann problem via saddle-point system with Lagrange multiplier.

    [[K,  B],   [y]   [b]
     [B^T, 0]] * [λ] = [0]

    Corresponds to FreeFEM:
      matrix AA = [[A,B],[B',0]];
      xx = AA^-1 * bb;

    Parameters
    ----------
    K : sparse matrix (N, N) — stiffness matrix
    b : array (N,) — load vector
    B : array (N,) — constraint vector

    Returns
    -------
    y : array (N,) — solution satisfying ∫_Γ y ds = 0
    """
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


def solve_robin_system(mesh, A_op, alpha, v):
    """Solve the Robin boundary value problem (for the regularized DtN map).

    Paper 1, Eq. (3.20):
      -Δz = 0  in Ω,   z + α ∂z/∂n = v  on Γ

    Weak form:
      ∫_Ω ∇z·∇w dx + (1/α) ∫_Γ z·w ds = (1/α) ∫_Γ v·w ds

    Parameters
    ----------
    mesh : EllipticMesh
    A_op : sparse matrix (N, N) — interior operator (stiffness + optional mass)
    alpha : float — regularization parameter
    v : array (N,) — boundary data (full-domain vector, only boundary nodes matter)

    Returns
    -------
    z : array (N,)
    """
    M_bdry = assemble_boundary_mass_matrix(mesh)

    system_matrix = A_op + (1.0 / alpha) * M_bdry
    rhs = (1.0 / alpha) * M_bdry.dot(v)

    z = spsolve(system_matrix, rhs)
    return z


def compute_boundary_normal_flux(mesh, sigma, y):
    """Compute the boundary normal flux σ ∂y/∂n.

    For each boundary edge, the flux is computed from the gradient in the
    adjacent triangle.

    Parameters
    ----------
    mesh : EllipticMesh
    sigma : array (M,) — conductivity (P0)
    y : array (N,) — FEM solution

    Returns
    -------
    flux : array (N,) — normal flux at boundary nodes
    """
    edge_to_tri = {}
    for tri_idx, tri in enumerate(mesh.triangles):
        for i in range(3):
            e = tuple(sorted([tri[i], tri[(i + 1) % 3]]))
            edge_to_tri.setdefault(e, []).append(tri_idx)

    p = mesh.points
    n_pts = mesh.n_points
    flux = np.zeros(n_pts)
    weight = np.zeros(n_pts)

    for edge in mesh.boundary_edges:
        e_key = tuple(sorted([edge[0], edge[1]]))
        tri_list = edge_to_tri.get(e_key, [])
        if not tri_list:
            continue

        tri_idx = tri_list[0]
        tri = mesh.triangles[tri_idx]

        grad_y = np.zeros(2)
        for i in range(3):
            grad_y += y[tri[i]] * mesh.grad_phi[tri_idx, i, :]

        n0, n1 = edge[0], edge[1]
        dx = p[n1, 0] - p[n0, 0]
        dy = p[n1, 1] - p[n0, 1]
        length = np.sqrt(dx ** 2 + dy ** 2)
        normal = np.array([dy, -dx]) / length

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
# Partial boundary mass matrix (Paper 3, partial data)
# ============================================================

def assemble_partial_boundary_mass_matrix(mesh, gamma_d_node_mask):
    """Split the boundary mass matrix into Γ_D and Γ_N parts.

    Used for the heterogeneously regularized DtN map Λ_{α,D}(A) in Paper 3:
      α_D = α_d · χ_{Γ_D} + α_n · χ_{Γ_N}

    Edge classification rule:
      - If both endpoints are in Γ_D → contributes to M_bdry_D
      - Otherwise → contributes to M_bdry_N

    Parameters
    ----------
    mesh : EllipticMesh
    gamma_d_node_mask : array (N,), bool
        True if node belongs to the accessible boundary Γ_D.

    Returns
    -------
    M_bdry_D : sparse matrix (N, N) — boundary mass on Γ_D edges only
    M_bdry_N : sparse matrix (N, N) — boundary mass on Γ_N edges only
    """
    n = mesh.n_points
    rows_D, cols_D, vals_D = [], [], []
    rows_N, cols_N, vals_N = [], [], []

    p = mesh.points
    for edge in mesh.boundary_edges:
        n0, n1 = edge[0], edge[1]
        length = np.sqrt((p[n1, 0] - p[n0, 0])**2 + (p[n1, 1] - p[n0, 1])**2)

        diag_val = length / 3.0
        off_val = length / 6.0

        if gamma_d_node_mask[n0] and gamma_d_node_mask[n1]:
            rows_D.extend([n0, n1, n0, n1])
            cols_D.extend([n0, n1, n1, n0])
            vals_D.extend([diag_val, diag_val, off_val, off_val])
        else:
            rows_N.extend([n0, n1, n0, n1])
            cols_N.extend([n0, n1, n1, n0])
            vals_N.extend([diag_val, diag_val, off_val, off_val])

    M_bdry_D = sparse.csr_matrix(
        (vals_D, (rows_D, cols_D)), shape=(n, n)
    ) if vals_D else sparse.csr_matrix((n, n))

    M_bdry_N = sparse.csr_matrix(
        (vals_N, (rows_N, cols_N)), shape=(n, n)
    ) if vals_N else sparse.csr_matrix((n, n))

    return M_bdry_D, M_bdry_N


# ============================================================
# 通用边界法向导数（不依赖特定几何形状）
# ============================================================

def compute_boundary_normal_derivative(mesh, z, sigma_bg=1.0):
    """计算 σ₀ ∂z/∂n（通用版本，适用于任意 2D 域）。

    对每个边界边，从相邻三角形获取 P1 梯度 ∇z，
    用边的几何外法向 n̂ 计算 σ₀ (∇z · n̂)。

    Parameters
    ----------
    mesh : EllipticMesh
    z : array (N,) — P1 FEM 解
    sigma_bg : float — 背景电导率 σ₀

    Returns
    -------
    flux : array (N,) — σ₀ ∂z/∂n，边界节点有值，内部节点为零
    """
    edge_to_tri = {}
    for tri_idx in range(mesh.n_triangles):
        tri = mesh.triangles[tri_idx]
        for i in range(3):
            e = tuple(sorted([int(tri[i]), int(tri[(i + 1) % 3])]))
            edge_to_tri.setdefault(e, []).append(tri_idx)

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
        vals = [sigma_bg * np.dot(g, n) for g, n in node_grad[n_idx]]
        flux[n_idx] = np.mean(vals)

    return flux
