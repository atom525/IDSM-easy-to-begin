"""
fem_skfem.py — P1 finite element core components based on scikit-fem (skfem)

Uses skfem library for all FEM assembly and solving, replacing the manual implementation.
Maintains identical function signatures and return types as fem_legacy.py.

Dependency: scikit-fem >= 9.0
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
# Internal tools: build skfem objects from EllipticMesh
# ============================================================

def _build_skfem_mesh(mesh):
    """Build skfem.MeshTri from EllipticMesh.

    skfem conventions:
      - points: shape (2, N), i.e., transpose of coordinate matrix
      - triangles: shape (3, M), i.e., transpose of connectivity matrix, dtype=int32
    """
    p = np.ascontiguousarray(mesh.points.T, dtype=np.float64)  # (2, N)
    t = np.ascontiguousarray(mesh.triangles.T, dtype=np.int32)  # (3, M)
    return MeshTri(p, t)


def _build_basis(mesh):
    """Build P1 interior basis function object."""
    skfem_mesh = _build_skfem_mesh(mesh)
    return Basis(skfem_mesh, ElementTriP1())


def _build_facet_basis(mesh, facets=None):
    """Build boundary facet basis function object.

    Parameters
    ----------
    mesh : EllipticMesh
    facets : array or None
        Specified boundary facet indices. None means all boundary facets.
    """
    skfem_mesh = _build_skfem_mesh(mesh)
    if facets is not None:
        return FacetBasis(skfem_mesh, ElementTriP1(), facets=facets)
    return FacetBasis(skfem_mesh, ElementTriP1())


# ============================================================
# Stiffness matrix
# ============================================================

def assemble_stiffness_matrix(mesh, sigma):
    """Assemble P1 stiffness matrix K.

    K_{ij} = int_Omega sigma(x) grad(phi_i) . grad(phi_j) dx

    Parameters
    ----------
    mesh : EllipticMesh
    sigma : array (M,) or scalar
        Conductivity per triangle (P0 representation), or uniform scalar.

    Returns
    -------
    K : scipy.sparse.csr_matrix (N, N)
    """
    basis = _build_basis(mesh)

    if np.isscalar(sigma):
        sigma = np.full(mesh.n_triangles, float(sigma))
    sigma = np.asarray(sigma, dtype=np.float64)

    # Project P0 coefficients to skfem quadrature points
    # basis.dx already includes Jacobian, so just provide coefficient in form
    @BilinearForm
    def stiffness(u, v, w):
        # w.x[0], w.x[1] are quadrature point coordinates
        # Need P0 values: one constant per element
        return w["sigma"] * dot(grad(u), grad(v))

    # Interpolate P0 sigma onto ElementTriP0
    skfem_mesh = _build_skfem_mesh(mesh)
    p0_basis = Basis(skfem_mesh, ElementTriP0())
    sigma_proj = p0_basis.zeros()
    sigma_proj[:] = sigma

    K = asm(stiffness, basis, sigma=p0_basis.interpolate(sigma_proj))
    return K.tocsr()


# ============================================================
# Mass matrix
# ============================================================

def assemble_mass_matrix(mesh, coeff=None):
    """Assemble P1 mass matrix M.

    M_{ij} = int_Omega c(x) phi_i phi_j dx

    Parameters
    ----------
    mesh : EllipticMesh
    coeff : array (M,) or scalar or None
        Coefficient per triangle (P0), default 1.

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
# Boundary mass matrix
# ============================================================

def assemble_boundary_mass_matrix(mesh):
    """Assemble boundary mass matrix M_Gamma.

    (M_Gamma)_{ij} = int_Gamma phi_i phi_j ds

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
# Boundary load vector
# ============================================================

def assemble_boundary_load(mesh, f_func):
    """Assemble boundary load vector.

    b_i = int_Gamma f(x) phi_i(x) ds

    Parameters
    ----------
    mesh : EllipticMesh
    f_func : callable
        Boundary source function f(x, y) -> scalar.

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
# Boundary mean constraint
# ============================================================

def assemble_boundary_mean_constraint(mesh):
    """Assemble boundary mean constraint vector B.

    Constraint: int_Gamma y ds = 0, i.e., B^T y = 0
    where B_i = int_Gamma phi_i ds

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
# Neumann solver
# ============================================================

def solve_neumann_system(K, b, B):
    """Solve Neumann problem via saddle-point system (Lagrange multiplier method).

    [[K,  B],   [y]   [b]
     [B^T, 0]] * [λ] = [0]

    Corresponds to FreeFEM:
      matrix AA = [[A,B],[B',0]];
      xx = AA^-1 * bb;

    Parameters
    ----------
    K : sparse matrix (N, N) -- stiffness matrix
    b : array (N,) -- load vector
    B : array (N,) -- constraint vector

    Returns
    -------
    y : array (N,) -- solution satisfying int_Gamma y ds = 0
    """
    # Note: this function only does linear algebra, doesn't depend on FEM library
    # Keeps identical implementation to legacy version
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
# Robin solver
# ============================================================

def solve_robin_system(mesh, A_op, alpha, v):
    """Solve Robin boundary value problem (regularized DtN map).

    Paper 1, Eq. (3.20):
      -Delta z = 0 in Omega,   z + alpha dz/dn = v on Gamma

    Weak form:
      int_Omega grad(z).grad(w) dx + (1/alpha) int_Gamma z.w ds = (1/alpha) int_Gamma v.w ds

    Parameters
    ----------
    mesh : EllipticMesh
    A_op : sparse matrix (N, N) -- interior operator (stiffness + optional mass)
    alpha : float -- regularization parameter
    v : array (N,) -- boundary data (full-domain vector, only boundary nodes effective)

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
# Boundary normal flux
# ============================================================

def compute_boundary_normal_flux(mesh, sigma, y):
    """Compute boundary normal flux sigma dy/dn.

    For each boundary edge, computes P1 gradient from the adjacent triangle,
    then takes dot product with the geometric outward normal.

    Parameters
    ----------
    mesh : EllipticMesh
    sigma : array (M,) -- conductivity (P0)
    y : array (N,) -- FEM solution

    Returns
    -------
    flux : array (N,) -- normal flux at boundary nodes
    """
    # Build edge -> triangle mapping
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

        # P1 gradient: grad(y)|_T = sum_i y[tri[i]] * grad_phi[tri_idx, i, :]
        grad_y = np.zeros(2)
        for i in range(3):
            grad_y += y[tri[i]] * mesh.grad_phi[tri_idx, i, :]

        n0, n1 = int(edge[0]), int(edge[1])
        dx = p[n1, 0] - p[n0, 0]
        dy = p[n1, 1] - p[n0, 1]
        length = np.sqrt(dx ** 2 + dy ** 2)
        # Outward normal: rotate edge vector by 90 degrees
        normal = np.array([dy, -dx]) / length

        # Ensure normal points outward (away from triangle centroid)
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
# General boundary normal derivative (geometry-independent)
# ============================================================

def compute_boundary_normal_derivative(mesh, z, sigma_bg=1.0):
    """Compute sigma_0 dz/dn (general version, works for any 2D domain).

    For each boundary edge, gets P1 gradient grad(z) from the adjacent triangle,
    computes sigma_0 (grad(z) . n_hat) using the geometric outward normal.

    This is a general replacement for compute_ellipse_normal_derivative,
    without depending on elliptic geometry n_hat = (x1/a^2, x2/b^2)/||.||.

    Parameters
    ----------
    mesh : EllipticMesh
    z : array (N,) -- P1 FEM solution
    sigma_bg : float -- background conductivity sigma_0

    Returns
    -------
    flux : array (N,) -- sigma_0 dz/dn, nonzero at boundary nodes, zero at interior nodes
    """
    # Build edge -> triangle mapping
    edge_to_tri = {}
    for tri_idx in range(mesh.n_triangles):
        tri = mesh.triangles[tri_idx]
        for i in range(3):
            e = tuple(sorted([int(tri[i]), int(tri[(i + 1) % 3])]))
            edge_to_tri.setdefault(e, []).append(tri_idx)

    # Boundary nodes: collect grad(z) from adjacent boundary triangles
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
        # Edge outward normal
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
        # Average grad(z).n_hat over all adjacent boundary edges
        vals = [sigma_bg * np.dot(g, n) for g, n in node_grad[n_idx]]
        flux[n_idx] = np.mean(vals)

    return flux


# ============================================================
# Partial boundary mass matrix (Paper 3, partial data)
# ============================================================

def assemble_partial_boundary_mass_matrix(mesh, gamma_d_node_mask):
    """Split boundary mass matrix into Gamma_D and Gamma_N parts.

    Used for the heterogeneous regularized DtN map Lambda_{alpha,D}(A) in Paper 3:
      alpha_D = alpha_d * chi_{Gamma_D} + alpha_n * chi_{Gamma_N}

    Edge classification rule:
      - Both endpoints in Gamma_D -> contributes to M_bdry_D
      - Otherwise -> contributes to M_bdry_N

    Parameters
    ----------
    mesh : EllipticMesh
    gamma_d_node_mask : array (N,), bool
        True if node belongs to the accessible boundary Gamma_D.

    Returns
    -------
    M_bdry_D : scipy.sparse.csr_matrix (N, N) -- boundary mass on Gamma_D edges
    M_bdry_N : scipy.sparse.csr_matrix (N, N) -- boundary mass on Gamma_N edges
    """
    # Classify boundary edges into D and N categories
    skfem_mesh = _build_skfem_mesh(mesh)
    n = mesh.n_points
    p = mesh.points

    facets_d = []
    facets_n = []

    # skfem_mesh.facets is (2, E_total) format for all edges
    # We need to find boundary facets and classify by D/N
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

    # Assemble Gamma_D part
    if facets_d:
        fb_d = FacetBasis(skfem_mesh, ElementTriP1(),
                          facets=np.array(facets_d, dtype=np.int64))
        M_bdry_D = asm(bdry_mass, fb_d).tocsr()
    else:
        M_bdry_D = sparse.csr_matrix((n, n))

    # Assemble Gamma_N part
    if facets_n:
        fb_n = FacetBasis(skfem_mesh, ElementTriP1(),
                          facets=np.array(facets_n, dtype=np.int64))
        M_bdry_N = asm(bdry_mass, fb_n).tocsr()
    else:
        M_bdry_N = sparse.csr_matrix((n, n))

    return M_bdry_D, M_bdry_N
