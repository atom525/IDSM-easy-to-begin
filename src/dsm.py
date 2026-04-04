"""
dsm.py - Classical Direct Sampling Method (Standard DSM)

Implements the DSM strictly per Ito, Jin, Wang, Zou (2025) Section 2.2:
  Index function η(x) = ⟨G(·,x), y_d^s⟩_{H^γ(Γ)} / ‖G(·,x)‖_{H^γ(Γ)}    (Eq. 2.8)

where:
  - G(·,x) is Green's function (Neumann function)
  - y_d^s is the scattering data (background solution minus measured data)
  - H^γ(Γ) is the boundary Sobolev space induced by (−Δ_Γ)^γ

Numerator computation (Eq. 2.9): via solving an auxiliary Neumann problem
  −∇·(σ₀∇ζ) = 0 in Ω,  σ₀ ∂ζ/∂n = (−Δ_Γ)^γ y_d^s on Γ

Denominator approximation (Eq. 2.10): d(x,Γ)^γ or integral approximation

Reference: FreeFEM Example1.edp L252-264 (diagFunc) and L317-335 (iteration 0).
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
# Laplace-Beltrami operator discretization
# ============================================================

class LaplaceBeltramiOperator:
    """Discrete representation of (−Δ_Γ)^γ on the boundary Γ.

    Constructed via 1D finite element eigendecomposition:
      K_Γ v_i = λ_i M_Γ v_i

    where K_Γ is the 1D boundary stiffness matrix (arc-length derivative)
    and M_Γ is the 1D boundary mass matrix.

    Operator action: (−Δ_Γ)^γ f = Σ_{i: λ_i>0} λ_i^γ (f, v_i)_{M_Γ} v_i

    Expansion coefficients: c_i = v_i^T M_Γ f (since V^T M_Γ V = I)

    Attributes
    ----------
    eigenvalues : array (K,) — sorted eigenvalues λ_i
    eigenvectors : array (B, K) — M_Γ-orthonormal eigenvectors
    gamma : float — fractional power γ
    n_full : int — total number of domain nodes
    boundary_nodes : array — boundary node indices
    M_bdry_local : sparse (B, B) — local boundary mass matrix
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
        """Apply (−Δ_Γ)^γ to a full-domain vector (only boundary part is effective).

        Parameters
        ----------
        f_full : array (N,)

        Returns
        -------
        result : array (N,) — only boundary nodes are non-zero
        """
        f_bdry = f_full[self.boundary_nodes]
        result_bdry = self.apply_to_boundary(f_bdry)

        result = np.zeros(self.n_full)
        result[self.boundary_nodes] = result_bdry
        return result

    def apply_to_boundary(self, f_bdry):
        """Apply (−Δ_Γ)^γ directly to a boundary vector.

        c_i = (M_Γ v_i)^T f,  result = Σ_i λ_i^γ c_i v_i

        Parameters
        ----------
        f_bdry : array (B,)

        Returns
        -------
        result_bdry : array (B,)
        """
        coeffs = self._MV.T.dot(f_bdry)
        weighted = self._lam_gamma * coeffs
        result_bdry = self.eigenvectors.dot(weighted)
        return result_bdry


def discretize_laplace_beltrami(mesh, gamma=0.5, n_eigenvalues=None):
    """Discretize the Laplace-Beltrami operator (−Δ_Γ)^γ on the boundary Γ.

    Procedure:
      1. Assemble 1D boundary stiffness: (K_Γ)_{ij} = ∫_Γ (dφ_i/ds)(dφ_j/ds) ds
      2. Assemble 1D boundary mass: (M_Γ)_{ij} = ∫_Γ φ_i φ_j ds
      3. Solve generalized eigenproblem: K_Γ v = λ M_Γ v
      4. Construct (−Δ_Γ)^γ f = Σ_i λ_i^γ (f, v_i)_{M_Γ} v_i

    For each boundary edge e=(n0,n1) of length L_e:
      (K_Γ)_e = (1/L_e) [[1, −1], [−1, 1]]   (1D stiffness)
      (M_Γ)_e = (L_e/6) [[2, 1], [1, 2]]     (1D mass)

    Parameters
    ----------
    mesh : EllipticMesh
    gamma : float — fractional power (Paper 1 recommends γ=1/2, matching the DtN map)
    n_eigenvalues : int or None — number of eigenvalues to compute (default: all)

    Returns
    -------
    LaplaceBeltramiOperator

    Reference: Paper 1, Section 2.2 — (−Δ_Γ)^γ defines the H^γ(Γ) inner product.
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
# Scattering data
# ============================================================

def compute_scattering_data(cauchy_data):
    """Compute scattering data y_d^s = y_∅ − y_d.

    Paper 1, Section 2.1 (after Eq. 2.7):
      y_d^s = T·y_∅ − y_d
    For Neumann boundary conditions, T is the identity (trace) operator.

    Parameters
    ----------
    cauchy_data : dict from forward_solver.generate_cauchy_data

    Returns
    -------
    scatter : list of array (N,)
    """
    scatter = []
    for y_empty, y_data in zip(cauchy_data['y_empty'], cauchy_data['y_data']):
        scatter.append(y_empty - y_data)
    return scatter


# ============================================================
# DSM numerator: auxiliary PDE solve
# ============================================================

def compute_dsm_numerator(mesh, scatter_full, lb_operator, sigma_bg=1.0):
    """Compute the DSM numerator: ζ(x) = ⟨G(·,x), y_d^s⟩_{H^γ(Γ)}.

    Solves the auxiliary Neumann problem (Paper 1, Eq. 2.9):
      −∇·(σ₀∇ζ) = 0 in Ω,  σ₀ ∂ζ/∂n = (−Δ_Γ)^γ y_d^s on Γ
      Constraint: ∫_Γ ζ ds = 0

    The solution ζ(x) simultaneously gives the numerator at all interior points x.

    Parameters
    ----------
    mesh : EllipticMesh
    scatter_full : array (N,) — scattering data
    lb_operator : LaplaceBeltramiOperator
    sigma_bg : float — background conductivity

    Returns
    -------
    zeta : array (N,) — auxiliary problem solution = numerator ⟨G(·,x), y_d^s⟩_{H^γ(Γ)}
    """
    g_full = lb_operator.apply(scatter_full)

    M_bdry = assemble_boundary_mass_matrix(mesh)
    b = M_bdry.dot(g_full)

    K = assemble_stiffness_matrix(mesh, sigma_bg)
    B = assemble_boundary_mean_constraint(mesh)

    zeta = solve_neumann_system(K, b, B)
    return zeta


# ============================================================
# DSM denominator: normalization factor
# ============================================================

def compute_dsm_denominator_distance(mesh, points, gamma=0.5):
    """Denominator approximation method 1: distance-based.

    Paper 1, Eq. (2.10) for d=2:
      ‖G(·,x)‖_{H^γ(Γ)} ≈ C / d(x,Γ)^γ

    Parameters
    ----------
    mesh : EllipticMesh
    points : array (K, 2) — sampling points
    gamma : float

    Returns
    -------
    denom : array (K,) — denominator values
    """
    dist = distance_to_boundary(mesh, points)
    dist = np.maximum(dist, 1e-12)
    denom = 1.0 / (dist ** gamma)
    return denom


def compute_dsm_denominator_integral(mesh, points):
    """Denominator approximation method 2: integral-based (FreeFEM style).

    Reference: FreeFEM Example1.edp L260-261:
      diagFunc(i) = 1 / ((int1d(Th)(1/dis^2.0))^0.5)

    i.e., R(x) = 1 / √(∫_Γ 1/|x−x'|² ds(x'))

    This approximates 1/‖∇Φ_x‖_{L²(Γ)} where Φ_x = −1/(2π) ln|x−x'|.

    Parameters
    ----------
    mesh : EllipticMesh
    points : array (K, 2)

    Returns
    -------
    denom : array (K,) — denominator values (= 1/‖∇Φ_x‖_{L²(Γ)})
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
# DSM indicator function — main driver
# ============================================================

def compute_dsm_indicator(mesh, cauchy_data, gamma=0.5, n_grid=201,
                          denom_method='integral', sigma_bg=1.0,
                          n_eigenvalues=None):
    """Compute the DSM indicator function η(x) on a uniform grid.

    Paper 1, Eq. (2.8):
      η(x) = ⟨G(·,x), y_d^s⟩_{H^γ(Γ)} / ‖G(·,x)‖_{H^γ(Γ)}

    For L Cauchy data pairs, aggregated using absolute values:
      η(x) = (1/L) Σ_ℓ |ζ_ℓ(x)| / R(x)

    Procedure:
      1. Discretize (−Δ_Γ)^γ (eigendecomposition)
      2. Compute scattering data y_d^s
      3. For each data pair, solve auxiliary PDE → ζ_ℓ(x)
      4. Aggregate numerators on FEM mesh
      5. Interpolate to uniform scanning grid
      6. Compute denominator R(x)
      7. η(x) = numerator / denominator

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
# P1 interpolation to grid points
# ============================================================

def _interpolate_p1_to_grid(mesh, values, grid_points):
    """Interpolate P1 FEM solution to arbitrary grid points via barycentric coordinates."""
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
# Visualization helper
# ============================================================

def plot_dsm_indicator(result, title='DSM Indicator $\\eta(x)$',
                       figsize=(8, 6), cmap='hot', save_path=None,
                       inclusion_boxes=None, vmin=None, vmax=None):
    """Visualize the DSM indicator function."""
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
