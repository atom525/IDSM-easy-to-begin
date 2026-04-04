"""
mesh.py - Unstructured triangular mesh generation for the elliptic domain

Implements the domain setup per Ito et al. (2025) Section 4:
  Domain Ω = {(x₁, x₂) : x₁² + x₂²/0.64 < 1}
  Boundary parametrization: x = cos(2πt), y = 0.8·sin(2πt)

Reference: FreeFEM Example1.edp:
  border bound(t=0,1){x = cos(2*pi*t); y = 0.8*sin(2*pi*t); label = 1;};
  mesh Th = buildmesh(bound(nSolve));   // nSolve = 500
"""

import numpy as np
import meshpy.triangle as triangle


class EllipticMesh:
    """P1 triangular finite element mesh on an elliptic domain.

    Attributes
    ----------
    points : array (N, 2) — node coordinates
    triangles : array (M, 3) — triangle-to-node connectivity (counter-clockwise)
    boundary_edges : array (E, 2) — boundary edge node indices
    boundary_nodes : array (B,) — ordered boundary node indices
    n_points : int — total number of nodes
    n_triangles : int — total number of triangles
    n_boundary : int — number of boundary nodes
    areas : array (M,) — triangle areas
    grad_phi : array (M, 3, 2) — P1 basis function gradients per triangle
    centroids : array (M, 2) — triangle centroids
    """

    def __init__(self, points, triangles, boundary_edges):
        self.points = np.asarray(points, dtype=np.float64)
        self.triangles = np.asarray(triangles, dtype=np.int64)
        self.boundary_edges = np.asarray(boundary_edges, dtype=np.int64)

        self.boundary_nodes = self._extract_ordered_boundary_nodes()
        self.n_points = len(self.points)
        self.n_triangles = len(self.triangles)
        self.n_boundary = len(self.boundary_nodes)

        self._precompute_geometry()

    def _extract_ordered_boundary_nodes(self):
        """Extract boundary nodes in chain order from boundary edges."""
        edges = self.boundary_edges
        if len(edges) == 0:
            return np.array([], dtype=np.int64)

        adjacency = {}
        for e in edges:
            n0, n1 = int(e[0]), int(e[1])
            adjacency.setdefault(n0, []).append(n1)
            adjacency.setdefault(n1, []).append(n0)

        ordered = [int(edges[0, 0])]
        visited = {ordered[0]}
        current = ordered[0]

        while True:
            neighbors = adjacency[current]
            next_node = None
            for nb in neighbors:
                if nb not in visited:
                    next_node = nb
                    break
            if next_node is None:
                break
            ordered.append(next_node)
            visited.add(next_node)
            current = next_node

        return np.array(ordered, dtype=np.int64)

    def _precompute_geometry(self):
        """Precompute triangle areas and P1 basis function gradients.

        For a P1 triangle with vertices (x₀,y₀), (x₁,y₁), (x₂,y₂):
          area = 0.5 * |det([x₁−x₀, x₂−x₀; y₁−y₀, y₂−y₀])|
          ∇φ₀ = (1/2A) * (y₁−y₂, x₂−x₁)
          ∇φ₁ = (1/2A) * (y₂−y₀, x₀−x₂)
          ∇φ₂ = (1/2A) * (y₀−y₁, x₁−x₀)
        """
        p = self.points
        t = self.triangles

        v0 = p[t[:, 0]]
        v1 = p[t[:, 1]]
        v2 = p[t[:, 2]]

        d10 = v1 - v0
        d20 = v2 - v0

        signed_area = 0.5 * (d10[:, 0] * d20[:, 1] - d10[:, 1] * d20[:, 0])

        neg_mask = signed_area < 0
        if np.any(neg_mask):
            self.triangles[neg_mask, 1], self.triangles[neg_mask, 2] = (
                self.triangles[neg_mask, 2].copy(),
                self.triangles[neg_mask, 1].copy(),
            )
            signed_area[neg_mask] *= -1
            v0 = p[self.triangles[:, 0]]
            v1 = p[self.triangles[:, 1]]
            v2 = p[self.triangles[:, 2]]

        self.areas = signed_area

        inv_2A = 1.0 / (2.0 * self.areas)

        self.grad_phi = np.zeros((self.n_triangles, 3, 2))
        self.grad_phi[:, 0, 0] = inv_2A * (v1[:, 1] - v2[:, 1])
        self.grad_phi[:, 0, 1] = inv_2A * (v2[:, 0] - v1[:, 0])
        self.grad_phi[:, 1, 0] = inv_2A * (v2[:, 1] - v0[:, 1])
        self.grad_phi[:, 1, 1] = inv_2A * (v0[:, 0] - v2[:, 0])
        self.grad_phi[:, 2, 0] = inv_2A * (v0[:, 1] - v1[:, 1])
        self.grad_phi[:, 2, 1] = inv_2A * (v1[:, 0] - v0[:, 0])

        self.centroids = (v0 + v1 + v2) / 3.0

    def boundary_edge_lengths(self):
        """Compute boundary edge lengths.

        Returns
        -------
        lengths : array (E,)
        """
        p = self.points
        e = self.boundary_edges
        d = p[e[:, 1]] - p[e[:, 0]]
        return np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)

    def boundary_midpoints(self):
        """Compute boundary edge midpoints."""
        p = self.points
        e = self.boundary_edges
        return 0.5 * (p[e[:, 0]] + p[e[:, 1]])


def generate_elliptic_mesh(n_boundary=256, max_area=None):
    """Generate a triangular mesh for the elliptic domain Ω = {x₁² + x₂²/0.64 < 1}.

    Parameters
    ----------
    n_boundary : int
        Number of boundary nodes (Paper 1 reference value: nSolve=500).
    max_area : float or None
        Maximum triangle area constraint. If None, estimated from boundary spacing.

    Returns
    -------
    EllipticMesh
    """
    t_vals = np.linspace(0, 2 * np.pi, n_boundary, endpoint=False)
    boundary_points = np.column_stack([np.cos(t_vals), 0.8 * np.sin(t_vals)])

    info = triangle.MeshInfo()
    info.set_points(boundary_points.tolist())

    facets = [(i, (i + 1) % n_boundary) for i in range(n_boundary)]
    info.set_facets(facets)

    if max_area is None:
        avg_edge_len = np.mean(
            np.sqrt(np.sum(np.diff(np.vstack([boundary_points, boundary_points[0:1]]),
                                    axis=0) ** 2, axis=1))
        )
        max_area = 0.5 * avg_edge_len ** 2

    mesh_data = triangle.build(info, max_volume=max_area, min_angle=25)

    points = np.array(mesh_data.points)
    triangles = np.array(mesh_data.elements)

    boundary_edges = _extract_boundary_edges(triangles)

    return EllipticMesh(points, triangles, boundary_edges)


def _extract_boundary_edges(triangles):
    """Extract boundary edges from triangle connectivity.

    Boundary edges are shared by exactly one triangle.
    """
    edge_count = {}
    for tri in triangles:
        for i in range(3):
            e = tuple(sorted([tri[i], tri[(i + 1) % 3]]))
            edge_count[e] = edge_count.get(e, 0) + 1

    boundary = []
    for e, count in edge_count.items():
        if count == 1:
            boundary.append(e)

    return np.array(boundary, dtype=np.int64)


def generate_sampling_grid(n_grid=201, domain='ellipse'):
    """Generate a uniform sampling grid inside the domain for DSM/IDSM indicator scans.

    Parameters
    ----------
    n_grid : int — grid points per axis
    domain : str — 'ellipse' for x₁² + x₂²/0.64 < 1

    Returns
    -------
    grid_points : array (N_in, 2) — interior grid point coordinates
    grid_x : array (n_grid,)
    grid_y : array (n_grid,)
    mask : array (n_grid, n_grid), bool — interior mask
    """
    grid_x = np.linspace(-1.0, 1.0, n_grid)
    grid_y = np.linspace(-0.8, 0.8, n_grid)
    X, Y = np.meshgrid(grid_x, grid_y, indexing='xy')

    if domain == 'ellipse':
        mask = X ** 2 + Y ** 2 / 0.64 < 1.0
    else:
        mask = np.ones_like(X, dtype=bool)

    grid_points = np.column_stack([X[mask], Y[mask]])
    return grid_points, grid_x, grid_y, mask


def generate_coarse_mesh(target_triangles=1770, n_boundary=220):
    """Generate a coarse elliptic mesh for stabilizer projection.

    Paper 3, Section 5, Table 1: fine mesh T_f has ~15728 triangles,
    coarse mesh T_c has ~1770 triangles.

    Parameters
    ----------
    target_triangles : int
    n_boundary : int

    Returns
    -------
    EllipticMesh
    """
    area = np.pi * 0.8
    max_area = area / float(max(target_triangles, 1))
    return generate_elliptic_mesh(n_boundary=n_boundary, max_area=max_area)


def fine_to_coarse_p0(fine_mesh, coarse_mesh, fine_values):
    """Project P0 values from fine mesh to coarse mesh by area-weighted centroid matching."""
    fine_values = np.asarray(fine_values, dtype=np.float64)
    if fine_values.shape[0] != fine_mesh.n_triangles:
        raise ValueError("fine_values length must match fine_mesh.n_triangles")

    from scipy.spatial import cKDTree

    tree = cKDTree(coarse_mesh.centroids)
    _, nearest = tree.query(fine_mesh.centroids, k=1)

    coarse_sum = np.zeros(coarse_mesh.n_triangles, dtype=np.float64)
    coarse_w = np.zeros(coarse_mesh.n_triangles, dtype=np.float64)
    np.add.at(coarse_sum, nearest, fine_values * fine_mesh.areas)
    np.add.at(coarse_w, nearest, fine_mesh.areas)

    coarse_values = np.zeros_like(coarse_sum)
    nz = coarse_w > 0
    coarse_values[nz] = coarse_sum[nz] / coarse_w[nz]

    if np.any(~nz):
        known_idx = np.flatnonzero(nz)
        if known_idx.size > 0:
            fill_tree = cKDTree(coarse_mesh.centroids[known_idx])
            _, fill_j = fill_tree.query(coarse_mesh.centroids[~nz], k=1)
            coarse_values[~nz] = coarse_values[known_idx[fill_j]]
    return coarse_values


def coarse_to_fine_p0(fine_mesh, coarse_mesh, coarse_values):
    """Interpolate coarse P0 values back to fine mesh by nearest centroid."""
    coarse_values = np.asarray(coarse_values, dtype=np.float64)
    if coarse_values.shape[0] != coarse_mesh.n_triangles:
        raise ValueError("coarse_values length must match coarse_mesh.n_triangles")

    from scipy.spatial import cKDTree

    tree = cKDTree(coarse_mesh.centroids)
    _, nearest = tree.query(fine_mesh.centroids, k=1)
    return coarse_values[nearest]
