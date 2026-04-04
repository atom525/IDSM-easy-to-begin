"""
utils.py - Utility functions

Visualization, distance computation, evaluation metrics, etc.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import Normalize


def plot_mesh(mesh, title='Mesh', figsize=(8, 6), save_path=None):
    """Visualize the triangular mesh."""
    fig, ax = plt.subplots(figsize=figsize)
    triang = mtri.Triangulation(mesh.points[:, 0], mesh.points[:, 1], mesh.triangles)
    ax.triplot(triang, 'k-', linewidth=0.3, alpha=0.5)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_field(mesh, values, title='', figsize=(8, 6), cmap='RdBu_r',
               vmin=None, vmax=None, save_path=None, show_boundary=True,
               inclusion_boxes=None):
    """Visualize a scalar field on the domain (P1 nodal values).

    Parameters
    ----------
    mesh : EllipticMesh
    values : array (N,) — nodal values
    inclusion_boxes : list of dict, e.g. {'center': (cx,cy), 'half_width': hw, 'color': 'w'}
    """
    fig, ax = plt.subplots(figsize=figsize)
    triang = mtri.Triangulation(mesh.points[:, 0], mesh.points[:, 1], mesh.triangles)

    im = ax.tripcolor(triang, values, cmap=cmap, shading='gouraud',
                       vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax)

    if show_boundary:
        bdry = mesh.boundary_nodes
        bdry_pts = mesh.points[bdry]
        ax.plot(np.append(bdry_pts[:, 0], bdry_pts[0, 0]),
                np.append(bdry_pts[:, 1], bdry_pts[0, 1]),
                'k-', linewidth=1)

    if inclusion_boxes:
        for box in inclusion_boxes:
            cx, cy = box['center']
            hw = box['half_width']
            color = box.get('color', 'w')
            rect = plt.Rectangle((cx - hw, cy - hw), 2 * hw, 2 * hw,
                                  linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_p0_field(mesh, values, title='', figsize=(8, 6), cmap='RdBu_r',
                  vmin=None, vmax=None, save_path=None, inclusion_boxes=None):
    """Visualize a P0 (piecewise constant) scalar field."""
    fig, ax = plt.subplots(figsize=figsize)
    triang = mtri.Triangulation(mesh.points[:, 0], mesh.points[:, 1], mesh.triangles)

    im = ax.tripcolor(triang, facecolors=values, cmap=cmap,
                       vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax)

    if inclusion_boxes:
        for box in inclusion_boxes:
            cx, cy = box['center']
            hw = box['half_width']
            color = box.get('color', 'w')
            rect = plt.Rectangle((cx - hw, cy - hw), 2 * hw, 2 * hw,
                                  linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_boundary_data(mesh, values_list, labels=None, title='Boundary Data',
                       figsize=(10, 4), save_path=None):
    """Visualize data on boundary nodes, parametrized by arc length."""
    fig, ax = plt.subplots(figsize=figsize)
    bdry = mesh.boundary_nodes
    pts = mesh.points[bdry]

    diffs = np.diff(pts, axis=0)
    seg_lengths = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
    arc = np.zeros(len(bdry))
    arc[1:] = np.cumsum(seg_lengths)

    if labels is None:
        labels = ['Data %d' % (i+1) for i in range(len(values_list))]

    for vals, label in zip(values_list, labels):
        ax.plot(arc, vals[bdry], label=label, linewidth=1)

    ax.set_xlabel('Arc length')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def distance_to_boundary(mesh, points):
    """Compute the distance from interior points to the boundary Γ.

    d(x, Γ) = inf_{x' ∈ Γ} |x − x'|

    Used in the DSM denominator approximation (Paper 1, Eq. 2.10):
      ⟨G(·,x), G(·,x)⟩_Γ ≈ C·d(x,Γ)^γ

    Parameters
    ----------
    mesh : EllipticMesh
    points : array (K, 2)

    Returns
    -------
    dist : array (K,) — shortest distance to boundary
    """
    bdry_pts = mesh.points[mesh.boundary_nodes]
    points = np.asarray(points)

    diff = points[:, None, :] - bdry_pts[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    return np.min(dists, axis=1)


def fundamental_solution_2d(x, x_prime):
    """2D Laplace fundamental solution.

    Φ_x(x') = −1/(2π) ln|x − x'|

    Parameters
    ----------
    x : array (2,) or (K, 2) — source point(s)
    x_prime : array (M, 2) — field points

    Returns
    -------
    Phi : array (K, M) or (M,)
    """
    x = np.atleast_2d(x)
    diff = x[:, None, :] - x_prime[None, :, :]
    r = np.sqrt(np.sum(diff**2, axis=2))
    r = np.maximum(r, 1e-15)
    Phi = -1.0 / (2 * np.pi) * np.log(r)
    return Phi.squeeze()


def compute_iou(u_true, u_pred, mesh):
    """Compute Intersection over Union (IoU) using area-matched thresholding.

    The predicted field is ranked by absolute value in descending order, and
    the top elements matching the true inclusion area are selected as the
    predicted inclusion region.  This approach is insensitive to reconstruction
    amplitude and evaluates spatial localization accuracy only.

    Parameters
    ----------
    u_true : array (M,) — true inclusion (P0, non-zero = inclusion)
    u_pred : array (M,) — predicted inclusion (P0)
    mesh : EllipticMesh

    Returns
    -------
    iou : float in [0, 1]
    """
    true_mask = np.abs(u_true) > 1e-10
    true_area = np.sum(mesh.areas[true_mask])

    if true_area < 1e-15:
        return 0.0

    sorted_idx = np.argsort(-np.abs(u_pred))
    cumarea = np.cumsum(mesh.areas[sorted_idx])
    k = np.searchsorted(cumarea, true_area)
    k = min(k, len(u_pred) - 1)

    pred_mask = np.zeros(len(u_pred), dtype=bool)
    pred_mask[sorted_idx[:k + 1]] = True

    intersection = np.sum(mesh.areas[true_mask & pred_mask])
    union = np.sum(mesh.areas[true_mask | pred_mask])

    if union < 1e-15:
        return 0.0
    return intersection / union


def p0_to_grid(mesh, p0_values, grid_points):
    """Map triangle-wise P0 values to arbitrary grid points via nearest centroid."""
    p0_values = np.asarray(p0_values, dtype=np.float64)
    if p0_values.shape[0] != mesh.n_triangles:
        raise ValueError("p0_values length must equal mesh.n_triangles")
    from scipy.spatial import cKDTree
    tree = cKDTree(mesh.centroids)
    _, tri_idx = tree.query(np.asarray(grid_points), k=1)
    return p0_values[tri_idx]


def compute_iou_from_grid(mesh, u_true, indicator_grid, mask):
    """Compute IoU for a grid-based indicator map.

    Parameters
    ----------
    mesh : EllipticMesh
    u_true : array (M,) — ground-truth P0 inclusion field
    indicator_grid : array (n_grid, n_grid) — indicator values (NaN outside domain)
    mask : array (n_grid, n_grid), bool — interior point mask

    Returns
    -------
    float — IoU between ground-truth and area-matched thresholded indicator
    """
    if indicator_grid.shape != mask.shape:
        raise ValueError("indicator_grid and mask must have same shape")

    true_tri_mask = np.abs(u_true) > 1e-10
    true_area = np.sum(mesh.areas[true_tri_mask])
    if true_area <= 1e-15:
        return 0.0

    inside_count = int(np.sum(mask))
    domain_area = np.sum(mesh.areas)
    target_count = int(np.round((true_area / max(domain_area, 1e-30)) * inside_count))
    target_count = max(1, min(target_count, inside_count))

    grid_idx = np.column_stack(np.nonzero(mask))
    n_grid_y, n_grid_x = indicator_grid.shape
    xs = grid_idx[:, 1] / max(n_grid_x - 1, 1) * 2.0 - 1.0
    ys = grid_idx[:, 0] / max(n_grid_y - 1, 1) * 1.6 - 0.8
    grid_points = np.column_stack([xs, ys])
    true_vals_grid = p0_to_grid(mesh, u_true, grid_points)
    true_mask_grid = np.abs(true_vals_grid) > 1e-10

    pred_vals = np.asarray(indicator_grid[mask], dtype=np.float64)
    order = np.argsort(-np.abs(pred_vals))
    pred_mask_grid = np.zeros_like(pred_vals, dtype=bool)
    pred_mask_grid[order[:target_count]] = True

    inter = np.sum(true_mask_grid & pred_mask_grid)
    union = np.sum(true_mask_grid | pred_mask_grid)
    if union == 0:
        return 0.0
    return float(inter) / float(union)


# Inclusion box markers for Example 1
EXAMPLE1_BOXES = [
    {'center': (0.4, 0.2), 'half_width': 0.2, 'color': 'w'},
    {'center': (-0.5, -0.2), 'half_width': 0.2, 'color': 'w'},
]

SINGLE_INCLUSION_CIRCLE = [
    {'center': (0.3, 0.0), 'radius': 0.25, 'color': 'w'},
]
