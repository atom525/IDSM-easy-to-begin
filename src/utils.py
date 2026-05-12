"""
utils.py - Utility functions

Visualization, distance computation, evaluation metrics, etc.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize, PowerNorm, TwoSlopeNorm


def plot_mesh(mesh, title='Mesh', figsize=(8, 6), save_path=None):
    """Visualize the triangular mesh."""
    fig, ax = plt.subplots(figsize=figsize)
    triang = mtri.Triangulation(mesh.points[:, 0], mesh.points[:, 1], mesh.triangles)
    ax.triplot(triang, color='#202020', linewidth=0.38, alpha=0.65)
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
                                  linewidth=2.5, edgecolor=color, facecolor='none')
            rect.set_path_effects([pe.Stroke(linewidth=4.0, foreground='black'), pe.Normal()])
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
                                  linewidth=2.5, edgecolor=color, facecolor='none')
            rect.set_path_effects([pe.Stroke(linewidth=4.0, foreground='black'), pe.Normal()])
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
    {'center': (0.4, 0.2), 'half_width': 0.2, 'color': 'lime'},
    {'center': (-0.5, -0.2), 'half_width': 0.2, 'color': 'lime'},
]

SINGLE_INCLUSION_CIRCLE = [
    {'center': (0.3, 0.0), 'radius': 0.25, 'color': 'lime'},
]


# ============================================================
# 统一可视化规范 (Phase 1-5 共用)
# ============================================================
# 统一 colormap 选择
CMAP_SIGMA = 'viridis'       # 导电率重建：低值深色，高值亮色
CMAP_SIGNED = 'RdBu_r'       # u = sigma - sigma_0：负值蓝，正值红
CMAP_INDICATOR = 'magma'     # DSM/IDSM η ∈ [0,1] 指示器：高值亮色
CMAP_FORWARD = 'RdBu_r'      # 前向解 y(x)
CMAP_POTENTIAL = 'plasma'    # 势 V 重建：高吸收更亮
CMAP_CLASSIFY = 'Greys'      # 分类二值图

# 统一 colormap 数值范围 (Example 1: cA=1.0 background, cU=0.3 truth; cB=0.01 is projection lower bound)
SIGMA_VMIN_EX1 = 0.01  # projection lower bound (FreeFEM cB)
SIGMA_VMAX_EX1 = 1.0   # = cA
# Partial / Example 2 high-contrast (cB=0.01 极弱导电):
SIGMA_VMIN_EX2 = 0.01
SIGMA_VMAX_EX2 = 1.0
# 势重建 (Example 3: vA=1, vB=30):
V_VMIN = 1.0
V_VMAX = 30.0
# 指示器范围
ETA_VMIN = 0.0
ETA_VMAX = 1.0

# 真值 inclusion 框样式 (亮色 + 黑色描边，保证在深/浅背景上都可见)
TRUTH_RECT_KW = dict(linewidth=2.8, edgecolor='#baff00', facecolor='none')
TRUTH_CIRCLE_KW = dict(linewidth=2.8, edgecolor='#baff00', facecolor='none')


def sigma_norm(vmin=SIGMA_VMIN_EX1, vmax=SIGMA_VMAX_EX1, gamma=1.45):
    """Contrast-enhancing normalization for conductivity fields."""
    return PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)


def signed_contrast_norm(limit=0.8):
    """Centered normalization for signed contrast u = sigma - sigma_0."""
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def _outline_artist(artist, stroke='black'):
    artist.set_path_effects([
        pe.Stroke(linewidth=artist.get_linewidth() + 1.7, foreground=stroke),
        pe.Normal(),
    ])


def add_truth_boxes(ax, boxes, kw=None):
    """在 ax 上画真值 inclusion 框（统一 lime 2.5 样式）。

    boxes : list of {'center': (cx,cy), 'half_width': hw, ...}
    """
    if kw is None:
        kw = TRUTH_RECT_KW
    for box in boxes:
        cx, cy = box['center']
        hw = box['half_width']
        rect = plt.Rectangle((cx - hw, cy - hw), 2 * hw, 2 * hw, **kw)
        _outline_artist(rect)
        ax.add_patch(rect)


def add_truth_circles(ax, circles, kw=None):
    """在 ax 上画真值圆形 inclusion (Phase 5 抛物 / 单圆例)。

    circles : list of {'center': (cx,cy), 'radius': r, ...}
    """
    if kw is None:
        kw = TRUTH_CIRCLE_KW
    for c in circles:
        cx, cy = c['center']
        r = c['radius']
        circ = plt.Circle((cx, cy), r, **kw)
        _outline_artist(circ)
        ax.add_patch(circ)


def plot_sigma_reconstruction(ax, mesh, sigma, vmin=SIGMA_VMIN_EX1,
                               vmax=SIGMA_VMAX_EX1, boxes=None, title=None,
                               show_colorbar=True):
    """统一 σ 重建可视化（在指定 ax 上）。

    返回 (im, cbar)。cbar 为 None 时未画。
    """
    triang = mtri.Triangulation(mesh.points[:, 0], mesh.points[:, 1],
                                  mesh.triangles)
    im = ax.tripcolor(triang, facecolors=sigma, cmap=CMAP_SIGMA,
                       norm=sigma_norm(vmin, vmax))
    if boxes is not None:
        add_truth_boxes(ax, boxes)
    ax.set_aspect('equal')
    if title is not None:
        ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046) if show_colorbar else None
    return im, cbar


def plot_indicator_grid(ax, eta, extent=(-1, 1, -0.8, 0.8), vmin=ETA_VMIN,
                         vmax=ETA_VMAX, title=None, boxes=None,
                         show_colorbar=True):
    """统一 DSM/IDSM 指示器（grid）可视化。

    eta : 2D array on uniform grid
    extent : (xmin, xmax, ymin, ymax) 用于 imshow 坐标
    """
    im = ax.imshow(eta, extent=extent, origin='lower',
                    cmap=CMAP_INDICATOR, vmin=vmin, vmax=vmax,
                    aspect='equal')
    if boxes is not None:
        add_truth_boxes(ax, boxes)
    if title is not None:
        ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046) if show_colorbar else None
    return im, cbar
