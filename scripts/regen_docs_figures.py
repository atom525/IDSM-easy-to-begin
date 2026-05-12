"""Regenerate docs/ showcase figures for Paper 1 Example 1.

Example1.edp uses cB=0.01 as the search/projection lower bound, but the
default unknown-coefficient truth is cU=0.3.  These figures visualize that
truth and the associated forward/scattering data without notebook overhead.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mesh import generate_elliptic_mesh
from src.forward_solver import (
    make_conductivity_example1,
    solve_forward,
    generate_cauchy_data,
)
from src.plot_style import (
    CONDUCTIVITY_CMAP,
    FORWARD_CMAP,
    apply_idsm_plot_style,
    contrast_norm,
    save_figure,
)
from src.utils import EXAMPLE1_BOXES, add_truth_boxes

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))
os.makedirs(OUT, exist_ok=True)
apply_idsm_plot_style()

mesh = generate_elliptic_mesh(n_boundary=256)
sigma_true, _ = make_conductivity_example1(mesh)

# Source f1 = x  (mirrors NB01 default)
f1 = lambda x, y: x

# 1. true_conductivity
fig, ax = plt.subplots(figsize=(8, 6))
triang = mtri.Triangulation(mesh.points[:, 0], mesh.points[:, 1], mesh.triangles)
im = ax.tripcolor(triang, facecolors=sigma_true, cmap=CONDUCTIVITY_CMAP,
                  norm=contrast_norm(0.3, 1.0, gamma=0.72))
plt.colorbar(im, ax=ax, shrink=0.85)
add_truth_boxes(ax, EXAMPLE1_BOXES)
ax.set_aspect("equal")
ax.set_title("True Conductivity (Example 1, $c_A=1.0$, $c_U=0.3$)")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
plt.tight_layout()
save_figure(fig, os.path.join(OUT, "true_conductivity.png"))
plt.close(fig)
print("[ok] true_conductivity.png")

# 2. forward_solution (with inclusion)
y_omega = solve_forward(mesh, sigma_true, f1)
fig, ax = plt.subplots(figsize=(8, 6))
vlim = float(np.max(np.abs(y_omega)))
im = ax.tripcolor(triang, y_omega, cmap=FORWARD_CMAP, shading="gouraud",
                  vmin=-vlim, vmax=vlim)
plt.colorbar(im, ax=ax, shrink=0.85)
add_truth_boxes(ax, EXAMPLE1_BOXES)
ax.set_aspect("equal")
ax.set_title("Forward Solution (f=$x_1$, with inclusion)")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
plt.tight_layout()
save_figure(fig, os.path.join(OUT, "forward_solution.png"))
plt.close(fig)
print("[ok] forward_solution.png")

# 3. scattering_field = y_empty - y_omega
sigma_bg = np.ones(mesh.n_triangles)
y_empty = solve_forward(mesh, sigma_bg, f1)
scatter = y_empty - y_omega
fig, ax = plt.subplots(figsize=(8, 6))
vlim = float(np.max(np.abs(scatter)))
im = ax.tripcolor(triang, scatter, cmap=FORWARD_CMAP, shading="gouraud",
                  vmin=-vlim, vmax=vlim)
plt.colorbar(im, ax=ax, shrink=0.85)
add_truth_boxes(ax, EXAMPLE1_BOXES)
ax.set_aspect("equal")
ax.set_title(r"Scattering Field $y_\emptyset - y_\Omega$ (f=$x_1$)")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
plt.tight_layout()
save_figure(fig, os.path.join(OUT, "scattering_field.png"))
plt.close(fig)
print("[ok] scattering_field.png")

# 4. boundary_data (noise=10%)
data = generate_cauchy_data(mesh, sigma_true, [f1], noise_level=0.10,
                            rng=np.random.default_rng(42))
bdry = mesh.boundary_nodes
pts = mesh.points[bdry]
diffs = np.diff(pts, axis=0)
seg = np.sqrt((diffs ** 2).sum(axis=1))
arc = np.zeros(len(bdry)); arc[1:] = np.cumsum(seg)
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(arc, data["y_omega"][0][bdry], label="$y_\\Omega$ (exact)", linewidth=1)
ax.plot(arc, data["y_data"][0][bdry], label="$y_d$ (noisy 10%)", linewidth=1)
ax.plot(arc, data["y_empty"][0][bdry], label="$y_\\emptyset$ (background)", linewidth=1)
ax.set_xlabel("Arc length")
ax.set_ylabel("Value")
ax.set_title("Boundary Data (f=$x_1$, noise=10%, $c_U=0.3$)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
save_figure(fig, os.path.join(OUT, "boundary_data.png"))
plt.close(fig)
print("[ok] boundary_data.png")
