"""Regenerate docs/ showcase figures with cB=0.01.

The four images embedded in README/historical docs were generated at the
cB=0.3 era; with the cB=0.01 (near-insulator) truth we now use, the
inclusions need a corresponding deeper colour-scale. This script reproduces
them without any Notebook overhead.
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
from src.utils import EXAMPLE1_BOXES, add_truth_boxes, CMAP_FORWARD

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))
os.makedirs(OUT, exist_ok=True)

mesh = generate_elliptic_mesh(n_boundary=256)
sigma_true, _ = make_conductivity_example1(mesh)

# Source f1 = x  (mirrors NB01 default)
f1 = lambda x, y: x

# 1. true_conductivity
fig, ax = plt.subplots(figsize=(8, 6))
triang = mtri.Triangulation(mesh.points[:, 0], mesh.points[:, 1], mesh.triangles)
im = ax.tripcolor(triang, facecolors=sigma_true, cmap="RdBu_r", vmin=0.01, vmax=1.0)
plt.colorbar(im, ax=ax)
add_truth_boxes(ax, EXAMPLE1_BOXES)
ax.set_aspect("equal")
ax.set_title("True Conductivity (Example 1, $c_A=1.0$, $c_B=0.01$)")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "true_conductivity.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("[ok] true_conductivity.png")

# 2. forward_solution (with inclusion)
y_omega = solve_forward(mesh, sigma_true, f1)
fig, ax = plt.subplots(figsize=(8, 6))
vlim = float(np.max(np.abs(y_omega)))
im = ax.tripcolor(triang, y_omega, cmap=CMAP_FORWARD, shading="gouraud",
                  vmin=-vlim, vmax=vlim)
plt.colorbar(im, ax=ax)
add_truth_boxes(ax, EXAMPLE1_BOXES)
ax.set_aspect("equal")
ax.set_title("Forward Solution (f=$x_1$, with inclusion)")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "forward_solution.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("[ok] forward_solution.png")

# 3. scattering_field = y_empty - y_omega
sigma_bg = np.ones(mesh.n_triangles)
y_empty = solve_forward(mesh, sigma_bg, f1)
scatter = y_empty - y_omega
fig, ax = plt.subplots(figsize=(8, 6))
vlim = float(np.max(np.abs(scatter)))
im = ax.tripcolor(triang, scatter, cmap=CMAP_FORWARD, shading="gouraud",
                  vmin=-vlim, vmax=vlim)
plt.colorbar(im, ax=ax)
add_truth_boxes(ax, EXAMPLE1_BOXES)
ax.set_aspect("equal")
ax.set_title(r"Scattering Field $y_\emptyset - y_\Omega$ (f=$x_1$)")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "scattering_field.png"), dpi=150, bbox_inches="tight")
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
ax.set_title("Boundary Data (f=$x_1$, noise=10%, $c_B=0.01$)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "boundary_data.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("[ok] boundary_data.png")
