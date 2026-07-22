"""Generate the Paper 3 Example 6.1 unit-disk figure.

This is the paper-profile counterpart to the shared-ellipse comparisons in
``run_nb04_figures.py``.  It uses the parameters in Paper 3 Table 1 and writes
one canonical figure to ``figures/04_comparative``.
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forward_solver import solve_forward
from src.idsm_partial import define_accessible_boundary, run_idsm_partial
from src.mesh import generate_disk_mesh_paper


def make_data(mesh, noise_level, seed):
    cent = mesh.centroids
    square = (
        (np.abs(cent[:, 0] - 0.10) < 0.14)
        & (np.abs(cent[:, 1] - 0.52) < 0.14)
    )
    circle = np.linalg.norm(cent - np.array([-0.25, -0.35]), axis=1) < 0.15
    sigma = np.ones(mesh.n_triangles)
    sigma[square | circle] = 0.1

    sources = [
        lambda x, y: np.sin(4.0 * np.pi * x) + 0.5,
        lambda x, y: np.cos(4.0 * np.pi * y) + 0.5,
    ]
    y_true = [solve_forward(mesh, sigma, f) for f in sources]
    y_empty = [solve_forward(mesh, np.ones(mesh.n_triangles), f) for f in sources]
    rng = np.random.default_rng(seed)
    y_data = [
        yt + noise_level * rng.uniform(-1.0, 1.0, mesh.n_points) * (yt - ye)
        for yt, ye in zip(y_true, y_empty)
    ]
    return sigma, {
        "y_omega": y_true,
        "y_empty": y_empty,
        "y_data": y_data,
        "sources": sources,
    }


def add_truth(ax):
    ax.add_patch(Rectangle(
        (0.10 - 0.14, 0.52 - 0.14), 0.28, 0.28,
        fill=False, edgecolor="black", linewidth=1.5,
    ))
    ax.add_patch(Circle(
        (-0.25, -0.35), 0.15,
        fill=False, edgecolor="black", linewidth=1.5,
    ))


def main():
    mesh = generate_disk_mesh_paper(target_triangles=15728)
    coarse = generate_disk_mesh_paper(target_triangles=1770)
    gamma_d = define_accessible_boundary(
        mesh, (-np.pi / 2, np.pi / 2), a=1.0, b=1.0,
    )

    histories = []
    for noise in (0.15, 0.30):
        _, data = make_data(mesh, noise, seed=42)
        histories.append(run_idsm_partial(
            mesh, data, gamma_d,
            sigma_bg=1.0, potential_bg=1e-10,
            sigma_range=0.01, potential_range=2e-10,
            alpha_d=0.05, alpha_n=2.0,
            n_iter=30, lowrank_method="BFG",
            problem_type="conductivity", coeff_known=False,
            gamma_D=4.0, epsilon_cutoff=0.02, p_norm=2.0,
            coarse_mesh=coarse, stabilization=True, verbose=False,
        ))

    tri = plt.matplotlib.tri.Triangulation(
        mesh.points[:, 0], mesh.points[:, 1], mesh.triangles,
    )
    fig, axes = plt.subplots(1, 7, figsize=(15.5, 2.5))

    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    axes[0].plot(np.cos(theta), np.sin(theta), color="0.75", linewidth=1.0)
    right = (theta <= np.pi / 2) | (theta >= 3.0 * np.pi / 2)
    axes[0].plot(np.cos(theta[right]), np.sin(theta[right]), color="black", linewidth=2.5)
    axes[0].set_title(r"$\Gamma_D$")
    axes[0].set_aspect("equal")
    axes[0].axis("off")

    snapshots = (0, 9, 29)
    image = None
    col = 1
    for noise, history in zip((0.15, 0.30), histories):
        for idx, label in zip(snapshots, (0, 10, 30)):
            contrast = (1.0 - history["sigma_guess"][idx]) / 0.9
            image = axes[col].tripcolor(
                tri, contrast, shading="flat", cmap="coolwarm",
                vmin=0.0, vmax=1.0,
            )
            add_truth(axes[col])
            axes[col].set_title(
                fr"$\epsilon={100.0 * noise:.0f}\%$, $k={label}$"
            )
            axes[col].set_aspect("equal")
            axes[col].axis("off")
            col += 1

    fig.colorbar(image, ax=axes, fraction=0.018, pad=0.015, label="normalized inclusion")
    fig.suptitle("Paper 3 Example 6.1: partial-data EIT on the unit disk", y=1.02)
    out = ROOT / "figures" / "04_comparative" / "04_paper3_example1.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
