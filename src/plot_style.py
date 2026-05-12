"""Shared plotting style for IDSM tutorial and reproduction figures."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import PowerNorm
from matplotlib.patches import Circle


CONDUCTIVITY_CMAP = "viridis"
POTENTIAL_CMAP = "inferno"
NONLINEAR_CMAP = "magma"
FORWARD_CMAP = "RdBu_r"

SIGMA_EDGE = "#ff2d20"
POTENTIAL_EDGE = "#00c7ff"
DOMAIN_EDGE = "#222222"


def apply_idsm_plot_style() -> None:
    """Apply a compact, high-contrast style used by all generated figures."""
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 180,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#4a4a4a",
        "axes.linewidth": 0.8,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "font.size": 9,
        "legend.fontsize": 8,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })


def contrast_norm(vmin: float, vmax: float, *, gamma: float = 0.72) -> PowerNorm:
    """Power-law normalization that preserves bounds while revealing weak signals."""
    return PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)


def add_domain_boundary(ax, *, radius: float = 1.0) -> None:
    """Draw the unit disk boundary so cropped heatmaps share the same frame."""
    ax.add_patch(Circle(
        (0.0, 0.0),
        radius,
        fill=False,
        edgecolor=DOMAIN_EDGE,
        linewidth=0.7,
        alpha=0.85,
    ))


def format_domain_axis(ax, *, radius: float = 1.05) -> None:
    """Use the same frameless equal-aspect disk axis across all heatmaps."""
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#777777")


def add_outline_effect(artist, *, stroke: str = "white") -> None:
    """Keep truth outlines visible on both dark and bright color maps."""
    artist.set_path_effects([
        pe.Stroke(linewidth=artist.get_linewidth() + 1.8, foreground=stroke),
        pe.Normal(),
    ])


def save_figure(fig, path: str | Path, *, dpi: Optional[int] = None) -> Path:
    """Save with the repository's common rendering settings."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi or plt.rcParams["savefig.dpi"], bbox_inches="tight")
    return out
