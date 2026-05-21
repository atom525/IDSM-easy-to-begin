"""Boundary integral equation (BIE) forward solvers used for Section 5.1 figures.

These routines implement the paper's true Dirichlet (sound-soft) and Neumann
(sound-hard) boundary value problems for the impenetrable obstacles in
Examples 1-2 of arXiv:2403.02584. The medium scatterers (Examples 3-4) are
left to the volume integral solver in :mod:`phaseless_scattering`.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import special

EPS = 1e-12


def _green_2d(p1: np.ndarray, p2: np.ndarray, k: float) -> np.ndarray:
    diff = p1[:, None, :] - p2[None, :, :]
    r = np.linalg.norm(diff, axis=-1)
    r = np.maximum(r, EPS)
    return 0.25j * special.hankel1(0, k * r)


def _green_dn_x(p1: np.ndarray, p2: np.ndarray, normals_x: np.ndarray, k: float) -> np.ndarray:
    """Normal derivative of the 2D Helmholtz Green function with respect to the source x."""
    diff = p1[:, None, :] - p2[None, :, :]
    r = np.linalg.norm(diff, axis=-1)
    r_safe = np.maximum(r, EPS)
    proj = (diff * normals_x[:, None, :]).sum(axis=-1) / r_safe
    return -0.25j * k * special.hankel1(1, k * r_safe) * proj


def boundary_circle(center: tuple[float, float], radius: float, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = (np.arange(n) + 0.5) * (2.0 * np.pi / n)
    pts = np.array(center) + radius * np.column_stack([np.cos(theta), np.sin(theta)])
    normals = np.column_stack([np.cos(theta), np.sin(theta)])
    lengths = np.full(n, 2.0 * np.pi * radius / n)
    return pts, normals, lengths


def boundary_polygon(vertices: np.ndarray, n_per_side: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discretize a closed polygon boundary into evenly spaced sample points.

    Parameters
    ----------
    vertices : (N_v, 2) array of ordered (CCW) polygon vertices.
    n_per_side : number of quadrature points to place on each polygon edge.
    """
    if vertices.shape[0] < 3:
        raise ValueError(f"polygon requires >= 3 vertices, got {vertices.shape[0]}")
    pts_list: list[np.ndarray] = []
    normals_list: list[np.ndarray] = []
    lengths_list: list[np.ndarray] = []
    n_v = vertices.shape[0]
    for i in range(n_v):
        c0 = vertices[i]
        c1 = vertices[(i + 1) % n_v]
        edge = c1 - c0
        L = float(np.linalg.norm(edge))
        if L < 1e-12:
            continue
        tangent = edge / L
        # Outward normal: rotate tangent by -90 deg assuming CCW polygon.
        outward = np.array([tangent[1], -tangent[0]])
        t = (np.arange(n_per_side) + 0.5) / n_per_side
        side_pts = (1.0 - t)[:, None] * c0[None, :] + t[:, None] * c1[None, :]
        pts_list.append(side_pts)
        normals_list.append(np.tile(outward, (n_per_side, 1)))
        lengths_list.append(np.full(n_per_side, L / n_per_side))
    pts = np.concatenate(pts_list, axis=0)
    normals = np.concatenate(normals_list, axis=0)
    lengths = np.concatenate(lengths_list, axis=0)
    return pts, normals, lengths


def boundary_square(center: tuple[float, float], half_w: float, n_per_side: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cx, cy = center
    corners = np.array(
        [
            (cx - half_w, cy - half_w),
            (cx + half_w, cy - half_w),
            (cx + half_w, cy + half_w),
            (cx - half_w, cy + half_w),
        ],
        dtype=float,
    )
    outward = np.array([(0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)])
    pts_list: list[np.ndarray] = []
    normals_list: list[np.ndarray] = []
    L_side = 2.0 * half_w / n_per_side
    for i in range(4):
        c0 = corners[i]
        c1 = corners[(i + 1) % 4]
        t = (np.arange(n_per_side) + 0.5) / n_per_side
        side_pts = (1.0 - t)[:, None] * c0[None, :] + t[:, None] * c1[None, :]
        pts_list.append(side_pts)
        normals_list.append(np.tile(outward[i], (n_per_side, 1)))
    pts = np.concatenate(pts_list, axis=0)
    normals = np.concatenate(normals_list, axis=0)
    lengths = np.full(n_per_side * 4, L_side)
    return pts, normals, lengths


def _diag_log_correction(L: np.ndarray, k: float) -> np.ndarray:
    """Approximate diagonal of the single-layer matrix via the small-argument limit
    of H_0^{(1)}.  This is sufficient for visualization-quality BIE.
    """
    out = np.empty(L.shape[0], dtype=complex)
    for i in range(L.shape[0]):
        out[i] = 0.25j * special.hankel1(0, k * L[i] / 4.0)
    return out


def solve_sound_soft(
    boundary_pts: np.ndarray,
    boundary_lengths: np.ndarray,
    k: float,
    angle: float,
) -> np.ndarray:
    """Solve single-layer Dirichlet: ``∫G(x, y) φ(y) ds = -u_inc(x)`` on the boundary."""
    G = _green_2d(boundary_pts, boundary_pts, k)
    diag_vals = _diag_log_correction(boundary_lengths, k)
    np.fill_diagonal(G, diag_vals)
    A = G * boundary_lengths[None, :]
    d = np.array([np.cos(angle), np.sin(angle)])
    u_inc = np.exp(1j * k * (boundary_pts @ d))
    return np.linalg.solve(A, -u_inc)


def solve_sound_hard(
    boundary_pts: np.ndarray,
    boundary_normals: np.ndarray,
    boundary_lengths: np.ndarray,
    k: float,
    angle: float,
) -> np.ndarray:
    """Solve indirect single-layer Neumann via the adjoint double-layer.

    ``(-I/2 + K^T) φ = -∂_n u_inc`` on the boundary.
    """
    KT = _green_dn_x(boundary_pts, boundary_pts, boundary_normals, k)
    np.fill_diagonal(KT, 0.0 + 0.0j)
    A = -0.5 * np.eye(KT.shape[0], dtype=complex) + KT * boundary_lengths[None, :]
    d = np.array([np.cos(angle), np.sin(angle)])
    u_inc = np.exp(1j * k * (boundary_pts @ d))
    du_inc_dn = 1j * k * (boundary_normals @ d) * u_inc
    return np.linalg.solve(A, -du_inc_dn)


def evaluate_total_at(
    recv_pts: np.ndarray,
    boundary_pts: np.ndarray,
    boundary_lengths: np.ndarray,
    phi: np.ndarray,
    k: float,
    angle: float,
) -> np.ndarray:
    G_r = _green_2d(recv_pts, boundary_pts, k)
    u_s_r = (G_r * boundary_lengths[None, :]) @ phi
    d = np.array([np.cos(angle), np.sin(angle)])
    u_inc_r = np.exp(1j * k * (recv_pts @ d))
    return u_inc_r + u_s_r


def stack_boundaries(parts: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.concatenate([p[0] for p in parts], axis=0)
    normals = np.concatenate([p[1] for p in parts], axis=0)
    lengths = np.concatenate([p[2] for p in parts], axis=0)
    return pts, normals, lengths
