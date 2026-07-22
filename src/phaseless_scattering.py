"""Phaseless acoustic DSM utilities for arXiv:2403.02584 style experiments.

This module provides a compact, reproducible implementation of the numerical
building blocks used in the phaseless DSM section of the paper:

- 2D Helmholtz Green kernel and plane-wave illuminations.
- Synthetic total-field generation with a Born/Lippmann-Schwinger surrogate.
- Phaseless noise model and corrected data Δ(x_r, d) from Eq. (3.12).
- DSM index for phaseless data from Eq. (3.11), including multi-incidence
  averaging from Eq. (5.2).

The implementation is intentionally vectorized and deterministic (seeded RNG)
to support both notebook exploration and script/test reuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import special
import torch


EPS = 1e-12


@dataclass(frozen=True)
class PhaselessDSMConfig:
    """Runtime configuration for phaseless DSM simulations."""

    wavelength: float = 0.75
    receiver_radius: float = 4.0
    n_receivers: int = 100
    extent: float = 1.0
    forward_grid_size: int = 48
    scan_grid_size: int = 128
    born_iter: int = 2
    born_damping: float = 0.8
    regularization: float = 1e-6

    @property
    def k(self) -> float:
        """Wavenumber k = 2π/λ."""
        return 2.0 * np.pi / self.wavelength


@dataclass(frozen=True)
class ExampleSpec:
    """Geometry definition for paper Section 5.1 style scatterers."""

    name: str
    shapes: tuple[dict, ...]


def _validate_points(points: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2), got {arr.shape}")
    return arr


def make_uniform_grid(n: int, *, extent: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a Cartesian grid in [-extent, extent]^2.

    Returns
    -------
    grid_x, grid_y : array (n,)
    points : array (n*n, 2)
    """
    if n < 8:
        raise ValueError(f"Grid size n must be >= 8, got {n}")
    grid_x = np.linspace(-extent, extent, n)
    grid_y = np.linspace(-extent, extent, n)
    xx, yy = np.meshgrid(grid_x, grid_y, indexing="xy")
    points = np.column_stack([xx.ravel(), yy.ravel()])
    return grid_x, grid_y, points


def make_receiver_points(radius: float, n_receivers: int) -> np.ndarray:
    """Uniform receiver ring Γ_r."""
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    if n_receivers < 8:
        raise ValueError(f"n_receivers must be >= 8, got {n_receivers}")
    theta = np.linspace(0.0, 2.0 * np.pi, n_receivers, endpoint=False)
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])


def make_incident_angles(n_incident: int, *, offset: float = np.pi / 4.0) -> np.ndarray:
    """Angles θ_j = 2π(j-1)/N_i + offset."""
    if n_incident < 1:
        raise ValueError(f"n_incident must be >= 1, got {n_incident}")
    return offset + 2.0 * np.pi * np.arange(n_incident) / float(n_incident)


def incident_plane_wave(points: np.ndarray, *, k: float, angle: float) -> np.ndarray:
    """Evaluate u^i(x, d)=exp(i k x·d)."""
    pts = _validate_points(points, name="points")
    direction = np.array([np.cos(angle), np.sin(angle)], dtype=float)
    phase = k * (pts @ direction)
    return np.exp(1j * phase)


def green_kernel_2d(eval_points: np.ndarray, src_points: np.ndarray, *, k: float) -> np.ndarray:
    """2D Helmholtz Green function G(x,y)=i/4 H0^{(1)}(k|x-y|).

    The diagonal is regularized by replacing r=0 with EPS.
    """
    x = _validate_points(eval_points, name="eval_points")
    y = _validate_points(src_points, name="src_points")
    diff = x[:, None, :] - y[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    r = np.maximum(r, EPS)
    return 0.25j * special.hankel1(0, k * r)


def _shape_mask(points: np.ndarray, shape: dict) -> np.ndarray:
    """Rasterize one geometry primitive on point cloud."""
    kind = shape.get("kind")
    cx, cy = shape.get("center", (0.0, 0.0))
    x = points[:, 0]
    y = points[:, 1]
    dx = x - float(cx)
    dy = y - float(cy)

    if kind == "circle":
        r = float(shape["radius"])
        return dx * dx + dy * dy <= r * r
    if kind == "square":
        half = float(shape["half_width"])
        return (np.abs(dx) <= half) & (np.abs(dy) <= half)
    if kind == "ring":
        r_outer = float(shape["r_outer"])
        r_inner = float(shape["r_inner"])
        rr = np.sqrt(dx * dx + dy * dy)
        return (rr <= r_outer) & (rr >= r_inner)
    raise ValueError(f"Unsupported shape kind: {kind}")


def example_specs() -> dict[str, ExampleSpec]:
    """Paper-style Section 5.1 examples for DSM with phaseless data."""
    return {
        "ex1_medium_square": ExampleSpec(
            name="Example 1: medium square",
            shapes=(
                {
                    "kind": "square",
                    "center": (0.0, 0.0),
                    "half_width": 0.075,
                    "material": "medium",
                    "n_value": 3.0,
                },
            ),
        ),
        "ex2_sound_soft_squares": ExampleSpec(
            name="Example 2: two sound-soft squares",
            shapes=(
                {
                    "kind": "square",
                    "center": (-0.5, 0.5),
                    "half_width": 0.075,
                    "material": "soft",
                },
                {
                    "kind": "square",
                    "center": (0.5, -0.5),
                    "half_width": 0.075,
                    "material": "soft",
                },
            ),
        ),
        "ex3_close_medium_squares": ExampleSpec(
            name="Example 3: close medium squares",
            shapes=(
                {
                    "kind": "square",
                    "center": (0.1, 0.1),
                    "half_width": 0.075,
                    "material": "medium",
                    "n_value": 3.0,
                },
                {
                    "kind": "square",
                    "center": (-0.1, -0.1),
                    "half_width": 0.075,
                    "material": "medium",
                    "n_value": 3.0,
                },
            ),
        ),
        "ex4_medium_ring": ExampleSpec(
            name="Example 4: medium ring",
            shapes=(
                {
                    "kind": "ring",
                    "center": (0.0, 0.0),
                    "r_outer": 0.30,
                    "r_inner": 0.25,
                    "material": "medium",
                    "n_value": 3.0,
                },
            ),
        ),
    }


def make_refractive_index(points: np.ndarray, spec: ExampleSpec, *, n_background: float = 1.0) -> np.ndarray:
    """Build coefficient profile n(x) for a geometry spec.

    Paper Eq. (2.4) writes the medium equation as
    ``Δu + k^2 n(x) u = 0`` with background value ``n=1``.  The values in
    the synthetic labels therefore represent this coefficient directly, not
    its square root.
    """
    pts = _validate_points(points, name="points")
    n_field = np.full(pts.shape[0], n_background, dtype=float)
    for shape in spec.shapes:
        mask = _shape_mask(pts, shape)
        material = shape.get("material", "medium")
        if material == "medium":
            value = float(shape.get("n_value", 3.0))
        elif material == "soft":
            value = float(shape.get("n_value", 0.2))
        elif material == "hard":
            value = float(shape.get("n_value", 3.5))
        else:
            raise ValueError(f"Unsupported material type: {material}")
        n_field[mask] = value
    return n_field


def make_truth_mask(points: np.ndarray, spec: ExampleSpec) -> np.ndarray:
    """Binary support mask of all scatterer components."""
    pts = _validate_points(points, name="points")
    mask = np.zeros(pts.shape[0], dtype=bool)
    for shape in spec.shapes:
        mask |= _shape_mask(pts, shape)
    return mask


def add_phaseless_noise(
    abs_total_field: np.ndarray,
    *,
    noise_level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Eq. (5.1): |u_δ| = |u| + δ ζ ||u||_2."""
    if noise_level < 0:
        raise ValueError(f"noise_level must be nonnegative, got {noise_level}")
    abs_field = np.asarray(abs_total_field, dtype=float)
    l2_norm = float(np.sqrt(np.mean(abs_field ** 2)))
    zeta = rng.standard_normal(abs_field.shape)
    noisy = abs_field + noise_level * zeta * l2_norm
    return np.clip(noisy, 0.0, None)


def corrected_phaseless_data(
    abs_total_noisy: np.ndarray,
    abs_incident: np.ndarray,
    incident_complex: np.ndarray,
) -> np.ndarray:
    """Eq. (3.12): Δ = (|u|^2 - |u^i|^2) / u^i."""
    abs_total_noisy = np.asarray(abs_total_noisy, dtype=float)
    abs_incident = np.asarray(abs_incident, dtype=float)
    incident_complex = np.asarray(incident_complex, dtype=complex)
    if abs_total_noisy.shape != abs_incident.shape or abs_incident.shape != incident_complex.shape:
        raise ValueError(
            "Shape mismatch in corrected_phaseless_data: "
            f"{abs_total_noisy.shape}, {abs_incident.shape}, {incident_complex.shape}"
        )
    numerator = abs_total_noisy ** 2 - abs_incident ** 2
    denom = np.where(np.abs(incident_complex) < EPS, EPS + 0j, incident_complex)
    return numerator / denom


def _born_total_field_on_grid(
    grid_points: np.ndarray,
    refractive_index: np.ndarray,
    *,
    k: float,
    angle: float,
    born_iter: int,
    damping: float,
    regularization: float,
) -> np.ndarray:
    """Iterative Born/Lippmann-Schwinger surrogate on Cartesian grid points."""
    pts = _validate_points(grid_points, name="grid_points")
    n_field = np.asarray(refractive_index, dtype=float)
    if n_field.shape != (pts.shape[0],):
        raise ValueError(
            f"refractive_index must have shape ({pts.shape[0]},), got {n_field.shape}"
        )
    u_inc = incident_plane_wave(pts, k=k, angle=angle)
    if born_iter <= 0:
        return u_inc

    side_len = np.max(pts[:, 0]) - np.min(pts[:, 0])
    cell_area = (side_len / max(int(np.sqrt(pts.shape[0])) - 1, 1)) ** 2

    G = green_kernel_2d(pts, pts, k=k)
    np.fill_diagonal(G, 0.0 + 0.0j)

    contrast = k ** 2 * (n_field - 1.0)
    op = (G * cell_area) * contrast[None, :]
    if regularization > 0:
        op = op / (1.0 + regularization)

    u = u_inc.copy()
    for _ in range(born_iter):
        u_sc = op @ u
        u_next = u_inc + u_sc
        u = damping * u_next + (1.0 - damping) * u
    return u


def synthesize_total_field(
    grid_points: np.ndarray,
    receiver_points: np.ndarray,
    refractive_index: np.ndarray,
    *,
    k: float,
    angle: float,
    born_iter: int = 2,
    damping: float = 0.8,
    regularization: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate total field at receivers and the incident baseline."""
    grid = _validate_points(grid_points, name="grid_points")
    recv = _validate_points(receiver_points, name="receiver_points")
    u_grid = _born_total_field_on_grid(
        grid,
        refractive_index,
        k=k,
        angle=angle,
        born_iter=born_iter,
        damping=damping,
        regularization=regularization,
    )
    u_inc_grid = incident_plane_wave(grid, k=k, angle=angle)
    contrast = k ** 2 * (np.asarray(refractive_index, dtype=float) - 1.0)

    side_len = np.max(grid[:, 0]) - np.min(grid[:, 0])
    cell_area = (side_len / max(int(np.sqrt(grid.shape[0])) - 1, 1)) ** 2

    G_rx = green_kernel_2d(recv, grid, k=k)
    u_sc_rx = G_rx @ (contrast * u_grid) * cell_area
    u_inc_rx = incident_plane_wave(recv, k=k, angle=angle)
    u_total_rx = u_inc_rx + u_sc_rx
    return u_total_rx, u_inc_rx


def compute_phaseless_dsm_indicator(
    receiver_points: np.ndarray,
    corrected_data: np.ndarray,
    scan_points: np.ndarray,
    *,
    k: float,
) -> np.ndarray:
    """Eq. (3.11): I(z)=|∫_{Γr} G(z,x_r) Δ(x_r,d) ds|."""
    recv = _validate_points(receiver_points, name="receiver_points")
    scan = _validate_points(scan_points, name="scan_points")
    delta = np.asarray(corrected_data, dtype=complex)
    if delta.shape != (recv.shape[0],):
        raise ValueError(
            f"corrected_data must have shape ({recv.shape[0]},), got {delta.shape}"
        )
    radius = float(np.mean(np.linalg.norm(recv, axis=1)))
    ds = 2.0 * np.pi * radius / recv.shape[0]
    G = green_kernel_2d(scan, recv, k=k)
    indicator = np.abs(G @ delta) * ds
    return indicator.real


def compute_multi_incidence_indicator(
    *,
    scan_points: np.ndarray,
    grid_points: np.ndarray,
    receiver_points: np.ndarray,
    refractive_index: np.ndarray,
    k: float,
    incident_angles: Sequence[float],
    noise_level: float,
    rng: np.random.Generator,
    born_iter: int = 2,
    damping: float = 0.8,
    regularization: float = 1e-6,
) -> np.ndarray:
    """Average DSM index over multiple incidences (Eq. 5.2)."""
    values = []
    for angle in incident_angles:
        u_total, u_inc = synthesize_total_field(
            grid_points,
            receiver_points,
            refractive_index,
            k=k,
            angle=float(angle),
            born_iter=born_iter,
            damping=damping,
            regularization=regularization,
        )
        abs_noisy = add_phaseless_noise(np.abs(u_total), noise_level=noise_level, rng=rng)
        delta = corrected_phaseless_data(abs_noisy, np.abs(u_inc), u_inc)
        values.append(
            compute_phaseless_dsm_indicator(
                receiver_points,
                delta,
                scan_points,
                k=k,
            )
        )
    return np.mean(np.stack(values, axis=0), axis=0)


def normalize_indicator(indicator: np.ndarray) -> np.ndarray:
    """Scale index to [0, 1] with safe fallback for zero signals."""
    arr = np.asarray(indicator, dtype=float)
    vmax = float(np.max(arr))
    if vmax <= EPS:
        return np.zeros_like(arr)
    return arr / vmax


def run_example_dsm(
    *,
    example_key: str,
    cfg: PhaselessDSMConfig,
    noise_level: float,
    n_incident: int,
    seed: int = 0,
) -> dict:
    """Run one phaseless DSM experiment and return visualization-ready arrays."""
    specs = example_specs()
    if example_key not in specs:
        raise KeyError(f"Unknown example key: {example_key}. Available: {sorted(specs)}")

    spec = specs[example_key]
    rng = np.random.default_rng(seed)

    gx_fwd, gy_fwd, pts_fwd = make_uniform_grid(cfg.forward_grid_size, extent=cfg.extent)
    _ = gx_fwd, gy_fwd  # kept for debugging parity
    refractive = make_refractive_index(pts_fwd, spec)

    gx_scan, gy_scan, pts_scan = make_uniform_grid(cfg.scan_grid_size, extent=cfg.extent)
    recv = make_receiver_points(cfg.receiver_radius, cfg.n_receivers)
    angles = make_incident_angles(n_incident)
    indicator = compute_multi_incidence_indicator(
        scan_points=pts_scan,
        grid_points=pts_fwd,
        receiver_points=recv,
        refractive_index=refractive,
        k=cfg.k,
        incident_angles=angles,
        noise_level=noise_level,
        rng=rng,
        born_iter=cfg.born_iter,
        damping=cfg.born_damping,
        regularization=cfg.regularization,
    )

    truth_scan = make_truth_mask(pts_scan, spec).astype(float)
    return {
        "example_key": example_key,
        "example_name": spec.name,
        "grid_x": gx_scan,
        "grid_y": gy_scan,
        "indicator": normalize_indicator(indicator).reshape(cfg.scan_grid_size, cfg.scan_grid_size),
        "truth_mask": truth_scan.reshape(cfg.scan_grid_size, cfg.scan_grid_size),
        "noise_level": noise_level,
        "n_incident": n_incident,
        "seed": seed,
        "cfg": cfg,
    }


def topk_peak_hit_rate(
    indicator: np.ndarray,
    truth_mask: np.ndarray,
    *,
    k_fraction: float = 0.02,
) -> float:
    """Fraction of top-k indicator pixels that fall inside truth support."""
    ind = np.asarray(indicator, dtype=float).ravel()
    truth = np.asarray(truth_mask, dtype=float).ravel() > 0.5
    if ind.size != truth.size:
        raise ValueError("indicator and truth_mask must have identical number of elements")
    k = max(1, int(np.ceil(k_fraction * ind.size)))
    idx = np.argpartition(ind, -k)[-k:]
    return float(np.mean(truth[idx]))


def peak_location_error(
    indicator: np.ndarray,
    truth_mask: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> float:
    """Distance between max-indicator point and truth-support centroid."""
    ind = np.asarray(indicator, dtype=float)
    truth = np.asarray(truth_mask, dtype=float)
    if ind.shape != truth.shape:
        raise ValueError(f"indicator and truth_mask shape mismatch: {ind.shape} vs {truth.shape}")
    if ind.ndim != 2:
        raise ValueError("indicator must be 2D")
    iy, ix = np.unravel_index(int(np.argmax(ind)), ind.shape)
    px = float(grid_x[ix])
    py = float(grid_y[iy])

    yy, xx = np.nonzero(truth > 0.5)
    if yy.size == 0:
        return float(np.hypot(px, py))
    cx = float(np.mean(grid_x[xx]))
    cy = float(np.mean(grid_y[yy]))
    return float(np.hypot(px - cx, py - cy))


def batch_run_examples(
    *,
    cfg: PhaselessDSMConfig,
    jobs: Iterable[tuple[str, float, int, int]],
) -> list[dict]:
    """Run multiple (example_key, noise, n_incident, seed) jobs."""
    out: list[dict] = []
    for example_key, noise, n_incident, seed in jobs:
        out.append(
            run_example_dsm(
                example_key=example_key,
                cfg=cfg,
                noise_level=float(noise),
                n_incident=int(n_incident),
                seed=int(seed),
            )
        )
    return out


# ============================================================
# Batched strict Helmholtz forward + phaseless DSM indicator
# ============================================================


class PhaselessBatchSimulator:
    """Vectorized 2D Helmholtz forward solver and phaseless DSM indicator.

    Implements the paper's forward problem (Eq. 2.4-2.5) via a direct dense
    solve of the discretized Lippmann-Schwinger equation, plus the index
    function in Eq. (3.11) for phaseless data or Eq. (3.1) for phased data.

    Strict reproduction notes (arXiv:2403.02584v2):
    - Medium labels are the paper's coefficient ``n(x)`` in Eq. (2.4), so the
      VIE contrast is ``k^2 (n - 1)``.
    - The metadata-aware path dispatches sound-soft scatterers to Dirichlet BIE.
      The legacy ``compute_dsm_inputs`` path still supports a complex absorber
      fallback via ``n_soft`` for fast experiments.
    - The forward grid is solved via LU, so the medium-only result has no
      Born-truncation error.
    """

    def __init__(
        self,
        *,
        wavelength: float = 0.75,
        scan_grid_size: int = 64,
        forward_grid_size: int = 40,
        extent: float = 1.0,
        receiver_radius: float = 4.0,
        n_receivers: int = 100,
        n_soft: complex = 1.0 + 10.0j,
        solve_batch: int = 8,
        device: str = "cuda",
    ):
        if scan_grid_size < 16:
            raise ValueError(f"scan_grid_size must be >= 16, got {scan_grid_size}")
        if forward_grid_size < 16:
            raise ValueError(f"forward_grid_size must be >= 16, got {forward_grid_size}")
        self.wavelength = float(wavelength)
        self.k = 2.0 * np.pi / self.wavelength
        self.extent = float(extent)
        self.scan_grid_size = int(scan_grid_size)
        self.forward_grid_size = int(forward_grid_size)
        self.receiver_radius = float(receiver_radius)
        self.n_receivers = int(n_receivers)
        self.n_soft = complex(n_soft)
        self.solve_batch = int(solve_batch)

        resolved = device
        if device == "cuda" and not torch.cuda.is_available():
            resolved = "cpu"
        self.device = torch.device(resolved)

        self._build_static_tensors()
        self._inc_cache: dict[float, tuple[torch.Tensor, torch.Tensor]] = {}

    def _build_static_tensors(self) -> None:
        gx = np.linspace(-self.extent, self.extent, self.forward_grid_size)
        gy = np.linspace(-self.extent, self.extent, self.forward_grid_size)
        XX, YY = np.meshgrid(gx, gy, indexing="xy")
        self.fwd_pts_np = np.column_stack([XX.ravel(), YY.ravel()])
        self.fwd_dA = float((2.0 * self.extent) / max(self.forward_grid_size - 1, 1)) ** 2

        sx = np.linspace(-self.extent, self.extent, self.scan_grid_size)
        sy = np.linspace(-self.extent, self.extent, self.scan_grid_size)
        SX, SY = np.meshgrid(sx, sy, indexing="xy")
        self.scan_pts_np = np.column_stack([SX.ravel(), SY.ravel()])

        theta = 2.0 * np.pi * np.arange(self.n_receivers) / max(self.n_receivers, 1)
        self.recv_pts_np = np.column_stack(
            [
                self.receiver_radius * np.cos(theta),
                self.receiver_radius * np.sin(theta),
            ]
        )
        self.recv_ds = float(2.0 * np.pi * self.receiver_radius / max(self.n_receivers, 1))

        G_self = green_kernel_2d(self.fwd_pts_np, self.fwd_pts_np, k=self.k)
        np.fill_diagonal(G_self, 0.0 + 0.0j)
        G_recv = green_kernel_2d(self.recv_pts_np, self.fwd_pts_np, k=self.k)
        G_scan_recv = green_kernel_2d(self.scan_pts_np, self.recv_pts_np, k=self.k)

        self.G_self = torch.from_numpy(G_self.astype(np.complex64)).to(self.device)
        self.G_recv = torch.from_numpy(G_recv.astype(np.complex64)).to(self.device)
        self.G_scan_recv = torch.from_numpy(G_scan_recv.astype(np.complex64)).to(self.device)

    def _plane_wave(self, pts_np: np.ndarray, angle: float) -> torch.Tensor:
        d = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        phase = self.k * (pts_np @ d)
        u = np.exp(1j * phase).astype(np.complex64)
        return torch.from_numpy(u).to(self.device)

    def _get_incident(self, angle: float) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._inc_cache.get(angle)
        if cached is not None:
            return cached
        u_inc_fwd = self._plane_wave(self.fwd_pts_np, angle)
        u_inc_recv = self._plane_wave(self.recv_pts_np, angle)
        self._inc_cache[angle] = (u_inc_fwd, u_inc_recv)
        return self._inc_cache[angle]

    def _resize_to_forward(self, labels_t: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(
            labels_t.unsqueeze(1),
            size=(self.forward_grid_size, self.forward_grid_size),
            mode="bilinear",
            align_corners=True,
        ).squeeze(1)

    def _lu_factor_batch(self, contrast: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Factor I - op * diag(contrast) once per sample for reuse across angles."""
        op = self.G_self * self.fwd_dA
        N = op.shape[0]
        eye = torch.eye(N, dtype=op.dtype, device=op.device)
        A = eye[None, :, :] - op[None, :, :] * contrast[:, None, :]
        LU, pivots = torch.linalg.lu_factor(A)
        return LU, pivots

    def _total_at_recv_with_lu(
        self,
        LU: torch.Tensor,
        pivots: torch.Tensor,
        contrast: torch.Tensor,
        angle: float,
    ) -> torch.Tensor:
        u_inc_fwd, u_inc_r = self._get_incident(angle)
        B = contrast.shape[0]
        rhs = u_inc_fwd[None, :, None].expand(B, -1, 1).contiguous()
        u = torch.linalg.lu_solve(LU, pivots, rhs).squeeze(-1)
        cu_final = contrast * u
        u_sc_r = (cu_final @ self.G_recv.T) * self.fwd_dA
        return u_sc_r + u_inc_r[None, :]

    def _forward_total_at_recv(self, n_field_b: torch.Tensor, angle: float) -> torch.Tensor:
        if not torch.is_complex(n_field_b):
            n_field_b = n_field_b.to(torch.complex64)
        contrast = (self.k ** 2) * (n_field_b - 1.0)
        u_total_r_chunks: list[torch.Tensor] = []
        B_total = contrast.shape[0]
        for start in range(0, B_total, self.solve_batch):
            stop = min(start + self.solve_batch, B_total)
            c_sub = contrast[start:stop]
            LU, piv = self._lu_factor_batch(c_sub)
            u_total_r_chunks.append(self._total_at_recv_with_lu(LU, piv, c_sub, angle))
        return torch.cat(u_total_r_chunks, dim=0)

    def forward_total_at_receivers(
        self,
        labels: np.ndarray,
        angle: float,
    ) -> np.ndarray:
        """Solve the strict medium VIE and return total receiver fields.

        Parameters
        ----------
        labels : ndarray, shape (H, W) or (B, H, W)
            Real refractive-index labels on the scan grid. This public teaching
            step is for penetrable media; obstacle metadata should use the BIE
            functions in :mod:`src.phaseless_bie`.
        angle : float
            Plane-wave incidence angle in radians.

        Returns
        -------
        ndarray, shape (B, n_receivers)
            Complex total fields. The method uses the same LU-discretized VIE
            as :func:`run_example_dsm_paper`, enabling exact manual/wrapper
            comparisons in Notebook 06.
        """
        arr = np.asarray(labels, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        if arr.ndim != 3 or arr.shape[1:] != (
            self.scan_grid_size,
            self.scan_grid_size,
        ):
            raise ValueError(
                "labels must have shape "
                f"(H, W) or (B, H, W) with H=W={self.scan_grid_size}; "
                f"got {arr.shape}"
            )
        labels_t = torch.from_numpy(arr).to(self.device)
        n_fwd_real = self._resize_to_forward(labels_t)
        n_fwd = self._labels_to_complex_index(n_fwd_real).reshape(arr.shape[0], -1)
        total = self._forward_total_at_recv(n_fwd, float(angle))
        return total.detach().cpu().numpy()

    def _phaseless_indicator(
        self,
        total_r: torch.Tensor,
        angle: float,
        noise_level: float,
        rng: np.random.Generator,
    ) -> torch.Tensor:
        """Eq. (3.11)-(3.12) phaseless DSM index."""
        abs_field = torch.abs(total_r)
        l2 = torch.sqrt(torch.mean(abs_field ** 2, dim=1, keepdim=True))
        zeta_np = rng.standard_normal(abs_field.shape).astype(np.float32)
        zeta = torch.from_numpy(zeta_np).to(self.device)
        abs_noisy = torch.clamp(abs_field + noise_level * zeta * l2, min=0.0)

        _, u_inc_r = self._get_incident(angle)
        abs_inc = torch.abs(u_inc_r)
        delta_real = abs_noisy ** 2 - abs_inc[None, :] ** 2
        denom = torch.where(
            torch.abs(u_inc_r) < 1e-12, torch.ones_like(u_inc_r), u_inc_r
        )
        delta = delta_real.to(torch.complex64) / denom[None, :]
        ind = (delta @ self.G_scan_recv.T) * self.recv_ds
        return torch.abs(ind)

    def _phased_indicator(
        self,
        total_r: torch.Tensor,
        angle: float,
        noise_level: float,
        rng: np.random.Generator,
    ) -> torch.Tensor:
        """Eq. (3.1) phased DSM index ``I_DSM(z) = |<G(z,.), u^s>|``."""
        _, u_inc_r = self._get_incident(angle)
        u_sc = total_r - u_inc_r[None, :]
        zeta_real = rng.standard_normal(u_sc.shape).astype(np.float32)
        zeta_imag = rng.standard_normal(u_sc.shape).astype(np.float32)
        zeta = torch.complex(
            torch.from_numpy(zeta_real).to(self.device),
            torch.from_numpy(zeta_imag).to(self.device),
        )
        l2 = torch.sqrt(torch.mean(torch.abs(u_sc) ** 2, dim=1, keepdim=True))
        u_sc_noisy = u_sc + noise_level * zeta * l2
        ind = (torch.conj(u_sc_noisy) @ self.G_scan_recv.T) * self.recv_ds
        return torch.abs(ind)

    def _labels_to_complex_index(self, labels_t: torch.Tensor) -> torch.Tensor:
        """Map real-valued label fields to complex refractive index.

        - background (label == 1.0)   -> n = 1.0
        - medium (label > 1)          -> n = label (real)
        - sound-soft (label < 0.5)    -> n = self.n_soft (complex absorber)
        """
        n_real = labels_t.to(torch.complex64)
        soft_mask = (labels_t < 0.5)
        n_complex_const = torch.tensor(self.n_soft, dtype=torch.complex64, device=self.device)
        return torch.where(soft_mask, n_complex_const, n_real)

    # ------------------------------------------------------------------
    # Strict BIE Dirichlet + VIE Born-superposition path
    # ------------------------------------------------------------------

    def _build_scatterer_boundary(
        self, meta: dict, n_per_object: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from . import phaseless_bie as bie

        geom = meta.get("geom")
        if geom == "circle":
            center = meta["center"]
            radius = float(meta["radius"])
            return bie.boundary_circle(center, radius, n_per_object)
        if geom == "polygon":
            verts = np.asarray(meta["vertices"], dtype=np.float64)
            n_sides = max(verts.shape[0], 1)
            per_side = max(8, int(np.ceil(n_per_object / n_sides)))
            return bie.boundary_polygon(verts, per_side)
        raise ValueError(f"Unsupported geometry: {geom}")

    def _u_sc_soft_at_recv(self, meta: dict, angle: float, n_per_object: int) -> np.ndarray:
        """Single-scatterer sound-soft Dirichlet BIE scattered field at receivers."""
        from . import phaseless_bie as bie

        pts, _normals, lengths = self._build_scatterer_boundary(meta, n_per_object)
        phi = bie.solve_sound_soft(pts, lengths, self.k, angle)
        G_r_b = bie._green_2d(self.recv_pts_np, pts, self.k)
        return ((G_r_b * lengths[None, :]) @ phi).astype(np.complex64)

    def _u_sc_medium_at_recv_batch(
        self,
        medium_only_labels: torch.Tensor,
        angle: float,
    ) -> torch.Tensor:
        n_field_real = self._resize_to_forward(medium_only_labels)
        n_fwd_complex = n_field_real.to(torch.complex64).reshape(
            medium_only_labels.shape[0], -1
        )
        contrast = (self.k ** 2) * (n_fwd_complex - 1.0)
        out_chunks: list[torch.Tensor] = []
        B = contrast.shape[0]
        u_inc_fwd, _u_inc_r = self._get_incident(angle)
        for s in range(0, B, self.solve_batch):
            e = min(s + self.solve_batch, B)
            c_sub = contrast[s:e]
            LU, piv = self._lu_factor_batch(c_sub)
            rhs = u_inc_fwd[None, :, None].expand(e - s, -1, 1).contiguous()
            u_total = torch.linalg.lu_solve(LU, piv, rhs).squeeze(-1)
            cu_final = c_sub * u_total
            u_sc_r = (cu_final @ self.G_recv.T) * self.fwd_dA
            out_chunks.append(u_sc_r)
        return torch.cat(out_chunks, dim=0)

    def _rasterize_soft_mask(self, meta: dict, H: int, W: int) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(-self.extent, self.extent, H, device=self.device),
            torch.linspace(-self.extent, self.extent, W, device=self.device),
            indexing="ij",
        )
        if meta["geom"] == "circle":
            cx, cy = meta["center"]
            r = float(meta["radius"])
            return ((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2) <= (r * r)
        if meta["geom"] == "polygon":
            verts = np.asarray(meta["vertices"], dtype=np.float64)
            mask_np = _point_in_polygon_grid(verts, H, W, self.extent)
            return torch.from_numpy(mask_np).to(self.device)
        raise ValueError(f"Unsupported geometry: {meta['geom']}")

    def _u_total_with_meta(
        self,
        labels: np.ndarray,
        metas: list[list[dict]],
        angle: float,
        n_per_object: int,
    ) -> torch.Tensor:
        """Born superposition: VIE medium contribution + BIE Dirichlet soft contribution + u_inc."""
        B, H, W = labels.shape
        labels_t = torch.from_numpy(labels.astype(np.float32)).to(self.device)
        medium_only = labels_t.clone()
        for i, sample_meta in enumerate(metas):
            for m in sample_meta:
                if m["kind"] == "soft":
                    mask = self._rasterize_soft_mask(m, H, W)
                    medium_only[i][mask] = 1.0
        u_sc_medium_recv = self._u_sc_medium_at_recv_batch(medium_only, angle)
        u_sc_soft_recv = torch.zeros_like(u_sc_medium_recv)
        for i, sample_meta in enumerate(metas):
            for m in sample_meta:
                if m["kind"] != "soft":
                    continue
                u_sc_i = self._u_sc_soft_at_recv(m, angle, n_per_object=n_per_object)
                u_sc_soft_recv[i] += torch.from_numpy(u_sc_i).to(self.device)
        _, u_inc_r = self._get_incident(angle)
        return u_sc_medium_recv + u_sc_soft_recv + u_inc_r[None, :]

    def compute_dsm_inputs_with_meta(
        self,
        labels: np.ndarray,
        metas: list[list[dict]],
        *,
        n_incident: int,
        noise_level: float,
        seed: int,
        W_norm: float | None = None,
        input_scale: float = 2.0,
        phased: bool = False,
        n_per_object: int = 240,
    ) -> tuple[np.ndarray, float]:
        """Strict BIE Dirichlet (paper Eq. 2.3) + VIE Born-superposition DSM inputs."""
        if labels.ndim != 3:
            raise ValueError(f"labels must have shape (B, H, W), got {labels.shape}")
        B, H, Wd = labels.shape
        if (H, Wd) != (self.scan_grid_size, self.scan_grid_size):
            raise ValueError(
                f"labels spatial size must be {self.scan_grid_size}x{self.scan_grid_size}, got {(H, Wd)}"
            )
        if len(metas) != B:
            raise ValueError(f"metas length {len(metas)} != labels batch {B}")

        rng = np.random.default_rng(int(seed))
        angles = (
            np.pi / 4.0
            + 2.0 * np.pi * np.arange(n_incident) / max(n_incident, 1)
        ).astype(np.float64)
        indicator_fn = self._phased_indicator if phased else self._phaseless_indicator

        out = np.zeros((B, n_incident, H, Wd), dtype=np.float32)
        for ci, ang in enumerate(angles):
            total_r = self._u_total_with_meta(labels, metas, float(ang), n_per_object)
            ind = indicator_fn(total_r, float(ang), noise_level, rng)
            out[:, ci] = ind.reshape(B, H, Wd).detach().cpu().numpy()

        if W_norm is None:
            W = float(np.max(out)) + 1e-12
        else:
            W = float(W_norm)
        out *= float(input_scale) / W
        return out, W

    def compute_dsm_inputs(
        self,
        labels: np.ndarray,
        *,
        n_incident: int,
        noise_level: float,
        seed: int,
        W_norm: float | None = None,
        input_scale: float = 2.0,
        batch_size: int = 16,
        phased: bool = False,
    ) -> tuple[np.ndarray, float]:
        """Compute DSM indicator channels from refractive labels."""
        if labels.ndim != 3:
            raise ValueError(f"labels must have shape (B, H, W), got {labels.shape}")
        B, H, Wd = labels.shape
        if (H, Wd) != (self.scan_grid_size, self.scan_grid_size):
            raise ValueError(
                f"labels spatial size must be {self.scan_grid_size}x{self.scan_grid_size}, got {(H, Wd)}"
            )
        if n_incident < 1:
            raise ValueError(f"n_incident must be >= 1, got {n_incident}")
        if noise_level < 0:
            raise ValueError(f"noise_level must be nonnegative, got {noise_level}")

        rng = np.random.default_rng(int(seed))
        angles = (
            np.pi / 4.0
            + 2.0 * np.pi * np.arange(n_incident) / max(n_incident, 1)
        ).astype(np.float64)
        indicator_fn = self._phased_indicator if phased else self._phaseless_indicator

        out_chunks: list[np.ndarray] = []
        labels_np = np.asarray(labels, dtype=np.float32)
        for start in range(0, B, batch_size):
            stop = min(start + batch_size, B)
            chunk_np = labels_np[start:stop]
            chunk_t = torch.from_numpy(chunk_np).to(self.device)
            n_fwd_real = self._resize_to_forward(chunk_t)
            n_fwd_complex = self._labels_to_complex_index(n_fwd_real).reshape(stop - start, -1)
            out = torch.zeros(
                (stop - start, n_incident, H, Wd),
                dtype=torch.float32,
                device=self.device,
            )
            B_sub = stop - start
            for s in range(0, B_sub, self.solve_batch):
                e = min(s + self.solve_batch, B_sub)
                c_sub = (self.k ** 2) * (n_fwd_complex[s:e] - 1.0)
                LU, piv = self._lu_factor_batch(c_sub)
                for ci, ang in enumerate(angles):
                    total_r = self._total_at_recv_with_lu(LU, piv, c_sub, float(ang))
                    ind = indicator_fn(total_r, float(ang), noise_level, rng)
                    out[s:e, ci] = ind.reshape(e - s, H, Wd)
            out_chunks.append(out.detach().cpu().numpy())

        out_np = np.concatenate(out_chunks, axis=0)
        if W_norm is None:
            W = float(np.max(out_np)) + 1e-12
        else:
            W = float(W_norm)
        out_np *= float(input_scale) / W
        return out_np, W


def run_example_dsm_paper(
    *,
    example_key: str,
    noise_level: float,
    n_incident: int,
    seed: int = 0,
    wavelength: float = 0.75,
    receiver_radius: float = 4.0,
    n_receivers: int = 100,
    scan_grid_size: int = 160,
    boundary_density: int = 320,
    device: str = "cuda",
) -> dict:
    """Run a paper-style Section 5.1 example with the correct forward physics.

    - Example 1 (medium square): exact volume integral equation.
    - Example 2 (sound-soft squares): BIE Dirichlet (single-layer formulation).
    - Examples 3-4 (medium scatterers): exact volume integral equation via the
      cached :class:`PhaselessBatchSimulator`.
    """
    from . import phaseless_bie as bie

    specs = example_specs()
    if example_key not in specs:
        raise KeyError(f"Unknown example key: {example_key}")
    spec = specs[example_key]
    k = 2.0 * np.pi / float(wavelength)
    rng = np.random.default_rng(int(seed))

    recv_pts = make_receiver_points(receiver_radius, n_receivers)
    grid_x, grid_y, scan_pts = make_uniform_grid(scan_grid_size, extent=1.0)
    truth_mask = make_truth_mask(scan_pts, spec).reshape(scan_grid_size, scan_grid_size)

    angles = make_incident_angles(n_incident)

    material_types = {shape.get("material") for shape in spec.shapes}
    impenetrable = material_types.issubset({"soft", "hard"})

    indicator_accum = np.zeros(scan_pts.shape[0], dtype=float)

    if impenetrable:
        # Build combined boundary discretization.
        parts = []
        is_hard = False
        for shape in spec.shapes:
            kind = shape["kind"]
            if shape.get("material") == "hard":
                is_hard = True
            if kind == "circle":
                parts.append(
                    bie.boundary_circle(shape["center"], float(shape["radius"]), boundary_density)
                )
            elif kind == "square":
                parts.append(
                    bie.boundary_square(shape["center"], float(shape["half_width"]), boundary_density // 4)
                )
            else:
                raise ValueError(f"BIE branch does not handle kind={kind}")
        boundary_pts, boundary_normals, boundary_lengths = bie.stack_boundaries(parts)

        for angle in angles:
            if is_hard:
                phi = bie.solve_sound_hard(boundary_pts, boundary_normals, boundary_lengths, k, float(angle))
            else:
                phi = bie.solve_sound_soft(boundary_pts, boundary_lengths, k, float(angle))
            u_total = bie.evaluate_total_at(recv_pts, boundary_pts, boundary_lengths, phi, k, float(angle))
            abs_total = np.abs(u_total)
            noisy = add_phaseless_noise(abs_total, noise_level=noise_level, rng=rng)
            delta = corrected_phaseless_data(noisy, np.abs(np.exp(1j * k * (recv_pts @ np.array([np.cos(angle), np.sin(angle)])))), np.exp(1j * k * (recv_pts @ np.array([np.cos(angle), np.sin(angle)]))))
            ind = compute_phaseless_dsm_indicator(recv_pts, delta, scan_pts, k=k)
            indicator_accum += ind
    else:
        # Medium scatterers use VIE via the strict simulator.
        sim = PhaselessBatchSimulator(
            wavelength=wavelength,
            scan_grid_size=scan_grid_size,
            forward_grid_size=48,
            receiver_radius=receiver_radius,
            n_receivers=n_receivers,
            device=device,
        )
        # Build refractive label for medium scatterer geometry on scan grid.
        n_field_label = make_refractive_index(scan_pts, spec).reshape(scan_grid_size, scan_grid_size)
        labels = n_field_label[None, :, :].astype(np.float32)
        labels_t = torch.from_numpy(labels).to(sim.device)
        n_fwd_real = sim._resize_to_forward(labels_t)
        n_fwd = sim._labels_to_complex_index(n_fwd_real).reshape(1, -1)
        for angle in angles:
            total_r = sim._forward_total_at_recv(n_fwd, float(angle))
            ind = sim._phaseless_indicator(total_r, float(angle), noise_level, rng)
            indicator_accum += ind[0].detach().cpu().numpy().reshape(-1)

    indicator_accum /= float(n_incident)
    indicator_norm = normalize_indicator(indicator_accum).reshape(scan_grid_size, scan_grid_size)

    return {
        "example_key": example_key,
        "example_name": spec.name,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "indicator": indicator_norm,
        "truth_mask": truth_mask,
        "noise_level": noise_level,
        "n_incident": n_incident,
        "seed": seed,
        "is_impenetrable": impenetrable,
    }


def _point_in_polygon_grid(verts: np.ndarray, H: int, W: int, extent: float) -> np.ndarray:
    """Boolean mask of (H, W) grid points lying inside the closed polygon."""
    ys = np.linspace(-extent, extent, H)
    xs = np.linspace(-extent, extent, W)
    YY, XX = np.meshgrid(ys, xs, indexing="ij")
    inside = np.zeros_like(XX, dtype=bool)
    n_v = verts.shape[0]
    j = n_v - 1
    for i in range(n_v):
        yi = verts[i, 1]; xi = verts[i, 0]
        yj = verts[j, 1]; xj = verts[j, 0]
        cond = ((yi > YY) != (yj > YY)) & (
            XX < (xj - xi) * (YY - yi) / (yj - yi + 1e-18) + xi
        )
        inside = np.where(cond, ~inside, inside)
        j = i
    return inside


def make_phaseless_simulator(
    *,
    wavelength: float = 0.75,
    scan_grid_size: int = 64,
    forward_grid_size: int = 48,
    receiver_radius: float = 4.0,
    n_receivers: int = 100,
    n_soft: complex = 1.0 + 10.0j,
    solve_batch: int = 4,
    device: str = "cuda",
) -> PhaselessBatchSimulator:
    """Factory matching dataset-level parameters."""
    return PhaselessBatchSimulator(
        wavelength=wavelength,
        scan_grid_size=scan_grid_size,
        forward_grid_size=forward_grid_size,
        receiver_radius=receiver_radius,
        n_receivers=n_receivers,
        n_soft=n_soft,
        solve_batch=solve_batch,
        device=device,
    )

