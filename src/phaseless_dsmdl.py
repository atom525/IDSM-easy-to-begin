"""DSM-DL components for phaseless inverse scattering notebook reproduction.

This module provides production-style components for:
1) dataset generation (polygon / MNIST+circle / mixed circle),
2) multi-channel DSM-like inputs,
3) U-Net model and training loop,
4) metrics and post-processing used in the paper tables.

The implementation is deliberately split from notebooks so scripts/tests can
reuse the same logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from .phaseless_scattering import PhaselessBatchSimulator

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MNIST_ROOT = str(_REPO_ROOT / "data" / "mnist")


Tensor = torch.Tensor


_SIMULATOR_CACHE: dict[tuple, PhaselessBatchSimulator] = {}


def _simulator_key(cfg: "DatasetConfig") -> tuple:
    return (
        int(cfg.image_size),
        int(cfg.forward_grid_size),
        float(cfg.wavelength),
        float(cfg.receiver_radius),
        int(cfg.n_receivers),
        float(cfg.n_soft_real),
        float(cfg.n_soft_imag),
        int(cfg.solve_batch),
        str(cfg.device),
    )


def get_simulator(cfg: "DatasetConfig") -> PhaselessBatchSimulator:
    """Return a cached simulator instance matching the dataset config."""
    key = _simulator_key(cfg)
    sim = _SIMULATOR_CACHE.get(key)
    if sim is None:
        sim = PhaselessBatchSimulator(
            wavelength=cfg.wavelength,
            scan_grid_size=cfg.image_size,
            forward_grid_size=cfg.forward_grid_size,
            receiver_radius=cfg.receiver_radius,
            n_receivers=cfg.n_receivers,
            n_soft=complex(cfg.n_soft_real, cfg.n_soft_imag),
            solve_batch=cfg.solve_batch,
            device=cfg.device,
        )
        _SIMULATOR_CACHE[key] = sim
    return sim


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset generation options aligned with arXiv:2403.02584 Section 5."""

    image_size: int = 64
    n_incident: int = 4
    wavelength: float = 0.75
    forward_grid_size: int = 40
    receiver_radius: float = 4.0
    n_receivers: int = 100
    n_soft_real: float = 1.0
    n_soft_imag: float = 10.0
    solve_batch: int = 8
    input_scale: float = 2.0
    seed: int = 0
    mnist_root: str = _DEFAULT_MNIST_ROOT
    mnist_download: bool = True
    device: str = "cuda"
    mnist_rotation_deg: float = 180.0


@dataclass(frozen=True)
class TrainingConfig:
    """Training settings following paper-style defaults (Section 5.2)."""

    epochs: int = 30
    batch_size: int = 10
    learning_rate: float = 1e-3
    lr_step: int = 3
    lr_gamma: float = 0.5
    alpha_tv: float = 0.5
    alpha_ssim: float = 0.5
    weight_decay: float = 1e-6
    device: str = "cpu"
    seed: int = 0


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _draw_polygon_mask(size: int, n_sides: int, radius: float, center: tuple[float, float], angle: float) -> np.ndarray:
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    cx, cy = center
    vertices = []
    for i in range(n_sides):
        theta = angle + 2.0 * np.pi * i / n_sides
        x = cx + radius * np.cos(theta)
        y = cy + radius * np.sin(theta)
        vertices.append((x, y))
    draw.polygon(vertices, fill=255)
    return (np.asarray(img, dtype=np.uint8) > 0).astype(np.float32)


def _draw_circle_mask(size: int, center: tuple[float, float], radius: float) -> np.ndarray:
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    cx, cy = center
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=255)
    return (np.asarray(img, dtype=np.uint8) > 0).astype(np.float32)


def generate_polygon_labels(
    n_samples: int,
    cfg: DatasetConfig,
) -> np.ndarray:
    """Generate mixed polygon labels with values in {0, 1, 3} (labels only)."""
    labels, _ = generate_polygon_labels_with_meta(n_samples, cfg)
    return labels


def generate_polygon_labels_with_meta(
    n_samples: int,
    cfg: DatasetConfig,
) -> tuple[np.ndarray, list[list[dict]]]:
    """Same as ``generate_polygon_labels`` but also returns per-sample scatterer metadata.

    Each entry of the returned list is a list of one dict describing the polygon
    (since paper §5.2.1 places exactly one polygon per sample). The dict has
    ``kind`` ('medium' or 'soft'), ``geom`` ('polygon') and ``vertices`` in
    physical coordinates :math:`[-1, 1]^2` ordered counter-clockwise.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    size = cfg.image_size
    rng = _rng(cfg.seed)
    labels = np.ones((n_samples, size, size), dtype=np.float32)
    metas: list[list[dict]] = []
    half = size / 2.0
    extent = 1.0

    def _to_phys(px: float, py: float) -> tuple[float, float]:
        # Pixel (px, py) with origin at top-left in image array,
        # but our mask drawing uses (cx, cy) interpreted in PIL's drawing space
        # (x = column, y = row from top). For consistency with how the labels
        # are interpreted on the [-1, 1]^2 simulator grid, we keep the same
        # affine map used by simulator: x = (col / (size-1)) * 2 - 1,
        # y = (row / (size-1)) * 2 - 1. PIL drawing uses y from top, but we
        # treat label[i, j] = (row=i, col=j) and the simulator builds grid
        # from np.linspace, so both share the same orientation.
        x_phys = (px / (size - 1)) * 2.0 - 1.0
        y_phys = (py / (size - 1)) * 2.0 - 1.0
        return x_phys, y_phys

    for i in range(n_samples):
        n_sides = int(rng.integers(3, 7))
        radius = float(rng.uniform(0.30, 0.50)) * size * 0.50
        cx = float(rng.uniform(radius + 2.0, size - radius - 2.0))
        cy = float(rng.uniform(radius + 2.0, size - radius - 2.0))
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        mask = _draw_polygon_mask(size, n_sides, radius, (cx, cy), angle)
        is_medium = bool(rng.integers(0, 2))
        labels[i][mask > 0.5] = 3.0 if is_medium else 0.0
        # Build CCW vertex list in physical coordinates matching the PIL drawing.
        verts_px: list[tuple[float, float]] = []
        for j in range(n_sides):
            theta = angle + 2.0 * np.pi * j / n_sides
            verts_px.append((cx + radius * np.cos(theta), cy + radius * np.sin(theta)))
        verts_phys = np.array([list(_to_phys(px, py)) for px, py in verts_px], dtype=np.float64)
        # Ensure CCW ordering via signed area.
        area2 = sum(
            (verts_phys[(k + 1) % n_sides, 0] - verts_phys[k, 0])
            * (verts_phys[(k + 1) % n_sides, 1] + verts_phys[k, 1])
            for k in range(n_sides)
        )
        if area2 > 0:
            # shoelace > 0 means clockwise in image-style y-down coordinates;
            # reverse to obtain CCW in standard math orientation.
            verts_phys = verts_phys[::-1]
        metas.append([
            {
                "kind": "medium" if is_medium else "soft",
                "geom": "polygon",
                "n_value": 3.0 if is_medium else 0.0,
                "vertices": verts_phys,
            }
        ])
    return labels, metas


def generate_mixed_circle_labels(
    n_samples: int,
    cfg: DatasetConfig,
) -> np.ndarray:
    """Generate mixed-circle labels as in paper Section 5.2.3 (labels only)."""
    labels, _ = generate_mixed_circle_labels_with_meta(n_samples, cfg)
    return labels


def generate_mixed_circle_labels_with_meta(
    n_samples: int,
    cfg: DatasetConfig,
) -> tuple[np.ndarray, list[list[dict]]]:
    """Paper §5.2.3 mixed-circle dataset with per-circle metadata for strict forwards."""
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    size = cfg.image_size
    rng = _rng(cfg.seed)
    labels = np.ones((n_samples, size, size), dtype=np.float32)
    metas: list[list[dict]] = []

    def _to_phys(px: float, py: float) -> tuple[float, float]:
        return (px / (size - 1)) * 2.0 - 1.0, (py / (size - 1)) * 2.0 - 1.0

    for i in range(n_samples):
        n_obj = int(rng.integers(1, 4))
        centers: list[tuple[float, float, float]] = []
        sample_meta: list[dict] = []
        for _ in range(n_obj):
            for _try in range(80):
                radius = float(rng.uniform(0.20, 0.30) * size * 0.50)
                cx = float(rng.uniform(radius + 2.0, size - radius - 2.0))
                cy = float(rng.uniform(radius + 2.0, size - radius - 2.0))
                ok = True
                for px, py, pr in centers:
                    if np.hypot(cx - px, cy - py) <= (radius + pr + 2.0):
                        ok = False
                        break
                if ok:
                    centers.append((cx, cy, radius))
                    break

        for cx, cy, radius in centers:
            mask = _draw_circle_mask(size, (cx, cy), radius)
            if bool(rng.integers(0, 2)):
                value = float(rng.uniform(1.5, 3.0))
                kind = "medium"
            else:
                value = 0.0
                kind = "soft"
            labels[i][mask > 0.5] = value
            cx_phys, cy_phys = _to_phys(cx, cy)
            r_phys = radius / (size - 1) * 2.0
            sample_meta.append({
                "kind": kind,
                "geom": "circle",
                "n_value": value,
                "center": (cx_phys, cy_phys),
                "radius": r_phys,
            })
        metas.append(sample_meta)
    return labels, metas


def generate_mnist_circle_labels(
    n_samples: int,
    cfg: DatasetConfig,
    *,
    split: str = "train",
) -> np.ndarray:
    """Generate MNIST+circle labels with coefficients in [1.2, 1.7]."""
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    size = cfg.image_size
    rng = _rng(cfg.seed)
    root = Path(cfg.mnist_root)
    root.mkdir(parents=True, exist_ok=True)

    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split}")
    ds = None
    try:
        ds = datasets.MNIST(
            root=str(root),
            train=(split == "train"),
            download=cfg.mnist_download,
        )
    except Exception:
        # Network or mirror failures should not block full reproductions.
        # Fallback keeps the same "digit-like + circle" structure and
        # coefficient range so downstream DSM-DL experiments can proceed.
        ds = None

    def _synthetic_digit_like(seed_idx: int) -> np.ndarray:
        img = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(img)
        local_rng = _rng(cfg.seed + 17 * seed_idx)
        n_strokes = int(local_rng.integers(2, 5))
        for _ in range(n_strokes):
            mode = int(local_rng.integers(0, 3))
            w = int(local_rng.integers(max(1, size // 20), max(2, size // 12)))
            if mode == 0:
                x1 = float(local_rng.uniform(size * 0.15, size * 0.85))
                y1 = float(local_rng.uniform(size * 0.15, size * 0.85))
                x2 = float(local_rng.uniform(size * 0.15, size * 0.85))
                y2 = float(local_rng.uniform(size * 0.15, size * 0.85))
                draw.line((x1, y1, x2, y2), fill=255, width=w)
            elif mode == 1:
                cx = float(local_rng.uniform(size * 0.25, size * 0.75))
                cy = float(local_rng.uniform(size * 0.25, size * 0.75))
                rx = float(local_rng.uniform(size * 0.12, size * 0.28))
                ry = float(local_rng.uniform(size * 0.12, size * 0.28))
                draw.arc((cx - rx, cy - ry, cx + rx, cy + ry), start=0, end=300, fill=255, width=w)
            else:
                x0 = float(local_rng.uniform(size * 0.2, size * 0.6))
                y0 = float(local_rng.uniform(size * 0.2, size * 0.6))
                ww = float(local_rng.uniform(size * 0.18, size * 0.40))
                hh = float(local_rng.uniform(size * 0.18, size * 0.40))
                draw.rectangle((x0, y0, x0 + ww, y0 + hh), outline=255, width=w)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = ndimage.gaussian_filter(arr, sigma=0.6, mode="nearest")
        return arr

    labels = np.ones((n_samples, size, size), dtype=np.float32)

    rot_range = float(cfg.mnist_rotation_deg)
    for i in range(n_samples):
        if ds is not None:
            idx = int(rng.integers(0, len(ds)))
            pil_img, _ = ds[idx]
            angle = float(rng.uniform(-rot_range, rot_range))
            digit = pil_img.rotate(angle, resample=Image.BILINEAR)
            digit = digit.resize((size, size), resample=Image.BILINEAR)
            arr = np.asarray(digit, dtype=np.float32) / 255.0
        else:
            arr = _synthetic_digit_like(i)
        mask_digit = arr > 0.3

        c_digit = float(rng.uniform(1.2, 1.7))
        labels[i][mask_digit] = c_digit

        radius = float(rng.uniform(0.2, 0.4) * size * 0.50)
        cx = float(rng.uniform(radius + 2.0, size - radius - 2.0))
        cy = float(rng.uniform(radius + 2.0, size - radius - 2.0))
        mask_circle = _draw_circle_mask(size, (cx, cy), radius) > 0.5
        c_circle = float(rng.uniform(1.2, 1.7))
        labels[i][mask_circle] = c_circle
    return labels


def compute_strict_dsm_inputs_with_meta(
    labels: np.ndarray,
    metas: list[list[dict]],
    *,
    cfg: DatasetConfig,
    noise_level: float,
    seed: int,
    W_norm: float | None = None,
    phased: bool = False,
    n_per_object: int = 240,
) -> tuple[np.ndarray, float]:
    """Strict BIE Dirichlet + VIE Born-superposition path (paper Eq. 2.3)."""
    simulator = get_simulator(cfg)
    return simulator.compute_dsm_inputs_with_meta(
        labels,
        metas,
        n_incident=cfg.n_incident,
        noise_level=float(noise_level),
        seed=int(seed),
        W_norm=W_norm,
        input_scale=cfg.input_scale,
        phased=phased,
        n_per_object=n_per_object,
    )


def compute_strict_dsm_inputs(
    labels: np.ndarray,
    *,
    cfg: DatasetConfig,
    noise_level: float,
    seed: int,
    W_norm: float | None = None,
    batch_size: int = 16,
    phased: bool = False,
) -> tuple[np.ndarray, float]:
    """Run the paper-protocol Helmholtz forward + DSM index.

    Set ``phased=True`` to use Eq. (3.1) for Section 5.2.3 training; defaults
    to Eq. (3.11)/(3.12) phaseless reconstruction used throughout the rest.
    """
    simulator = get_simulator(cfg)
    return simulator.compute_dsm_inputs(
        labels,
        n_incident=cfg.n_incident,
        noise_level=float(noise_level),
        seed=int(seed),
        W_norm=W_norm,
        input_scale=cfg.input_scale,
        batch_size=batch_size,
        phased=phased,
    )


def make_dataset_tensors(
    inputs: np.ndarray,
    labels: np.ndarray,
) -> tuple[Tensor, Tensor]:
    """Convert numpy arrays into torch tensors with channel-first format."""
    x = np.asarray(inputs, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(f"inputs must have shape (N, C, H, W), got {x.shape}")
    if y.ndim != 3:
        raise ValueError(f"labels must have shape (N, H, W), got {y.shape}")
    if x.shape[0] != y.shape[0] or x.shape[2:] != y.shape[1:]:
        raise ValueError(
            f"Batch/shape mismatch between inputs {x.shape} and labels {y.shape}"
        )
    return torch.from_numpy(x), torch.from_numpy(y[:, None, :, :])


def save_dataset_cache(
    cache_path: str | Path,
    *,
    inputs: np.ndarray,
    labels: np.ndarray,
    meta: dict,
) -> Path:
    """Persist dataset with metadata for reproducible reruns."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        inputs=np.asarray(inputs, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        meta=json.dumps(meta, ensure_ascii=False),
    )
    return path


def load_dataset_cache(cache_path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load cached dataset."""
    data = np.load(Path(cache_path), allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    return data["inputs"], data["labels"], meta


class ConvBlock(nn.Module):
    """Two Conv-BN-ReLU layers."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class UNetDSMDL(nn.Module):
    """U-Net used for DSM-DL experiments (paper Fig. 1)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base_channels: int = 64,
        depth: int = 4,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.depth = int(depth)
        widths = [base_channels * (2 ** k) for k in range(depth + 1)]

        self.encoders = nn.ModuleList()
        self.encoders.append(ConvBlock(in_channels, widths[0]))
        for i in range(depth - 1):
            self.encoders.append(ConvBlock(widths[i], widths[i + 1]))
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(widths[depth - 1], widths[depth])

        self.up_blocks = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in reversed(range(depth)):
            self.up_blocks.append(
                nn.ConvTranspose2d(widths[i + 1], widths[i], kernel_size=2, stride=2)
            )
            self.decoders.append(ConvBlock(widths[i] * 2, widths[i]))
        self.head = nn.Conv2d(widths[0], out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        skips: list[Tensor] = []
        h = x
        for enc in self.encoders:
            h = enc(h)
            skips.append(h)
            h = self.pool(h)
        h = self.bottleneck(h)
        for up, dec, skip in zip(self.up_blocks, self.decoders, reversed(skips)):
            h = up(h)
            h = dec(torch.cat([h, skip], dim=1))
        return self.head(h)


def tv_loss(x: Tensor) -> Tensor:
    """Anisotropic TV regularizer."""
    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    return dx + dy


def _gaussian_window(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    return g


def _ssim_kernel(window_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    g = _gaussian_window(window_size, sigma, device, dtype)
    win_2d = g[:, None] @ g[None, :]
    win = win_2d.expand(channels, 1, window_size, window_size).contiguous()
    return win


def ssim_loss(
    x: Tensor,
    y: Tensor,
    *,
    window_size: int = 11,
    sigma: float = 1.5,
    c1: float = 0.01 ** 2,
    c2: float = 0.03 ** 2,
) -> Tensor:
    """Window SSIM loss in [0, 1] (paper Eq. 4.1 third term)."""
    if x.shape != y.shape:
        raise ValueError(f"x and y must have same shape, got {x.shape} vs {y.shape}")
    channels = x.shape[1]
    win = _ssim_kernel(window_size, sigma, channels, x.device, x.dtype)
    pad = window_size // 2
    mu_x = F.conv2d(x, win, padding=pad, groups=channels)
    mu_y = F.conv2d(y, win, padding=pad, groups=channels)
    mu_xx = F.conv2d(x * x, win, padding=pad, groups=channels)
    mu_yy = F.conv2d(y * y, win, padding=pad, groups=channels)
    mu_xy = F.conv2d(x * y, win, padding=pad, groups=channels)
    var_x = mu_xx - mu_x * mu_x
    var_y = mu_yy - mu_y * mu_y
    cov_xy = mu_xy - mu_x * mu_y
    num = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    ssim_map = num / (den + 1e-12)
    return 1.0 - ssim_map.mean()


def dsmdl_loss(
    pred: Tensor,
    target: Tensor,
    *,
    alpha_tv: float = 0.5,
    alpha_ssim: float = 0.5,
) -> Tensor:
    """Eq. (4.1) loss: MSE + α1·TV(X) + α2·(1 - SSIM(X, Y))."""
    mse = F.mse_loss(pred, target)
    return mse + alpha_tv * tv_loss(pred) + alpha_ssim * ssim_loss(pred, target)


def threshold_to_classes(pred: np.ndarray, *, t_soft: float = 0.5, t_medium: float = 2.0) -> np.ndarray:
    """Map continuous prediction to {0,1,3} for accuracy evaluation."""
    arr = np.asarray(pred, dtype=np.float32)
    out = np.ones_like(arr, dtype=np.float32)
    out[arr <= t_soft] = 0.0
    out[arr >= t_medium] = 3.0
    return out


def relative_l2_error(pred: np.ndarray, truth: np.ndarray) -> float:
    """Re(X,Y)=||X-Y||_2/||Y||_2."""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    denom = float(np.linalg.norm(t.ravel())) + 1e-12
    return float(np.linalg.norm((p - t).ravel()) / denom)


def pixel_accuracy(pred_cls: np.ndarray, truth_cls: np.ndarray) -> float:
    """Acc from Eq. (5.4)."""
    p = np.asarray(pred_cls, dtype=np.float32)
    t = np.asarray(truth_cls, dtype=np.float32)
    if p.shape != t.shape:
        raise ValueError(f"pred_cls and truth_cls must have same shape, got {p.shape} vs {t.shape}")
    return float(np.mean(np.isclose(p, t)))


class BalancedPolygonBatchSampler(torch.utils.data.Sampler):
    """5+5 medium/sound-soft batch sampler used by paper Section 5.2.1."""

    def __init__(self, medium_idx: list[int], soft_idx: list[int], *, half_batch: int, seed: int = 0):
        if half_batch <= 0:
            raise ValueError(f"half_batch must be > 0, got {half_batch}")
        if not medium_idx or not soft_idx:
            raise ValueError("Need at least one medium and one soft index")
        self.medium_idx = list(int(i) for i in medium_idx)
        self.soft_idx = list(int(i) for i in soft_idx)
        self.half_batch = int(half_batch)
        self.seed = int(seed)
        self._epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        med = self.medium_idx.copy()
        soft = self.soft_idx.copy()
        rng.shuffle(med)
        rng.shuffle(soft)
        n_pairs = min(len(med), len(soft)) // self.half_batch
        for i in range(n_pairs):
            batch = (
                med[i * self.half_batch : (i + 1) * self.half_batch]
                + soft[i * self.half_batch : (i + 1) * self.half_batch]
            )
            rng.shuffle(batch)
            yield batch
        self._epoch += 1

    def __len__(self):
        return min(len(self.medium_idx), len(self.soft_idx)) // self.half_batch


def _polygon_label_kind(label_image: np.ndarray) -> str:
    """Return 'medium' if any pixel > 1.5, else 'soft'."""
    return "medium" if float(np.max(label_image)) > 1.5 else "soft"


def make_polygon_dataloader(
    x: Tensor,
    y: Tensor,
    *,
    batch_size: int,
    val_fraction: float = 0.0,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Build polygon dataloader(s) with 5+5 medium/soft batch sampling.

    Paper §5.2.1 uses **all** training samples for training (no validation
    split). Pass ``val_fraction > 0`` only if you want to set aside an
    in-sample monitoring set; the strict reproduction script keeps it at 0.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y batch size mismatch")
    if batch_size % 2 != 0:
        raise ValueError(f"polygon batch_size must be even, got {batch_size}")
    y_np = y.detach().cpu().numpy().reshape(y.shape[0], -1)
    medium_mask = (y_np.max(axis=1) > 1.5)
    soft_mask = ~medium_mask
    n = x.shape[0]
    idx = np.arange(n)
    rng = _rng(seed)
    rng.shuffle(idx)
    if val_fraction > 0:
        n_val = max(2, int(np.floor(n * val_fraction)))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
    else:
        val_idx = idx[: max(2, batch_size)]
        train_idx = idx

    train_medium = [int(i) for i in train_idx if medium_mask[i]]
    train_soft = [int(i) for i in train_idx if soft_mask[i]]

    train_ds = TensorDataset(x, y)
    val_ds = TensorDataset(x[val_idx], y[val_idx])
    half = batch_size // 2
    sampler = BalancedPolygonBatchSampler(train_medium, train_soft, half_batch=half, seed=seed)
    train_loader = DataLoader(train_ds, batch_sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def make_dataloaders(
    x: Tensor,
    y: Tensor,
    *,
    batch_size: int,
    val_fraction: float = 0.0,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Split tensors into train/val loaders (random batches)."""
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y batch size mismatch")
    n = x.shape[0]
    n_val = max(1, int(np.floor(n * val_fraction)))
    idx = np.arange(n)
    rng = _rng(seed)
    rng.shuffle(idx)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    train_ds = TensorDataset(x[train_idx], y[train_idx])
    val_ds = TensorDataset(x[val_idx], y[val_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_unet(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainingConfig,
) -> dict:
    """Train UNet for ``cfg.epochs`` epochs with the paper-style LR schedule.

    Paper §5.2 does not mention a validation hold-out: training runs through
    all examples for 30 epochs and the **final** weights are kept. We therefore
    do not perform best-checkpoint selection; ``val_loader`` is used only as a
    cheap monitoring signal and may be empty.
    """
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.lr_step,
        gamma=cfg.lr_gamma,
    )
    history = {"train_loss": [], "val_loss": [], "lr": []}

    for _epoch in range(cfg.epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = dsmdl_loss(
                pred,
                yb,
                alpha_tv=cfg.alpha_tv,
                alpha_ssim=cfg.alpha_ssim,
            )
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = dsmdl_loss(
                    pred,
                    yb,
                    alpha_tv=cfg.alpha_tv,
                    alpha_ssim=cfg.alpha_ssim,
                )
                val_losses.append(float(loss.item()))
        scheduler.step()

        history["train_loss"].append(float(np.mean(train_losses)) if train_losses else np.nan)
        history["val_loss"].append(float(np.mean(val_losses)) if val_losses else np.nan)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

    history["final_val_loss"] = history["val_loss"][-1] if history["val_loss"] else float("nan")
    return history


def predict(model: nn.Module, x: Tensor, *, device: str = "cpu", batch_size: int = 32) -> np.ndarray:
    """Batched model inference."""
    model.eval()
    dev = torch.device(device)
    model = model.to(dev)
    out = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = x[i : i + batch_size].to(dev)
            yb = model(xb).cpu().numpy()
            out.append(yb[:, 0])
    return np.concatenate(out, axis=0)


def save_model_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    dataset_cfg: DatasetConfig,
    train_cfg: TrainingConfig,
    extra_meta: dict | None = None,
) -> Path:
    """Save reproducible model artifact."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "dataset_cfg": asdict(dataset_cfg),
        "train_cfg": asdict(train_cfg),
        "meta": extra_meta or {},
    }
    torch.save(payload, out)
    return out


def load_model_checkpoint(
    path: str | Path,
    *,
    in_channels: int,
    map_location: str = "cpu",
) -> tuple[UNetDSMDL, dict]:
    """Load model + metadata."""
    payload = torch.load(Path(path), map_location=map_location)
    model = UNetDSMDL(in_channels=in_channels)
    model.load_state_dict(payload["state_dict"])
    meta = {
        "dataset_cfg": payload.get("dataset_cfg", {}),
        "train_cfg": payload.get("train_cfg", {}),
        "meta": payload.get("meta", {}),
    }
    return model, meta


def build_labels(
    kind: Literal["polygon", "mnist", "mixed_circle"],
    *,
    n_samples: int,
    cfg: DatasetConfig,
    mnist_split: str = "train",
) -> np.ndarray:
    """Generate ground-truth labels following paper geometric distributions."""
    if kind == "polygon":
        return generate_polygon_labels(n_samples, cfg)
    if kind == "mnist":
        return generate_mnist_circle_labels(n_samples, cfg, split=mnist_split)
    if kind == "mixed_circle":
        return generate_mixed_circle_labels(n_samples, cfg)
    raise ValueError(f"Unsupported dataset kind: {kind}")


def build_labels_with_meta(
    kind: Literal["polygon", "mnist", "mixed_circle"],
    *,
    n_samples: int,
    cfg: DatasetConfig,
    mnist_split: str = "train",
) -> tuple[np.ndarray, list[list[dict]] | None]:
    """Generate labels and (when applicable) per-sample scatterer metadata."""
    if kind == "polygon":
        return generate_polygon_labels_with_meta(n_samples, cfg)
    if kind == "mixed_circle":
        return generate_mixed_circle_labels_with_meta(n_samples, cfg)
    if kind == "mnist":
        # MNIST labels are continuous medium-only ⇒ no obstacle metadata required.
        return generate_mnist_circle_labels(n_samples, cfg, split=mnist_split), None
    raise ValueError(f"Unsupported dataset kind: {kind}")


def build_dataset_with_norm(
    kind: Literal["polygon", "mnist", "mixed_circle"],
    *,
    n_samples: int,
    cfg: DatasetConfig,
    noise_level: float = 0.01,
    W_norm: float | None = None,
    mnist_split: str = "train",
    phased: bool = False,
    strict_obstacle_bie: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Strict paper-protocol dataset builder.

    When ``strict_obstacle_bie=True`` (default) sound-soft regions are
    simulated with a Dirichlet BIE (paper Eq. 2.3); medium regions use the
    exact volume integral solve. Set to ``False`` to fall back to the
    legacy complex-refractive-index VIE surrogate.
    """
    labels, metas = build_labels_with_meta(kind, n_samples=n_samples, cfg=cfg, mnist_split=mnist_split)
    if strict_obstacle_bie and metas is not None:
        inputs, W = compute_strict_dsm_inputs_with_meta(
            labels,
            metas,
            cfg=cfg,
            noise_level=noise_level,
            seed=cfg.seed,
            W_norm=W_norm,
            phased=phased,
        )
    else:
        inputs, W = compute_strict_dsm_inputs(
            labels,
            cfg=cfg,
            noise_level=noise_level,
            seed=cfg.seed,
            W_norm=W_norm,
            phased=phased,
        )
    return inputs, labels, W


def build_dataset(
    kind: Literal["polygon", "mnist", "mixed_circle"],
    *,
    n_samples: int,
    cfg: DatasetConfig,
    noise_level: float = 0.01,
    W_norm: float | None = None,
    mnist_split: str = "train",
    phased: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-level dataset builder returning (inputs, labels)."""
    inputs, labels, _ = build_dataset_with_norm(
        kind,
        n_samples=n_samples,
        cfg=cfg,
        noise_level=noise_level,
        W_norm=W_norm,
        mnist_split=mnist_split,
        phased=phased,
    )
    return inputs, labels

