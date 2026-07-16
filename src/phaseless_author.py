"""Author-style phaseless DSM-DL pipeline in pure Python.

This module ports the two reference packages under ``reference/`` into a
reusable Python API:

- ``ISP_forward/DataMnist.m`` + ``annulus_gen_rand.m`` (dataset synthesis)
- ``DSMDL_phaseless/main.py`` + ``network.py`` (indicator + U_Net3Ab training)

The goal is parity with the author implementation while keeping code modular
and testable for Notebook 06 and scripts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import io as sio
from scipy import special
import torch
import torch.nn as nn
import torch.nn.functional as F

from .phaseless_dsmdl import ssim_loss


@dataclass(frozen=True)
class AuthorForwardConfig:
    """Port of the constants used in ``DataMnist.m``."""

    lam_0: float = 0.75
    n_receivers: int = 100
    n_incident: int = 16
    mx: int = 64
    max_extent: float = 1.0
    receiver_radius: float = 4.0
    eps_0: float = 8.85e-12
    c0: float = 3.0e8

    @property
    def k0(self) -> float:
        return 2.0 * np.pi / self.lam_0

    @property
    def omega(self) -> float:
        return self.k0 * self.c0

    @property
    def eta0(self) -> float:
        return 120.0 * np.pi

    @property
    def coef(self) -> complex:
        return 1j * self.k0 * self.eta0


@dataclass(frozen=True)
class AuthorTrainingConfig:
    """Training defaults from ``DSMDL_phaseless/main.py``."""

    n_incident: int = 16
    batch_size: int = 10
    batch_number: int = 1000
    epochs: int = 30
    learning_rate: float = 1e-3
    step_size: int = 3
    gamma: float = 0.5
    noise_level: float = 0.01
    tv_weight: float = 0.05
    ssim_weight: float = 0.05
    mm_scale: float = 500.0
    train_size: int = 10000
    test_start: int = 10500
    test_stop: int = 10699
    seed: int = 0
    device: str = "cuda"
    n_channels: int = 64


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def select_incidence_indices(n_total: int, n_used: int) -> list[int]:
    """Replicate the author incidence selector ``int(i*(16/N_in))``."""
    if n_used < 1 or n_used > n_total:
        raise ValueError(f"n_used must be in [1, {n_total}], got {n_used}")
    return [int(i * (n_total / n_used)) for i in range(n_used)]


def annulus_gen_rand(
    *,
    max_extent: float,
    mx: int,
    n_circles: int,
    radius_range: tuple[float, float],
    eps_range: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Python port of ``annulus_gen_rand.m``."""
    tmp = np.linspace(-max_extent, max_extent, mx)
    x0, y0 = np.meshgrid(tmp, -tmp, indexing="xy")
    eps = np.ones_like(y0, dtype=np.float32)
    para = np.zeros((n_circles, 5), dtype=np.float32)

    for i in range(n_circles):
        radius = float(rng.uniform(radius_range[0], radius_range[1]))
        loc_min = -1.0 + radius + 0.05
        loc_max = 1.0 - radius - 0.05
        lx = float(rng.uniform(loc_min, loc_max))
        ly = float(rng.uniform(loc_min, loc_max))
        dist = np.sqrt((x0 - lx) ** 2 + (y0 - ly) ** 2)
        mask = dist < radius
        ep = float(rng.uniform(eps_range[0], eps_range[1]))
        eps[mask] = ep
        para[i] = np.array([n_circles, radius, lx, ly, ep], dtype=np.float32)
    return eps, para


def _build_receiver_ring(cfg: AuthorForwardConfig) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, cfg.n_receivers, endpoint=False)
    x = cfg.receiver_radius * np.cos(theta)
    y = cfg.receiver_radius * np.sin(theta)
    return x, y


def _mom_static_terms(cfg: AuthorForwardConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    tmp = np.linspace(-cfg.max_extent, cfg.max_extent, cfg.mx)
    xg, yg = np.meshgrid(tmp, -tmp, indexing="xy")
    pts = np.column_stack([xg.ravel(), yg.ravel()])
    step = 2.0 * cfg.max_extent / max(cfg.mx - 1, 1)
    cell_area = step ** 2
    a_eqv = np.sqrt(cell_area / np.pi)
    return xg, yg, pts, a_eqv


def generate_author_forward_dataset(
    contrast: np.ndarray,
    *,
    cfg: AuthorForwardConfig = AuthorForwardConfig(),
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Pure Python forward-data generator inspired by ``DataMnist.m``.

    Notes:
    - This implementation keeps a fixed 64x64 grid for all samples so the
      resulting ``R_mat`` has stable shape `(N_rec, 4096)`.
    - It is intentionally faithful in formulas but optimized for readability.
    """
    rng = _rng(seed)
    contrast_arr = np.asarray(contrast, dtype=np.float32)
    if contrast_arr.ndim != 3:
        raise ValueError(f"contrast must have shape (N, mx, mx), got {contrast_arr.shape}")
    if contrast_arr.shape[1:] != (cfg.mx, cfg.mx):
        raise ValueError(f"contrast spatial shape must be {(cfg.mx, cfg.mx)}")

    n_samples = contrast_arr.shape[0]
    xg, yg, pts, a_eqv = _mom_static_terms(cfg)
    n_cells = pts.shape[0]
    x0 = pts[:, 0]
    y0 = pts[:, 1]
    step = 2.0 * cfg.max_extent / max(cfg.mx - 1, 1)
    cell_area = step ** 2

    xr, yr = _build_receiver_ring(cfg)
    theta_inc = np.linspace(0.0, 2.0 * np.pi, cfg.n_incident, endpoint=False)
    kx = cfg.k0 * np.cos(theta_inc)
    ky = cfg.k0 * np.sin(theta_inc)
    e_inc_grid = np.exp(1j * (x0[:, None] * kx[None, :] + y0[:, None] * ky[None, :]))
    e_inc_recv = np.exp(1j * (xr[:, None] * kx[None, :] + yr[:, None] * ky[None, :]))

    xx0, xx1 = np.meshgrid(x0, x0, indexing="xy")
    yy0, yy1 = np.meshgrid(y0, y0, indexing="xy")
    dist = np.sqrt((xx0 - xx1) ** 2 + (yy0 - yy1) ** 2)
    dist = np.maximum(dist, 1e-12)
    i1 = (1j / 4.0) * special.hankel1(0, cfg.k0 * dist)
    np.fill_diagonal(i1, 0.0 + 0.0j)
    phi = cfg.coef * i1
    i2 = (1j / 4.0) * (
        2.0 / (cfg.k0 * a_eqv) * special.hankel1(1, cfg.k0 * a_eqv)
        + 4.0j / (cfg.k0 ** 2 * cell_area)
    )
    phi += (cfg.coef * i2) * np.eye(n_cells, dtype=np.complex128)

    x0_tmp, xr_tmp = np.meshgrid(x0, xr, indexing="xy")
    y0_tmp, yr_tmp = np.meshgrid(y0, yr, indexing="xy")
    rho_rx = np.sqrt((x0_tmp - xr_tmp) ** 2 + (y0_tmp - yr_tmp) ** 2)
    r_mat = cfg.coef * (1j / 4.0) * special.hankel1(0, cfg.k0 * np.maximum(rho_rx, 1e-12))

    e_s = np.zeros((cfg.n_receivers, cfg.n_incident, n_samples), dtype=np.complex64)
    e_i = np.zeros_like(e_s)
    out_contrast = np.zeros_like(contrast_arr, dtype=np.float32)

    eye = np.eye(n_cells, dtype=np.complex128)
    for i in range(n_samples):
        eps = contrast_arr[i].copy()
        # Author script randomizes object amplitudes and adds one annulus.
        eps = np.where(eps > 1.2, 1.2 + 0.5 * rng.random(), eps)
        eps2, _ = annulus_gen_rand(
            max_extent=cfg.max_extent,
            mx=cfg.mx,
            n_circles=1,
            radius_range=(0.2, 0.4),
            eps_range=(1.2, 1.7),
            rng=rng,
        )
        eps = np.where(eps2 > 1.1, 1.2 + 0.5 * rng.random(), eps)
        out_contrast[i] = eps.astype(np.float32)

        xi = (-1j * cfg.omega * (eps.reshape(-1) - 1.0) * cfg.eps_0 * cell_area).astype(np.complex128)
        a_mat = eye - phi * xi[None, :]
        e_tot = np.linalg.solve(a_mat, e_inc_grid)
        e_s_i = r_mat @ (xi[:, None] * e_tot)
        e_tot_rx = e_s_i + e_inc_recv
        e_s[:, :, i] = e_s_i.astype(np.complex64)
        e_i[:, :, i] = (e_tot_rx - e_s_i).astype(np.complex64)

    return {
        "Contrast": out_contrast.astype(np.float32),
        "E_i": e_i.astype(np.complex64),
        "E_s": e_s.astype(np.complex64),
        "R_mat": r_mat.astype(np.complex64),
    }


def load_author_mat_dataset(path: str | Path) -> dict[str, np.ndarray]:
    """Load the author MAT dataset and normalize key arrays."""
    data = sio.loadmat(str(path))
    required = {"Contrast", "E_i", "E_s", "R_mat"}
    missing = required.difference(data)
    if missing:
        raise KeyError(f"Missing keys in MAT file: {sorted(missing)}")
    return {
        "Contrast": np.asarray(data["Contrast"], dtype=np.float32),
        "E_i": np.asarray(data["E_i"], dtype=np.complex64),
        "E_s": np.asarray(data["E_s"], dtype=np.complex64),
        "R_mat": np.asarray(data["R_mat"], dtype=np.complex64),
    }


def compute_author_inputs_from_fields(
    *,
    e_i: np.ndarray,
    e_s: np.ndarray,
    r_mat: np.ndarray,
    n_incident: int = 16,
    noise_level: float = 0.01,
    mm_scale: float = 500.0,
    seed: int = 0,
) -> dict[str, np.ndarray | list[int] | float]:
    """Port the indicator pipeline from ``main.py`` into reusable tensors."""
    e_i_c = np.asarray(e_i, dtype=np.complex64)
    e_s_c = np.asarray(e_s, dtype=np.complex64)
    r = np.asarray(r_mat, dtype=np.complex64)
    if e_i_c.shape != e_s_c.shape:
        raise ValueError(f"E_i/E_s shape mismatch: {e_i_c.shape} vs {e_s_c.shape}")
    n_re, n_total_inc, n_samples = e_i_c.shape
    idx = select_incidence_indices(n_total_inc, n_incident)
    e_i_sel = e_i_c[:, idx, :]
    e_s_sel = e_s_c[:, idx, :]

    # Author normalization of Green matrix.
    norm_gs = np.abs(np.mean(r * r) ** 0.5)
    gs = r / (norm_gs + 1e-12)

    e_t = np.abs(e_i_sel + e_s_sel).astype(np.float32)
    delta = (np.abs(e_t) ** 2 - np.abs(e_i_sel) ** 2) / np.where(np.abs(e_i_sel) < 1e-12, 1e-12 + 0j, e_i_sel)

    rng = _rng(seed)
    e_t_noise = np.empty_like(e_t)
    coeff_scale = (n_incident * n_re) ** 0.5
    for i in range(n_samples):
        coeff = noise_level * np.linalg.norm(e_t[:, :, i]) / coeff_scale
        e_t_noise[:, :, i] = coeff * rng.standard_normal((n_re, n_incident)).astype(np.float32) + e_t[:, :, i]
    delta_noise = (e_t_noise ** 2 - np.abs(e_i_sel) ** 2) / np.where(np.abs(e_i_sel) < 1e-12, 1e-12 + 0j, e_i_sel)

    def _reshape_indicator(delta_field: np.ndarray) -> np.ndarray:
        # Exactly mirrors: torch.matmul(torch.t(delta.reshape(N_re,-1)), Gs)
        merged = delta_field.reshape(n_re, -1).T  # (n_incident*n_samples, n_re)
        ind = merged @ gs  # (n_incident*n_samples, 4096)
        ind = ind.reshape(n_incident, n_samples, 64, 64).transpose(1, 0, 3, 2)
        return np.abs(ind).astype(np.float32)

    ind_clean = _reshape_indicator(delta)
    ind_noise = _reshape_indicator(delta_noise)
    inputs_clean = 2.0 * ind_clean / float(mm_scale)
    inputs_noisy = 2.0 * ind_noise / float(mm_scale)
    return {
        "indices": idx,
        "norm_gs": float(norm_gs),
        "inputs_clean": inputs_clean,
        "inputs_noisy": inputs_noisy,
        "delta": delta,
        "delta_noise": delta_noise,
        "mm_scale": float(mm_scale),
    }


def compute_author_inputs_from_mat(
    mat_data: dict[str, np.ndarray],
    *,
    n_incident: int = 16,
    noise_level: float = 0.01,
    mm_scale: float = 500.0,
    seed: int = 0,
) -> dict[str, np.ndarray | list[int] | float]:
    return compute_author_inputs_from_fields(
        e_i=mat_data["E_i"],
        e_s=mat_data["E_s"],
        r_mat=mat_data["R_mat"],
        n_incident=n_incident,
        noise_level=noise_level,
        mm_scale=mm_scale,
        seed=seed,
    )


class _AuthorConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _AuthorUpConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class AuthorUNet3Ab(nn.Module):
    """Python port of ``network.py::U_Net3Ab``."""

    def __init__(self, img_ch: int = 16, output_ch: int = 1, n_ch: int = 64):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = _AuthorConvBlock(img_ch, n_ch)
        self.conv2 = _AuthorConvBlock(n_ch, 2 * n_ch)
        self.conv3 = _AuthorConvBlock(2 * n_ch, 4 * n_ch)
        self.conv4 = _AuthorConvBlock(4 * n_ch, 8 * n_ch)

        self.up43 = _AuthorUpConv(8 * n_ch, 4 * n_ch)
        self.conv3d = _AuthorConvBlock(8 * n_ch, 4 * n_ch)
        self.up32 = _AuthorUpConv(4 * n_ch, 2 * n_ch)
        self.conv2d = _AuthorConvBlock(4 * n_ch, 2 * n_ch)
        self.up21 = _AuthorUpConv(2 * n_ch, n_ch)
        self.conv1d = _AuthorConvBlock(2 * n_ch, n_ch)
        self.conv_1 = nn.Conv2d(n_ch, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.maxpool(x1)
        x2 = self.conv2(x2)
        x3 = self.maxpool(x2)
        x3 = self.conv3(x3)
        x4 = self.maxpool(x3)
        x4 = self.conv4(x4)

        d3 = self.up43(x4)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.conv3d(d3)
        d2 = self.up32(d3)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.conv2d(d2)
        d1 = self.up21(d2)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.conv1d(d1)
        d1 = self.conv_1(d1)
        return torch.relu(d1)


def _author_tv(pred: torch.Tensor) -> torch.Tensor:
    g1 = pred[:, :, 0:-2, :] - pred[:, :, 1:-1, :]
    g2 = pred[:, :, :, 0:-2] - pred[:, :, :, 1:-1]
    return torch.mean(torch.abs(g1)) + torch.mean(torch.abs(g2))


def author_relative_l2(pred: torch.Tensor, truth: torch.Tensor) -> float:
    num = torch.linalg.norm((pred - truth).reshape(pred.shape[0], -1), dim=1)
    den = torch.linalg.norm(truth.reshape(truth.shape[0], -1), dim=1) + 1e-12
    return float(torch.mean(num / den).item())


def _author_loss(
    pred: torch.Tensor,
    truth: torch.Tensor,
    *,
    tv_weight: float,
    ssim_weight: float,
) -> tuple[torch.Tensor, float, float]:
    mse = F.mse_loss(pred, truth)
    tv = _author_tv(pred)
    # ssim_loss returns 1 - SSIM.
    one_minus_ssim = ssim_loss(pred, truth)
    loss = mse + tv_weight * tv + ssim_weight * one_minus_ssim
    return loss, float(tv.item()), float((1.0 - one_minus_ssim.item()))


def train_author_unet3ab(
    *,
    inputs_noisy: np.ndarray,
    contrast: np.ndarray,
    cfg: AuthorTrainingConfig = AuthorTrainingConfig(),
) -> dict[str, Any]:
    """Train the author U_Net3Ab using the original batching protocol."""
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)

    x_np = np.asarray(inputs_noisy, dtype=np.float32)
    y_np = np.asarray(contrast, dtype=np.float32)
    if x_np.ndim != 4 or y_np.ndim != 3:
        raise ValueError(f"Expected x=(N,C,H,W), y=(N,H,W), got {x_np.shape}, {y_np.shape}")
    if x_np.shape[0] != y_np.shape[0]:
        raise ValueError("inputs/contrast batch mismatch")
    if x_np.shape[1] != cfg.n_incident:
        raise ValueError(f"inputs channels {x_np.shape[1]} != cfg.n_incident {cfg.n_incident}")

    device = torch.device("cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = AuthorUNet3Ab(img_ch=cfg.n_incident, output_ch=1, n_ch=cfg.n_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.step_size, gamma=cfg.gamma)

    x_t = torch.from_numpy(x_np)
    y_t = torch.from_numpy(y_np[:, None, :, :])
    n_total = x_t.shape[0]
    max_train = min(cfg.train_size, n_total)
    test_slice = slice(min(cfg.test_start, n_total), min(cfg.test_stop, n_total))
    if test_slice.start >= test_slice.stop:
        test_slice = slice(max(0, n_total - 200), n_total)

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_test = float("inf")
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_rel_l2": [],
        "test_rel_l2": [],
        "tv": [],
        "ssim": [],
        "lr": [],
    }

    for _epoch in range(cfg.epochs):
        model.train()
        epoch_losses: list[float] = []
        epoch_tv: list[float] = []
        epoch_ssim: list[float] = []
        n_steps = min(cfg.batch_number, max_train // cfg.batch_size)
        for j in range(n_steps):
            s = j * cfg.batch_size
            e = s + cfg.batch_size
            xb = x_t[s:e].to(device)
            yb = y_t[s:e].to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss, tv_val, ssim_val = _author_loss(
                pred,
                yb,
                tv_weight=cfg.tv_weight,
                ssim_weight=cfg.ssim_weight,
            )
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.item()))
            epoch_tv.append(tv_val)
            epoch_ssim.append(ssim_val)

        scheduler.step()
        model.eval()
        with torch.no_grad():
            # Author script prints train error on the last batch in each epoch.
            xb_train = x_t[max(0, max_train - cfg.batch_size):max_train].to(device)
            yb_train = y_t[max(0, max_train - cfg.batch_size):max_train].to(device)
            pred_train = model(xb_train)
            train_rel = author_relative_l2(pred_train, yb_train)

            xb_test = x_t[test_slice].to(device)
            yb_test = y_t[test_slice].to(device)
            pred_test = model(xb_test)
            test_rel = author_relative_l2(pred_test, yb_test)
            if test_rel < best_test:
                best_test = test_rel
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        history["train_loss"].append(float(np.mean(epoch_losses)) if epoch_losses else float("nan"))
        history["train_rel_l2"].append(train_rel)
        history["test_rel_l2"].append(test_rel)
        history["tv"].append(float(np.mean(epoch_tv)) if epoch_tv else float("nan"))
        history["ssim"].append(float(np.mean(epoch_ssim)) if epoch_ssim else float("nan"))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

    model.load_state_dict(best_state)
    return {
        "model": model,
        "history": history,
        "best_test_rel_l2": best_test,
        "device": str(device),
        "config": asdict(cfg),
    }


def save_author_checkpoint(
    path: str | Path,
    *,
    model: AuthorUNet3Ab,
    train_cfg: AuthorTrainingConfig,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "train_cfg": asdict(train_cfg),
        "meta": extra_meta or {},
    }
    torch.save(payload, out)
    return out


def load_author_checkpoint(
    path: str | Path,
    *,
    map_location: str = "cpu",
) -> tuple[AuthorUNet3Ab, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location)
    train_cfg = payload.get("train_cfg", {})
    n_inc = int(train_cfg.get("n_incident", 16))
    n_ch = int(train_cfg.get("n_channels", 64))
    model = AuthorUNet3Ab(img_ch=n_inc, n_ch=n_ch)
    model.load_state_dict(payload["state_dict"])
    return model, {
        "train_cfg": train_cfg,
        "meta": payload.get("meta", {}),
    }


def save_author_summary(path: str | Path, summary: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
