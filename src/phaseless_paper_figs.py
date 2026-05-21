"""Shared helpers that build the paper Section 5.1/5.2 figure grids.

This module is imported by both ``scripts/run_phaseless_full_repro.py`` and
``notebooks/06_phaseless_dsmdl.ipynb`` so the figure layouts stay aligned.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from .plot_style import save_figure
from .phaseless_dsmdl import (
    DatasetConfig,
    build_labels,
    compute_strict_dsm_inputs,
    load_model_checkpoint,
    make_dataset_tensors,
    predict,
    threshold_to_classes,
)


# ---------------- deterministic OOD label sets ----------------

def draw_chinese_like_set(size: int = 64) -> np.ndarray:
    """Five deterministic character-like labels (paper Section 5.2.2.2)."""
    yy, xx = np.mgrid[0:size, 0:size]
    c_value = 1.5
    out: list[np.ndarray] = []

    m1 = np.ones((size, size), dtype=np.float32)
    m1[np.abs(xx - size * 0.5) < size * 0.03] = c_value
    m1[np.abs(yy - size * 0.5) < size * 0.03] = c_value
    m1[(xx > size * 0.2) & (xx < size * 0.8) & (np.abs(yy - size * 0.2) < size * 0.02)] = c_value
    out.append(m1)

    m2 = np.ones((size, size), dtype=np.float32)
    m2[np.abs(xx - size * 0.3) < size * 0.025] = c_value
    m2[np.abs(xx - size * 0.7) < size * 0.025] = c_value
    for frac in (0.25, 0.45, 0.65, 0.82):
        m2[(np.abs(yy - size * frac) < size * 0.02) & (xx > size * 0.3) & (xx < size * 0.7)] = c_value
    out.append(m2)

    m3 = np.ones((size, size), dtype=np.float32)
    for frac in (0.28, 0.50, 0.72):
        m3[(np.abs(yy - size * frac) < size * 0.025) & (xx > size * 0.2) & (xx < size * 0.8)] = c_value
    m3[(np.abs(xx - size * 0.5) < size * 0.025) & (yy > size * 0.2) & (yy < size * 0.8)] = c_value
    out.append(m3)

    m4 = np.ones((size, size), dtype=np.float32)
    diag1 = np.abs((yy - size * 0.2) - (xx - size * 0.2)) < size * 0.03
    diag2 = np.abs((yy - size * 0.8) + (xx - size * 0.2) - size * 0.6) < size * 0.03
    m4[diag1 | diag2] = c_value
    out.append(m4)

    m5 = np.ones((size, size), dtype=np.float32)
    frame = (
        ((np.abs(xx - size * 0.2) < size * 0.02) & (yy > size * 0.2) & (yy < size * 0.8))
        | ((np.abs(xx - size * 0.8) < size * 0.02) & (yy > size * 0.2) & (yy < size * 0.8))
        | ((np.abs(yy - size * 0.2) < size * 0.02) & (xx > size * 0.2) & (xx < size * 0.8))
        | ((np.abs(yy - size * 0.8) < size * 0.02) & (xx > size * 0.2) & (xx < size * 0.8))
    )
    m5[frame] = c_value
    m5[((xx - size * 0.5) ** 2 + (yy - size * 0.5) ** 2) < (size * 0.10) ** 2] = c_value
    out.append(m5)
    return np.stack(out, axis=0)


def draw_austria_like_set(size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Paper Section 5.2.2.3 -- two Austria-style profiles."""
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.sqrt((xx - size * 0.5) ** 2 + (yy - size * 0.5) ** 2)

    def austria(coef_circles: float) -> np.ndarray:
        base = np.ones((size, size), dtype=np.float32)
        ring = (r < size * 0.34) & (r > size * 0.26)
        base[ring] = 1.5
        left = ((xx - size * 0.33) ** 2 + (yy - size * 0.48) ** 2) < (size * 0.13) ** 2
        right = ((xx - size * 0.67) ** 2 + (yy - size * 0.48) ** 2) < (size * 0.13) ** 2
        base[left] = coef_circles
        base[right] = coef_circles
        return base

    return austria(1.5)[None, ...], austria(2.0)[None, ...]


# ---------------- prediction + grid helpers ----------------

def predict_on_labels(
    model,
    labels: np.ndarray,
    *,
    dcfg_train: DatasetConfig,
    W_norm: float,
    seed_test: int,
    delta: float,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Synthesize phaseless DSM input on shared labels then run the U-Net."""
    dcfg_test = replace(dcfg_train, seed=seed_test)
    inputs, _ = compute_strict_dsm_inputs(
        labels.astype(np.float32),
        cfg=dcfg_test,
        noise_level=delta,
        seed=seed_test,
        W_norm=W_norm,
    )
    x_t, _ = make_dataset_tensors(inputs, labels)
    return predict(model, x_t, device=device, batch_size=batch_size)


def render_recon_grid(
    *,
    title: str,
    truth_labels: np.ndarray,
    row_predictions: dict[str, np.ndarray],
    row_order: list[str],
    truth_vmin: float,
    truth_vmax: float,
    pred_vmin: float,
    pred_vmax: float,
    pred_postprocess: Callable[[np.ndarray], np.ndarray] | None = None,
) -> plt.Figure:
    n_cols = truth_labels.shape[0]
    n_rows = 1 + len(row_order)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.6 * n_rows), squeeze=False)
    truth_im = None
    pred_im = None
    for c in range(n_cols):
        ax = axes[0, c]
        truth_im = ax.imshow(truth_labels[c], origin="lower", extent=(-1, 1, -1, 1), cmap="viridis", vmin=truth_vmin, vmax=truth_vmax)
        ax.set_title(f"sample {c + 1}", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
    for r, label in enumerate(row_order):
        preds = row_predictions[label]
        for c in range(n_cols):
            ax = axes[r + 1, c]
            img = preds[c]
            if pred_postprocess is not None:
                img = pred_postprocess(img)
            pred_im = ax.imshow(img, origin="lower", extent=(-1, 1, -1, 1), cmap="viridis", vmin=pred_vmin, vmax=pred_vmax)
            ax.set_aspect("equal")
            ax.set_xticks([-1, 0, 1])
            ax.set_yticks([-1, 0, 1])
    for r, lbl in enumerate(["truth"] + row_order):
        axes[r, 0].set_ylabel(lbl, fontsize=10, rotation=0, ha="right", va="center")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0.02, 0, 0.92, 0.97])
    if truth_im is not None:
        cax_truth = fig.add_axes([0.93, 0.55, 0.012, 0.32])
        fig.colorbar(truth_im, cax=cax_truth, label="truth")
    if pred_im is not None and len(row_order) > 0:
        cax_pred = fig.add_axes([0.93, 0.10, 0.012, 0.40])
        fig.colorbar(pred_im, cax=cax_pred, label="recon")
    return fig


# ---------------- per-figure assemblers ----------------

def _polygon_fig6_inputs(seed: int, device: str, mnist_download: bool) -> np.ndarray:
    pool_cfg = DatasetConfig(image_size=64, n_incident=1, seed=seed + 7000, device=device, mnist_download=mnist_download)
    pool_labels = build_labels("polygon", n_samples=40, cfg=pool_cfg)
    medium_idx = [i for i in range(40) if float(np.max(pool_labels[i])) > 1.5]
    soft_idx = [i for i in range(40) if float(np.max(pool_labels[i])) < 1.5]
    return pool_labels[medium_idx[:3] + soft_idx[:2]]


def assemble_polygon_fig6(models: dict[int, dict], *, seed: int, device: str, mnist_download: bool) -> plt.Figure:
    truth = _polygon_fig6_inputs(seed, device, mnist_download)
    row_predictions: dict[str, np.ndarray] = {}
    row_order: list[str] = []
    for ni in (1, 4):
        case = models[ni]
        for delta in (0.02, 0.10):
            preds = predict_on_labels(
                case["model"], truth,
                dcfg_train=case["dataset_cfg_train"],
                W_norm=case["W_norm"],
                seed_test=seed + 7100 + ni + int(delta * 100),
                delta=delta,
                device=device,
                batch_size=10,
            )
            label = f"Ni={ni}, delta={int(delta * 100)}%"
            row_predictions[label] = preds
            row_order.append(label)
    return render_recon_grid(
        title="Fig.6 polygon dataset reconstructions",
        truth_labels=truth,
        row_predictions=row_predictions,
        row_order=row_order,
        truth_vmin=0.0, truth_vmax=3.0,
        pred_vmin=0.0, pred_vmax=3.0,
        pred_postprocess=threshold_to_classes,
    )


def assemble_mnist_fig7(models: dict[int, dict], *, seed: int, device: str, mnist_download: bool) -> plt.Figure:
    pool_cfg = DatasetConfig(image_size=64, n_incident=4, seed=seed + 8000, device=device, mnist_download=mnist_download)
    pool_labels = build_labels("mnist", n_samples=20, cfg=pool_cfg, mnist_split="test")
    truth = pool_labels[:5]
    row_predictions: dict[str, np.ndarray] = {}
    row_order: list[str] = []
    for ni in (4, 16):
        case = models[ni]
        for delta in (0.05, 0.10):
            preds = predict_on_labels(
                case["model"], truth,
                dcfg_train=case["dataset_cfg_train"],
                W_norm=case["W_norm"],
                seed_test=seed + 8100 + ni + int(delta * 100),
                delta=delta,
                device=device,
                batch_size=10,
            )
            label = f"Ni={ni}, delta={int(delta * 100)}%"
            row_predictions[label] = preds
            row_order.append(label)
    return render_recon_grid(
        title="Fig.7 MNIST dataset reconstructions",
        truth_labels=truth,
        row_predictions=row_predictions,
        row_order=row_order,
        truth_vmin=1.0, truth_vmax=1.7,
        pred_vmin=1.0, pred_vmax=1.7,
    )


def assemble_ood_grid(
    title: str,
    truth_labels: np.ndarray,
    models: dict[int, dict],
    *,
    seed: int,
    device: str,
) -> plt.Figure:
    row_predictions: dict[str, np.ndarray] = {}
    row_order: list[str] = []
    for ni in (4, 16):
        case = models[ni]
        for delta in (0.05, 0.10):
            preds = predict_on_labels(
                case["model"], truth_labels,
                dcfg_train=case["dataset_cfg_train"],
                W_norm=case["W_norm"],
                seed_test=seed + ni + int(delta * 100),
                delta=delta,
                device=device,
                batch_size=10,
            )
            label = f"Ni={ni}, delta={int(delta * 100)}%"
            row_predictions[label] = preds
            row_order.append(label)
    return render_recon_grid(
        title=title,
        truth_labels=truth_labels,
        row_predictions=row_predictions,
        row_order=row_order,
        truth_vmin=1.0, truth_vmax=2.0,
        pred_vmin=1.0, pred_vmax=2.0,
    )


def assemble_mixed_fig10(case: dict, *, seed: int, device: str, mnist_download: bool) -> plt.Figure:
    pool_cfg = DatasetConfig(
        image_size=64, n_incident=10, seed=seed + 10000, device=device,
        mnist_download=mnist_download, receiver_radius=8.0, n_receivers=180,
    )
    pool_labels = build_labels("mixed_circle", n_samples=12, cfg=pool_cfg)
    truth = pool_labels[:5]
    row_predictions: dict[str, np.ndarray] = {}
    row_order: list[str] = []
    for delta in (0.05, 0.10):
        preds = predict_on_labels(
            case["model"], truth,
            dcfg_train=case["dataset_cfg_train"],
            W_norm=case["W_norm"],
            seed_test=seed + 10100 + int(delta * 100),
            delta=delta,
            device=device,
            batch_size=20,
        )
        label = f"Ni=10, delta={int(delta * 100)}%"
        row_predictions[label] = preds
        row_order.append(label)
    return render_recon_grid(
        title="Fig.10 mixed-circle phaseless reconstructions (phased-trained)",
        truth_labels=truth,
        row_predictions=row_predictions,
        row_order=row_order,
        truth_vmin=0.0, truth_vmax=3.0,
        pred_vmin=0.0, pred_vmax=3.0,
    )


# ---------------- checkpoint loader ----------------

def load_case_from_checkpoint(path: str | Path, *, in_channels: int, device: str = "cpu") -> dict:
    """Reconstruct an evaluation-ready case dict from a saved checkpoint."""
    model, meta = load_model_checkpoint(path, in_channels=in_channels, map_location=device)
    model.to(device).eval()
    dataset_cfg_raw = meta.get("dataset_cfg", {})
    extra = meta.get("meta", {})
    dataset_cfg_raw = {**dataset_cfg_raw, "device": device}
    dcfg = DatasetConfig(**dataset_cfg_raw)
    return {
        "model": model,
        "dataset_cfg_train": dcfg,
        "W_norm": float(extra.get("W_norm", 1.0)),
        "phased_training": bool(extra.get("phased_training", False)),
    }


def render_and_save_all(
    *,
    ckpt_dir: Path,
    out_fig: Path,
    seed: int,
    device: str,
    mnist_download: bool = False,
) -> dict[str, Path]:
    """Render Fig.6-10 from checkpoints into out_fig; return saved paths."""
    ckpt_dir = Path(ckpt_dir)
    out_fig = Path(out_fig)
    out_fig.mkdir(parents=True, exist_ok=True)

    polygon_cases = {
        ni: load_case_from_checkpoint(ckpt_dir / f"polygon_Ni{ni}.pt", in_channels=ni, device=device)
        for ni in (1, 4)
    }
    mnist_cases = {
        ni: load_case_from_checkpoint(ckpt_dir / f"mnist_Ni{ni}.pt", in_channels=ni, device=device)
        for ni in (4, 16)
    }
    mixed_case = load_case_from_checkpoint(ckpt_dir / "mixed_circle_Ni10.pt", in_channels=10, device=device)

    saved: dict[str, Path] = {}

    fig = assemble_polygon_fig6(polygon_cases, seed=seed, device=device, mnist_download=mnist_download)
    saved["fig6"] = save_figure(fig, out_fig / "fig6_polygon_recon.png", dpi=160)
    plt.close(fig)

    fig = assemble_mnist_fig7(mnist_cases, seed=seed, device=device, mnist_download=mnist_download)
    saved["fig7"] = save_figure(fig, out_fig / "fig7_mnist_recon.png", dpi=160)
    plt.close(fig)

    fig = assemble_ood_grid(
        "Fig.8 Chinese-character OOD reconstructions",
        draw_chinese_like_set(64), mnist_cases, seed=seed + 8500, device=device,
    )
    saved["fig8"] = save_figure(fig, out_fig / "fig8_chinese_recon.png", dpi=160)
    plt.close(fig)

    austria_1, austria_2 = draw_austria_like_set(64)
    fig = assemble_ood_grid(
        "Fig.9 Austria ring OOD reconstructions",
        np.concatenate([austria_1, austria_2], axis=0), mnist_cases, seed=seed + 8800, device=device,
    )
    saved["fig9"] = save_figure(fig, out_fig / "fig9_austria_recon.png", dpi=160)
    plt.close(fig)

    fig = assemble_mixed_fig10(mixed_case, seed=seed, device=device, mnist_download=mnist_download)
    saved["fig10"] = save_figure(fig, out_fig / "fig10_mixed_recon.png", dpi=160)
    plt.close(fig)

    return saved
