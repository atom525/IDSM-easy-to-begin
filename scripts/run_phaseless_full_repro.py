"""Strict paper-protocol reproduction runner for notebook6 / arXiv:2403.02584.

This script executes paper-scale experiments and writes:
  - results/phaseless/full_summary.json
  - results/phaseless/full_comparison.md
  - figures/06_phaseless/full_*.png

Key strict alignments vs the previous proxy-based version:
  * DSM-DL inputs come from a Helmholtz forward + phaseless DSM indicator
    (Eq. 3.11-3.12), not from label-derived smoothing.
  * Training uses noise_level=0.01 (paper Section 5.2); test inputs are
    re-generated with the requested delta per Eq. (5.1).
  * MNIST uses the official torchvision train/test splits (or a deterministic
    by-description fallback when download is unavailable, with the same
    geometric distribution as the paper).
  * Mixed-circle uses receiver radius 8 with 180 receivers (paper §5.2.3).
  * Loss is Eq. (4.1) MSE + 0.5*TV + 0.5*(1-SSIM), no foreground re-weighting.
  * U-Net output is unconstrained; threshold for accuracy follows paper cutoffs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import time
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(ROOT))

from src.plot_style import apply_idsm_plot_style, contrast_norm, save_figure
from src.phaseless_scattering import (
    PhaselessDSMConfig,
    peak_location_error,
    run_example_dsm,
    run_example_dsm_paper,
    topk_peak_hit_rate,
)
from src.phaseless_dsmdl import (
    DatasetConfig,
    TrainingConfig,
    UNetDSMDL,
    build_dataset,
    build_dataset_with_norm,
    build_labels,
    compute_strict_dsm_inputs,
    make_dataset_tensors,
    make_dataloaders,
    make_polygon_dataloader,
    pixel_accuracy,
    predict,
    relative_l2_error,
    save_model_checkpoint,
    threshold_to_classes,
    train_unet,
)


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cleanup(*objs: object) -> None:
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------- OOD label constructors (by-description) ----------------

def _draw_chinese_like_set(size: int = 64) -> np.ndarray:
    """Five deterministic character-like labels (paper Section 5.2.2.2)."""
    yy, xx = np.mgrid[0:size, 0:size]
    out: list[np.ndarray] = []
    c_value = 1.5

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


def _draw_austria_like_set(size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Two Austria-style profiles (paper Section 5.2.2.3)."""
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


# ---------------- Training case helper ----------------

def _train_case(
    *,
    dataset_kind: str,
    n_incident: int,
    train_n: int,
    batch_size: int,
    epochs: int,
    seed: int,
    device: str,
    extra_cfg_kwargs: dict | None = None,
    mnist_download: bool = False,
    base_channels: int = 64,
    phased_training: bool = False,
) -> dict:
    cfg_kwargs = dict(
        image_size=64,
        n_incident=n_incident,
        seed=seed,
        mnist_download=mnist_download,
        device=device,
    )
    cfg_kwargs.update(extra_cfg_kwargs or {})
    dcfg_train = DatasetConfig(**cfg_kwargs)

    x_train_np, y_train_np, W_norm = build_dataset_with_norm(
        dataset_kind,
        n_samples=train_n,
        cfg=dcfg_train,
        noise_level=0.01,
        mnist_split="train",
        phased=phased_training,
    )
    x_train, y_train = make_dataset_tensors(x_train_np, y_train_np)
    if dataset_kind == "polygon":
        train_loader, val_loader = make_polygon_dataloader(
            x_train, y_train, batch_size=batch_size, val_fraction=0.1, seed=seed
        )
    else:
        train_loader, val_loader = make_dataloaders(
            x_train, y_train, batch_size=batch_size, val_fraction=0.1, seed=seed
        )
    model = UNetDSMDL(in_channels=n_incident, out_channels=1, base_channels=base_channels)
    tcfg = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=1e-3,
        lr_step=3,
        lr_gamma=0.5,
        alpha_tv=0.5,
        alpha_ssim=0.5,
        device=device,
        seed=seed,
    )
    t0 = time.time()
    hist = train_unet(model, train_loader, val_loader, tcfg)
    runtime = time.time() - t0
    return {
        "dataset_cfg_train": dcfg_train,
        "train_cfg": asdict(tcfg),
        "W_norm": W_norm,
        "model": model,
        "history": hist,
        "runtime_seconds": runtime,
        "phased_training": bool(phased_training),
    }


def _eval_polygon(model, dcfg_train: DatasetConfig, W_norm: float, *, test_n: int, batch_size: int, device: str, seed_test: int, delta: float, ni: int) -> float:
    dcfg_test = replace(dcfg_train, seed=seed_test)
    x_np, y_np = build_dataset(
        "polygon",
        n_samples=test_n,
        cfg=dcfg_test,
        noise_level=delta,
        W_norm=W_norm,
        mnist_split="test",
    )
    x_t, _ = make_dataset_tensors(x_np, y_np)
    pred = predict(model, x_t, device=device, batch_size=batch_size)
    return pixel_accuracy(threshold_to_classes(pred), y_np)


def _eval_mnist_split(model, dcfg_train: DatasetConfig, W_norm: float, *, test_n: int, batch_size: int, device: str, seed_test: int, delta: float) -> float:
    dcfg_test = replace(dcfg_train, seed=seed_test)
    x_np, y_np = build_dataset(
        "mnist",
        n_samples=test_n,
        cfg=dcfg_test,
        noise_level=delta,
        W_norm=W_norm,
        mnist_split="test",
    )
    x_t, _ = make_dataset_tensors(x_np, y_np)
    pred = predict(model, x_t, device=device, batch_size=batch_size)
    return relative_l2_error(pred, y_np)


def _eval_ood(model, labels: np.ndarray, *, dcfg_train: DatasetConfig, W_norm: float, batch_size: int, device: str, seed_test: int, delta: float) -> np.ndarray:
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


def _eval_mixed(model, dcfg_train: DatasetConfig, W_norm: float, *, test_n: int, batch_size: int, device: str, seed_test: int, delta: float) -> tuple[float, float]:
    dcfg_test = replace(dcfg_train, seed=seed_test)
    x_np, y_np = build_dataset(
        "mixed_circle",
        n_samples=test_n,
        cfg=dcfg_test,
        noise_level=delta,
        W_norm=W_norm,
    )
    x_t, _ = make_dataset_tensors(x_np, y_np)
    pred = predict(model, x_t, device=device, batch_size=batch_size)
    rel = relative_l2_error(pred, y_np)
    acc = pixel_accuracy(threshold_to_classes(pred), threshold_to_classes(y_np))
    return rel, acc


# ---------------- Paper Fig.6-10 reconstruction grids ----------------

def _predict_on_labels(model, labels: np.ndarray, *, dcfg_train: DatasetConfig, W_norm: float, seed_test: int, delta: float, device: str, batch_size: int) -> np.ndarray:
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


def _save_recon_grid(
    out_path: Path,
    *,
    title: str,
    truth_labels: np.ndarray,
    row_predictions: dict[str, np.ndarray],
    row_order: list[str],
    truth_vmin: float,
    truth_vmax: float,
    pred_vmin: float,
    pred_vmax: float,
    pred_postprocess: callable | None = None,
) -> None:
    n_cols = truth_labels.shape[0]
    n_rows = 1 + len(row_order)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.6 * n_rows), squeeze=False)
    for c in range(n_cols):
        ax = axes[0, c]
        ax.imshow(truth_labels[c], origin="lower", extent=(-1, 1, -1, 1), cmap="viridis", vmin=truth_vmin, vmax=truth_vmax)
        ax.set_title(f"sample {c + 1}", fontsize=9)
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    for r, label in enumerate(row_order):
        preds = row_predictions[label]
        for c in range(n_cols):
            ax = axes[r + 1, c]
            img = preds[c]
            if pred_postprocess is not None:
                img = pred_postprocess(img)
            ax.imshow(img, origin="lower", extent=(-1, 1, -1, 1), cmap="viridis", vmin=pred_vmin, vmax=pred_vmax)
            ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    for r, lbl in enumerate(["truth"] + row_order):
        axes[r, 0].set_ylabel(lbl, fontsize=10, rotation=0, ha="right", va="center")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0.02, 0, 1, 0.97])
    save_figure(fig, out_path)
    plt.close(fig)


def _save_polygon_fig6(out_path: Path, models: dict[int, dict], *, seed: int, device: str, mnist_download: bool) -> None:
    pool_cfg = DatasetConfig(image_size=64, n_incident=1, seed=seed + 7000, device=device, mnist_download=mnist_download)
    pool_labels = build_labels("polygon", n_samples=40, cfg=pool_cfg)
    medium_idx = [i for i in range(40) if float(np.max(pool_labels[i])) > 1.5]
    soft_idx = [i for i in range(40) if float(np.max(pool_labels[i])) < 1.5]
    selected = medium_idx[:3] + soft_idx[:2]
    truth_labels = pool_labels[selected]

    row_predictions: dict[str, np.ndarray] = {}
    row_order: list[str] = []
    for ni in (1, 4):
        case = models[ni]
        for delta in (0.02, 0.10):
            preds = _predict_on_labels(
                case["model"], truth_labels,
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
    _save_recon_grid(
        out_path,
        title="Fig.6 polygon dataset reconstructions",
        truth_labels=truth_labels,
        row_predictions=row_predictions,
        row_order=row_order,
        truth_vmin=0.0, truth_vmax=3.0,
        pred_vmin=0.0, pred_vmax=3.0,
        pred_postprocess=threshold_to_classes,
    )


def _save_mnist_fig7(out_path: Path, models: dict[int, dict], *, seed: int, device: str, mnist_download: bool) -> None:
    pool_cfg = DatasetConfig(image_size=64, n_incident=4, seed=seed + 8000, device=device, mnist_download=mnist_download)
    pool_labels = build_labels("mnist", n_samples=20, cfg=pool_cfg, mnist_split="test")
    truth_labels = pool_labels[:5]

    row_predictions: dict[str, np.ndarray] = {}
    row_order: list[str] = []
    for ni in (4, 16):
        case = models[ni]
        for delta in (0.05, 0.10):
            preds = _predict_on_labels(
                case["model"], truth_labels,
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
    _save_recon_grid(
        out_path,
        title="Fig.7 MNIST dataset reconstructions",
        truth_labels=truth_labels,
        row_predictions=row_predictions,
        row_order=row_order,
        truth_vmin=1.0, truth_vmax=1.7,
        pred_vmin=1.0, pred_vmax=1.7,
    )


def _save_ood_grid(
    out_path: Path,
    *,
    title: str,
    truth_labels: np.ndarray,
    models: dict[int, dict],
    seed: int,
    device: str,
) -> None:
    row_predictions: dict[str, np.ndarray] = {}
    row_order: list[str] = []
    for ni in (4, 16):
        case = models[ni]
        for delta in (0.05, 0.10):
            preds = _predict_on_labels(
                case["model"], truth_labels,
                dcfg_train=case["dataset_cfg_train"],
                W_norm=case["W_norm"],
                seed_test=seed + 9000 + ni + int(delta * 100),
                delta=delta,
                device=device,
                batch_size=10,
            )
            label = f"Ni={ni}, delta={int(delta * 100)}%"
            row_predictions[label] = preds
            row_order.append(label)
    _save_recon_grid(
        out_path,
        title=title,
        truth_labels=truth_labels,
        row_predictions=row_predictions,
        row_order=row_order,
        truth_vmin=1.0, truth_vmax=2.0,
        pred_vmin=1.0, pred_vmax=2.0,
    )


def _save_mixed_fig10(out_path: Path, case: dict, *, seed: int, device: str, mnist_download: bool) -> None:
    pool_cfg = DatasetConfig(
        image_size=64, n_incident=10, seed=seed + 10000, device=device,
        mnist_download=mnist_download, receiver_radius=8.0, n_receivers=180,
    )
    pool_labels = build_labels("mixed_circle", n_samples=12, cfg=pool_cfg)
    truth_labels = pool_labels[:5]

    row_predictions: dict[str, np.ndarray] = {}
    row_order: list[str] = []
    for delta in (0.05, 0.10):
        preds = _predict_on_labels(
            case["model"], truth_labels,
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
    _save_recon_grid(
        out_path,
        title="Fig.10 mixed-circle phaseless reconstructions (phased-trained)",
        truth_labels=truth_labels,
        row_predictions=row_predictions,
        row_order=row_order,
        truth_vmin=0.0, truth_vmax=3.0,
        pred_vmin=0.0, pred_vmax=3.0,
    )


# ---------------- Section 5.1 DSM rendering ----------------

def _section51_dsm(summary: dict, *, seed: int, out_fig: Path) -> None:
    """Section 5.1 figures: paper-style forward physics + viridis colormap."""
    def _truth_field(truth_mask: np.ndarray) -> np.ndarray:
        return 1.0 - truth_mask.astype(np.float32)

    def _show_truth(ax, truth_mask: np.ndarray, kind: str) -> None:
        if kind == "medium":
            field = 1.0 + 2.0 * truth_mask.astype(np.float32)
            im = ax.imshow(field, origin="lower", extent=(-1, 1, -1, 1), cmap="viridis", vmin=1.0, vmax=3.0)
            plt.colorbar(im, ax=ax, shrink=0.8)
        else:
            im = ax.imshow(_truth_field(truth_mask), origin="lower", extent=(-1, 1, -1, 1), cmap="viridis", vmin=0.0, vmax=1.0)
            plt.colorbar(im, ax=ax, shrink=0.8)

    def _show_indicator(ax, indicator: np.ndarray, truth_mask: np.ndarray) -> None:
        # Paper-style overlay: zero out obstacle support so the obstacle remains visible.
        overlay = indicator.copy()
        overlay[truth_mask.astype(bool)] = 0.0
        im = ax.imshow(overlay, origin="lower", extent=(-1, 1, -1, 1), cmap="viridis", vmin=0.0, vmax=1.0)
        plt.colorbar(im, ax=ax, shrink=0.8)

    dsm_out: dict[str, dict] = {}
    dsm_jobs = [
        ("ex1_sound_hard_circle", 0.05, 1),
        ("ex1_sound_hard_circle", 0.10, 1),
        ("ex2_sound_soft_squares", 0.05, 1),
        ("ex2_sound_soft_squares", 0.10, 1),
        ("ex3_close_medium_squares", 0.05, 1),
        ("ex3_close_medium_squares", 0.10, 1),
        ("ex4_medium_ring", 0.05, 1),
        ("ex4_medium_ring", 0.10, 1),
        ("ex4_medium_ring", 0.05, 3),
        ("ex4_medium_ring", 0.10, 3),
        ("ex4_medium_ring", 0.05, 5),
        ("ex4_medium_ring", 0.10, 5),
    ]
    cache: dict[tuple, dict] = {}
    for key, noise, ni in dsm_jobs:
        if (key, noise, ni) not in cache:
            cache[(key, noise, ni)] = run_example_dsm_paper(
                example_key=key,
                noise_level=noise,
                n_incident=ni,
                seed=seed,
            )
        out = cache[(key, noise, ni)]
        hit = topk_peak_hit_rate(out["indicator"], out["truth_mask"], k_fraction=0.02)
        err = peak_location_error(out["indicator"], out["truth_mask"], out["grid_x"], out["grid_y"])
        dsm_out[f"{key}|ni={ni}|noise={noise:.2f}"] = {
            "hit_rate_top2pct": hit,
            "peak_location_error": err,
        }
    summary["dsm"] = dsm_out

    trip = [
        ("ex1_sound_hard_circle", "obstacle"),
        ("ex2_sound_soft_squares", "obstacle"),
        ("ex3_close_medium_squares", "medium"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 9.0))
    for i, (key, kind) in enumerate(trip):
        r5 = cache[(key, 0.05, 1)]
        r10 = cache[(key, 0.10, 1)]
        _show_truth(axes[i, 0], r5["truth_mask"], kind)
        _show_indicator(axes[i, 1], r5["indicator"], r5["truth_mask"])
        _show_indicator(axes[i, 2], r10["indicator"], r10["truth_mask"])
        axes[i, 0].set_title(f"Ex{i+1} (a) truth", fontsize=9)
        axes[i, 1].set_title(f"Ex{i+1} (b) $\\delta=5\\%$", fontsize=9)
        axes[i, 2].set_title(f"Ex{i+1} (c) $\\delta=10\\%$", fontsize=9)
        for j in range(3):
            axes[i, j].set_xticks([-1, 0, 1])
            axes[i, j].set_yticks([-1, 0, 1])
    fig.tight_layout()
    save_figure(fig, out_fig / "full_fig2_4_examples1_3.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(12.5, 9.0))
    for row, ni in enumerate((1, 3, 5)):
        r_truth = cache[("ex4_medium_ring", 0.05, ni)]
        r5 = cache[("ex4_medium_ring", 0.05, ni)]
        r10 = cache[("ex4_medium_ring", 0.10, ni)]
        _show_truth(axes[row, 0], r_truth["truth_mask"], "medium")
        _show_indicator(axes[row, 1], r5["indicator"], r5["truth_mask"])
        _show_indicator(axes[row, 2], r10["indicator"], r10["truth_mask"])
        axes[row, 0].set_title(f"$N_i={ni}$ (a) truth", fontsize=9)
        axes[row, 1].set_title(f"$N_i={ni}$ (b) $\\delta=5\\%$", fontsize=9)
        axes[row, 2].set_title(f"$N_i={ni}$ (c) $\\delta=10\\%$", fontsize=9)
        for j in range(3):
            axes[row, j].set_xticks([-1, 0, 1])
            axes[row, j].set_yticks([-1, 0, 1])
    fig.tight_layout()
    save_figure(fig, out_fig / "full_fig5_ring_ni_sweep.png", dpi=170)
    plt.close(fig)


# ---------------- Comparison report ----------------

def _emit_comparison(summary: dict, *, out_res: Path) -> None:
    paper_table1 = {
        "Ni=1,delta=0.02": 0.9949,
        "Ni=1,delta=0.10": 0.9772,
        "Ni=4,delta=0.02": 0.9977,
        "Ni=4,delta=0.10": 0.9916,
    }
    paper_table2 = {
        "mnist,Ni=4,delta=0.05": 0.0827,
        "mnist,Ni=4,delta=0.10": 0.1043,
        "mnist,Ni=16,delta=0.05": 0.0617,
        "mnist,Ni=16,delta=0.10": 0.0755,
        "chinese_like,Ni=4,delta=0.05": 0.1096,
        "chinese_like,Ni=4,delta=0.10": 0.1252,
        "chinese_like,Ni=16,delta=0.05": 0.0721,
        "chinese_like,Ni=16,delta=0.10": 0.0854,
        "austria_ring_1,Ni=4,delta=0.05": 0.1163,
        "austria_ring_1,Ni=4,delta=0.10": 0.1258,
        "austria_ring_1,Ni=16,delta=0.05": 0.0851,
        "austria_ring_1,Ni=16,delta=0.10": 0.0922,
        "austria_ring_2,Ni=4,delta=0.05": 0.1897,
        "austria_ring_2,Ni=4,delta=0.10": 0.1810,
        "austria_ring_2,Ni=16,delta=0.05": 0.1260,
        "austria_ring_2,Ni=16,delta=0.10": 0.1367,
    }

    lines = [
        "# Full Phaseless Strict Reproduction Comparison",
        "",
        f"- paper: `{summary['paper']}`",
        f"- device: `{summary['device']}`",
        f"- seed: `{summary['seed']}`",
        f"- protocol: strict (Helmholtz forward + Eq. 3.11 DSM input, no proxy)",
        "",
        "## Table 1 (Polygon accuracy)",
        "",
        "| case | paper | ours | diff |",
        "|---|---:|---:|---:|",
    ]
    for k, v in paper_table1.items():
        ours = summary["dsmdl"]["polygon"]["accuracy"].get(k, float("nan"))
        lines.append(f"| {k} | {v:.4f} | {ours:.4f} | {ours - v:+.4f} |")

    lines.extend([
        "",
        "## Table 2 (Relative L2)",
        "",
        "| case | paper | ours | diff |",
        "|---|---:|---:|---:|",
    ])
    rel_map = summary["dsmdl"]["mnist_family"]["relative_l2"]
    for k, v in paper_table2.items():
        ours = rel_map.get(k, float("nan"))
        lines.append(f"| {k} | {v:.4f} | {ours:.4f} | {ours - v:+.4f} |")

    lines.extend([
        "",
        "## Section 5.2.3 mixed_circle",
        "",
        "| metric | value |",
        "|---|---:|",
    ])
    for k, v in summary["dsmdl"]["mixed_circle"]["metrics"].items():
        lines.append(f"| {k} | {v:.4f} |")

    lines.extend([
        "",
        "## Notes",
        "",
        "- No public author repo was found; protocol/parameters follow arXiv:2403.02584v2.",
        "- DSM-DL inputs are computed via Eq. (3.11)/(3.12) from a Helmholtz Lippmann-Schwinger solve.",
        "- Training noise = 1%; test noise per Eq. (5.1) at each evaluated delta.",
        "- MNIST uses official torchvision train/test split when reachable; otherwise a by-description fallback is used and disclosed in the JSON metadata.",
        "- Chinese-like and Austria-like profiles are constructed from the paper's textual description; the original test images are not public.",
    ])

    (out_res / "full_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    (out_res / "full_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    _seed_everything(args.seed)
    apply_idsm_plot_style()

    out_res = ROOT / "results" / "phaseless"
    out_fig = ROOT / "figures" / "06_phaseless"
    out_res.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "paper": "arXiv:2403.02584v2",
        "seed": args.seed,
        "device": args.device,
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dsm": {},
        "dsmdl": {},
    }

    _section51_dsm(summary, seed=args.seed, out_fig=out_fig)

    ckpt_dir = out_res / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Section 5.2.1 Polygon (Ni=1, 4)
    poly_cfg = {"train_n": 7000, "test_n": 200, "batch": 10, "epochs": 30}
    poly_metrics: dict[str, float] = {}
    poly_runtime: dict[str, float] = {}
    poly_cases: dict[int, dict] = {}
    for ni in (1, 4):
        case = _train_case(
            dataset_kind="polygon",
            n_incident=ni,
            train_n=poly_cfg["train_n"],
            batch_size=poly_cfg["batch"],
            epochs=poly_cfg["epochs"],
            seed=args.seed,
            device=args.device,
            extra_cfg_kwargs={
                "receiver_radius": 4.0,
                "n_receivers": 100,
            },
            mnist_download=args.mnist_download,
        )
        poly_runtime[f"Ni{ni}"] = case["runtime_seconds"]
        save_model_checkpoint(
            ckpt_dir / f"polygon_Ni{ni}.pt",
            model=case["model"],
            dataset_cfg=case["dataset_cfg_train"],
            train_cfg=TrainingConfig(**case["train_cfg"]),
            extra_meta={"W_norm": case["W_norm"], "phased_training": case["phased_training"]},
        )
        for delta in (0.02, 0.10):
            acc = _eval_polygon(
                case["model"],
                case["dataset_cfg_train"],
                case["W_norm"],
                test_n=poly_cfg["test_n"],
                batch_size=poly_cfg["batch"],
                device=args.device,
                seed_test=args.seed + 1001 + ni,
                delta=delta,
                ni=ni,
            )
            poly_metrics[f"Ni={ni},delta={delta:.2f}"] = acc
        poly_cases[ni] = case
    summary["dsmdl"]["polygon"] = {
        "train_runtime_seconds": poly_runtime,
        "accuracy": poly_metrics,
    }
    _save_polygon_fig6(
        out_fig / "fig6_polygon_recon.png",
        poly_cases, seed=args.seed, device=args.device, mnist_download=args.mnist_download,
    )
    for ni in list(poly_cases):
        _cleanup(poly_cases[ni]["model"])
    poly_cases.clear()

    # Section 5.2.2 MNIST + circle (Ni=4, 16)
    mnist_cfg = {"train_n": 10000, "test_n": 200, "batch": 10, "epochs": 30}
    mnist_metrics: dict[str, float] = {}
    mnist_runtime: dict[str, float] = {}
    char_labels = _draw_chinese_like_set(64)
    austria_1, austria_2 = _draw_austria_like_set(64)
    mnist_cases: dict[int, dict] = {}
    for ni in (4, 16):
        case = _train_case(
            dataset_kind="mnist",
            n_incident=ni,
            train_n=mnist_cfg["train_n"],
            batch_size=mnist_cfg["batch"],
            epochs=mnist_cfg["epochs"],
            seed=args.seed,
            device=args.device,
            extra_cfg_kwargs={
                "receiver_radius": 4.0,
                "n_receivers": 100,
            },
            mnist_download=args.mnist_download,
        )
        mnist_runtime[f"Ni{ni}"] = case["runtime_seconds"]
        save_model_checkpoint(
            ckpt_dir / f"mnist_Ni{ni}.pt",
            model=case["model"],
            dataset_cfg=case["dataset_cfg_train"],
            train_cfg=TrainingConfig(**case["train_cfg"]),
            extra_meta={"W_norm": case["W_norm"], "phased_training": case["phased_training"]},
        )
        for delta in (0.05, 0.10):
            mnist_metrics[f"mnist,Ni={ni},delta={delta:.2f}"] = _eval_mnist_split(
                case["model"],
                case["dataset_cfg_train"],
                case["W_norm"],
                test_n=mnist_cfg["test_n"],
                batch_size=mnist_cfg["batch"],
                device=args.device,
                seed_test=args.seed + 2001 + ni,
                delta=delta,
            )

            pred_c = _eval_ood(
                case["model"], char_labels,
                dcfg_train=case["dataset_cfg_train"],
                W_norm=case["W_norm"],
                batch_size=mnist_cfg["batch"],
                device=args.device,
                seed_test=args.seed + 3001 + ni + int(delta * 100),
                delta=delta,
            )
            mnist_metrics[f"chinese_like,Ni={ni},delta={delta:.2f}"] = relative_l2_error(pred_c, char_labels)

            pred_a1 = _eval_ood(
                case["model"], austria_1,
                dcfg_train=case["dataset_cfg_train"],
                W_norm=case["W_norm"],
                batch_size=mnist_cfg["batch"],
                device=args.device,
                seed_test=args.seed + 4001 + ni + int(delta * 100),
                delta=delta,
            )
            pred_a2 = _eval_ood(
                case["model"], austria_2,
                dcfg_train=case["dataset_cfg_train"],
                W_norm=case["W_norm"],
                batch_size=mnist_cfg["batch"],
                device=args.device,
                seed_test=args.seed + 5001 + ni + int(delta * 100),
                delta=delta,
            )
            mnist_metrics[f"austria_ring_1,Ni={ni},delta={delta:.2f}"] = relative_l2_error(pred_a1, austria_1)
            mnist_metrics[f"austria_ring_2,Ni={ni},delta={delta:.2f}"] = relative_l2_error(pred_a2, austria_2)
        mnist_cases[ni] = case
    summary["dsmdl"]["mnist_family"] = {
        "train_runtime_seconds": mnist_runtime,
        "relative_l2": mnist_metrics,
    }
    _save_mnist_fig7(
        out_fig / "fig7_mnist_recon.png",
        mnist_cases, seed=args.seed, device=args.device, mnist_download=args.mnist_download,
    )
    _save_ood_grid(
        out_fig / "fig8_chinese_recon.png",
        title="Fig.8 Chinese-character OOD reconstructions",
        truth_labels=char_labels,
        models=mnist_cases, seed=args.seed + 8500, device=args.device,
    )
    _save_ood_grid(
        out_fig / "fig9_austria_recon.png",
        title="Fig.9 Austria ring OOD reconstructions",
        truth_labels=np.concatenate([austria_1, austria_2], axis=0),
        models=mnist_cases, seed=args.seed + 8800, device=args.device,
    )
    for ni in list(mnist_cases):
        _cleanup(mnist_cases[ni]["model"])
    mnist_cases.clear()

    # Section 5.2.3 mixed-circle (Ni=10, receiver radius 8, 180 receivers)
    mix_cfg = {"train_n": 20000, "test_n": 200, "batch": 20, "epochs": 30}
    mix_metrics: dict[str, float] = {}
    case = _train_case(
        dataset_kind="mixed_circle",
        n_incident=10,
        train_n=mix_cfg["train_n"],
        batch_size=mix_cfg["batch"],
        epochs=mix_cfg["epochs"],
        seed=args.seed,
        device=args.device,
        extra_cfg_kwargs={
            "receiver_radius": 8.0,
            "n_receivers": 180,
        },
        mnist_download=args.mnist_download,
        phased_training=True,
    )
    mix_runtime = case["runtime_seconds"]
    save_model_checkpoint(
        ckpt_dir / "mixed_circle_Ni10.pt",
        model=case["model"],
        dataset_cfg=case["dataset_cfg_train"],
        train_cfg=TrainingConfig(**case["train_cfg"]),
        extra_meta={"W_norm": case["W_norm"], "phased_training": case["phased_training"]},
    )
    for delta in (0.05, 0.10):
        rel, acc = _eval_mixed(
            case["model"],
            case["dataset_cfg_train"],
            case["W_norm"],
            test_n=mix_cfg["test_n"],
            batch_size=mix_cfg["batch"],
            device=args.device,
            seed_test=args.seed + 9001 + int(delta * 100),
            delta=delta,
        )
        mix_metrics[f"Ni=10,delta={delta:.2f},rel_l2"] = rel
        mix_metrics[f"Ni=10,delta={delta:.2f},acc"] = acc
    summary["dsmdl"]["mixed_circle"] = {
        "train_runtime_seconds": mix_runtime,
        "metrics": mix_metrics,
    }
    _save_mixed_fig10(
        out_fig / "fig10_mixed_recon.png",
        case, seed=args.seed, device=args.device, mnist_download=args.mnist_download,
    )
    _cleanup(case["model"])

    _emit_comparison(summary, out_res=out_res)
    print(f"[ok] wrote {out_res / 'full_summary.json'}")
    print(f"[ok] wrote {out_res / 'full_comparison.md'}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict full phaseless reproduction.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--mnist-download", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
