"""Tests for phaseless_dsmdl.py."""

import numpy as np
import torch

from src.phaseless_dsmdl import (
    DatasetConfig,
    UNetDSMDL,
    build_dataset,
    build_labels_with_meta,
    compute_strict_dsm_inputs,
    compute_strict_dsm_inputs_with_meta,
    dsmdl_loss,
    make_dataset_tensors,
)


def test_unet_forward_shape():
    model = UNetDSMDL(in_channels=4, out_channels=1, base_channels=16)
    x = torch.randn(2, 4, 64, 64)
    y = model(x)
    assert y.shape == (2, 1, 64, 64)
    y_trace, trace = model.forward_with_trace(x)
    assert y_trace.shape == y.shape
    assert trace["input"] == (2, 4, 64, 64)
    assert trace["bottleneck"] == (2, 256, 4, 4)
    assert trace["output"] == (2, 1, 64, 64)


def test_dsmdl_loss_is_finite():
    pred = torch.randn(3, 1, 32, 32)
    truth = torch.randn(3, 1, 32, 32)
    val = dsmdl_loss(pred, truth, alpha_tv=0.5, alpha_ssim=0.5)
    assert torch.isfinite(val)
    assert float(val.item()) >= 0.0


def test_build_polygon_dataset_shapes():
    cfg = DatasetConfig(image_size=64, n_incident=4, seed=0)
    x_np, y_np = build_dataset("polygon", n_samples=8, cfg=cfg)
    assert x_np.shape == (8, 4, 64, 64)
    assert y_np.shape == (8, 64, 64)
    x, y = make_dataset_tensors(x_np, y_np)
    assert x.shape == (8, 4, 64, 64)
    assert y.shape == (8, 1, 64, 64)
    assert np.isfinite(x_np).all()
    assert np.isfinite(y_np).all()


def test_bie_vie_meta_pipeline_polygon():
    """Strict BIE Dirichlet + VIE Born-superposition path produces finite,
    distinct outputs for soft vs. medium polygon samples."""
    cfg = DatasetConfig(image_size=48, n_incident=2, seed=0, forward_grid_size=32)
    labels, metas = build_labels_with_meta("polygon", n_samples=4, cfg=cfg)
    x_bie, _ = compute_strict_dsm_inputs_with_meta(
        labels, metas, cfg=cfg, noise_level=0.0, seed=0
    )
    x_legacy, _ = compute_strict_dsm_inputs(labels, cfg=cfg, noise_level=0.0, seed=0)
    assert x_bie.shape == x_legacy.shape == (4, 2, 48, 48)
    assert np.isfinite(x_bie).all()
    soft_idx = [i for i, m in enumerate(metas) if m[0]["kind"] == "soft"]
    medium_idx = [i for i, m in enumerate(metas) if m[0]["kind"] == "medium"]
    assert soft_idx and medium_idx, "test data must contain both kinds"
    if soft_idx:
        soft_diff = float(
            np.linalg.norm(x_bie[soft_idx[0]] - x_legacy[soft_idx[0]])
            / (np.linalg.norm(x_legacy[soft_idx[0]]) + 1e-9)
        )
        assert soft_diff > 1e-3, (
            "BIE Dirichlet soft scatterers should differ measurably from complex-n VIE surrogate"
        )


def test_bie_vie_meta_pipeline_mixed_circle():
    """Mixed-circle pipeline handles multi-scatterer Born superposition."""
    cfg = DatasetConfig(image_size=48, n_incident=2, seed=1, forward_grid_size=32)
    labels, metas = build_labels_with_meta("mixed_circle", n_samples=3, cfg=cfg)
    x, _ = compute_strict_dsm_inputs_with_meta(labels, metas, cfg=cfg, noise_level=0.0, seed=1)
    assert x.shape == (3, 2, 48, 48)
    assert np.isfinite(x).all()
    assert (x.max() - x.min()) > 0.1

