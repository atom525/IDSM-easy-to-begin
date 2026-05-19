"""Tests for phaseless_dsmdl.py."""

import numpy as np
import torch

from src.phaseless_dsmdl import (
    DatasetConfig,
    UNetDSMDL,
    build_dataset,
    dsmdl_loss,
    make_dataset_tensors,
)


def test_unet_forward_shape():
    model = UNetDSMDL(in_channels=4, out_channels=1, base_channels=16)
    x = torch.randn(2, 4, 64, 64)
    y = model(x)
    assert y.shape == (2, 1, 64, 64)


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

