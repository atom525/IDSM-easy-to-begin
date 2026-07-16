"""Tests for the reference-code Python port."""

import numpy as np
import torch

from src.phaseless_reference import (
    ForwardConfig,
    TrainingConfigLegacy,
    UNet3Ab,
    compute_inputs_from_fields,
    generate_forward_dataset,
    select_incidence_indices,
    train_unet3ab,
)


def test_select_incidence_indices_matches_reference_rule():
    idx = select_incidence_indices(16, 4)
    assert idx == [0, 4, 8, 12]


def test_compute_inputs_shapes():
    rng = np.random.default_rng(0)
    n_re = 100
    n_inc = 16
    n_samples = 3
    e_i = rng.normal(size=(n_re, n_inc, n_samples)).astype(np.float32)
    e_i = e_i + 1j * rng.normal(size=(n_re, n_inc, n_samples)).astype(np.float32)
    e_s = 0.05 * (
        rng.normal(size=(n_re, n_inc, n_samples)).astype(np.float32)
        + 1j * rng.normal(size=(n_re, n_inc, n_samples)).astype(np.float32)
    )
    r_mat = (
        rng.normal(size=(n_re, 64 * 64)).astype(np.float32)
        + 1j * rng.normal(size=(n_re, 64 * 64)).astype(np.float32)
    )
    out = compute_inputs_from_fields(
        e_i=e_i,
        e_s=e_s,
        r_mat=r_mat,
        n_incident=4,
        noise_level=0.01,
        seed=1,
    )
    assert out["inputs_noisy"].shape == (n_samples, 4, 64, 64)
    assert np.isfinite(out["inputs_noisy"]).all()


def test_unet3ab_forward_shape():
    model = UNet3Ab(img_ch=4, output_ch=1, n_ch=16)
    x = torch.randn(2, 4, 64, 64)
    y = model(x)
    assert y.shape == (2, 1, 64, 64)


def test_train_unet3ab_smoke():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(30, 4, 64, 64)).astype(np.float32)
    y = (1.0 + 0.7 * rng.random(size=(30, 64, 64))).astype(np.float32)
    cfg = TrainingConfigLegacy(
        n_incident=4,
        epochs=1,
        batch_size=5,
        batch_number=4,
        train_size=20,
        test_start=20,
        test_stop=30,
        n_channels=16,
        device="cpu",
    )
    out = train_unet3ab(inputs_noisy=x, contrast=y, cfg=cfg)
    assert "best_test_rel_l2" in out
    assert np.isfinite(out["best_test_rel_l2"])
    assert len(out["history"]["train_loss"]) == 1


def test_generate_forward_dataset_small_grid():
    cfg = ForwardConfig(mx=16, n_incident=2, n_receivers=20)
    contrast = np.ones((1, 16, 16), dtype=np.float32)
    contrast[0, 6:10, 6:10] = 1.4
    out = generate_forward_dataset(contrast, cfg=cfg, seed=0)
    assert out["Contrast"].shape == (1, 16, 16)
    assert out["E_i"].shape == (20, 2, 1)
    assert out["E_s"].shape == (20, 2, 1)
    assert out["R_mat"].shape == (20, 16 * 16)
