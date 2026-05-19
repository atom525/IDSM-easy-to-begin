"""Tests for phaseless_scattering.py."""

import numpy as np

from src.phaseless_scattering import (
    PhaselessDSMConfig,
    add_phaseless_noise,
    corrected_phaseless_data,
    make_incident_angles,
    make_receiver_points,
    make_uniform_grid,
    run_example_dsm,
)


def test_corrected_phaseless_data_shape_and_finite():
    cfg = PhaselessDSMConfig()
    recv = make_receiver_points(cfg.receiver_radius, cfg.n_receivers)
    angle = make_incident_angles(1)[0]
    phase = cfg.k * (recv[:, 0] * np.cos(angle) + recv[:, 1] * np.sin(angle))
    u_inc = np.exp(1j * phase)
    abs_inc = np.abs(u_inc)
    abs_noisy = abs_inc + 0.1
    delta = corrected_phaseless_data(abs_noisy, abs_inc, u_inc)
    assert delta.shape == (cfg.n_receivers,)
    assert np.all(np.isfinite(delta.real))
    assert np.all(np.isfinite(delta.imag))


def test_noise_model_rms_scales_with_delta():
    rng = np.random.default_rng(0)
    base = np.linspace(0.1, 2.0, 200)
    y1 = add_phaseless_noise(base, noise_level=0.02, rng=rng)
    rng = np.random.default_rng(0)
    y2 = add_phaseless_noise(base, noise_level=0.20, rng=rng)
    e1 = np.sqrt(np.mean((y1 - base) ** 2))
    e2 = np.sqrt(np.mean((y2 - base) ** 2))
    assert e2 > 4.0 * e1


def test_run_example_dsm_smoke_and_normalized():
    cfg = PhaselessDSMConfig(forward_grid_size=28, scan_grid_size=48, n_receivers=64)
    out = run_example_dsm(
        example_key="ex1_sound_hard_circle",
        cfg=cfg,
        noise_level=0.05,
        n_incident=1,
        seed=0,
    )
    assert out["indicator"].shape == (cfg.scan_grid_size, cfg.scan_grid_size)
    assert out["truth_mask"].shape == (cfg.scan_grid_size, cfg.scan_grid_size)
    assert np.nanmax(out["indicator"]) <= 1.000001
    assert np.nanmin(out["indicator"]) >= -1e-12

