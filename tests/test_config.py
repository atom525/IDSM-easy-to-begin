"""Tests for config.py -- configuration dataclasses."""

import os
import pytest

from cooperation.ghy.IDSM.src.config import (
    RuntimeConfig,
    MeshConfig,
    FullIDSMConfig,
    PartialIDSMConfig,
    Notebook04Config,
)


def test_runtime_config_defaults():
    """RuntimeConfig defaults should be sensible."""
    cfg = RuntimeConfig()
    assert cfg.use_gpu is False
    assert cfg.gpu_backend == "auto"
    assert cfg.random_seed == 42


def test_runtime_config_from_env_defaults(monkeypatch):
    """from_env() should use defaults when no env vars are set."""
    monkeypatch.delenv("IDSM_USE_GPU", raising=False)
    monkeypatch.delenv("IDSM_GPU_BACKEND", raising=False)
    monkeypatch.delenv("IDSM_SEED", raising=False)
    cfg = RuntimeConfig.from_env()
    assert cfg.use_gpu is False
    assert cfg.gpu_backend == "auto"
    assert cfg.random_seed == 42


def test_runtime_config_from_env_custom(monkeypatch):
    """from_env() should read environment variables correctly."""
    monkeypatch.setenv("IDSM_USE_GPU", "1")
    monkeypatch.setenv("IDSM_GPU_BACKEND", "cupy")
    monkeypatch.setenv("IDSM_SEED", "123")
    cfg = RuntimeConfig.from_env()
    assert cfg.use_gpu is True
    assert cfg.gpu_backend == "cupy"
    assert cfg.random_seed == 123


def test_runtime_resolve_device_cpu():
    """resolve_device should return CPU when GPU is disabled."""
    cfg = RuntimeConfig(use_gpu=False)
    dev = cfg.resolve_device()
    assert dev["enabled"] is False
    assert dev["backend"] == "cpu"


def test_runtime_resolve_device_gpu_fallback():
    """GPU enabled should safely fall back to CPU."""
    cfg = RuntimeConfig(use_gpu=True)
    dev = cfg.resolve_device()
    assert dev["enabled"] is False
    assert dev["backend"] == "cpu"
    assert "CPU" in dev["reason"] or "cpu" in dev["reason"].lower()


def test_mesh_config_defaults():
    """MeshConfig defaults should be sensible."""
    cfg = MeshConfig()
    assert cfg.n_boundary == 500
    assert cfg.n_grid == 201


def test_full_idsm_config_defaults():
    """FullIDSMConfig defaults should be sensible."""
    cfg = FullIDSMConfig()
    assert cfg.sigma_bg == 1.0
    assert cfg.alpha == 1.0
    assert cfg.n_iter == 22
    assert cfg.lowrank_method == "BFG"
    assert cfg.problem_type == "conductivity"
    assert 0.0 < cfg.sigma_range < cfg.sigma_bg
    # sigma_range is the search lower bound (FreeFEM: cB=0.01), not inclusion truth
    assert cfg.sigma_range == 0.01
    # R_0 initialization exponent (FreeFEM L260-263)
    assert cfg.cond_exponent == 0.5
    assert cfg.pot_exponent == 0.0


def test_partial_idsm_config_defaults():
    """PartialIDSMConfig defaults should be sensible."""
    cfg = PartialIDSMConfig()
    assert cfg.alpha_d < cfg.alpha_n  # alpha_d < alpha_n (Paper 3 design)
    assert cfg.p_norm >= 1.0
    assert cfg.gamma_D > 0


def test_notebook04_config_nested():
    """Notebook04Config should contain nested configs."""
    cfg = Notebook04Config()
    assert isinstance(cfg.mesh, MeshConfig)
    assert isinstance(cfg.full, FullIDSMConfig)
    assert isinstance(cfg.partial, PartialIDSMConfig)
    assert len(cfg.noise_levels) > 0
    assert all(0 <= nl <= 1 for nl in cfg.noise_levels)


def test_config_dataclass_immutable_default():
    """Two Notebook04Config instances should have independent list defaults."""
    cfg1 = Notebook04Config()
    cfg2 = Notebook04Config()
    cfg1.noise_levels.append(0.99)
    assert 0.99 not in cfg2.noise_levels
