"""测试 config.py — 配置 dataclass 测试。"""

import os
import pytest

from IDSM.src.config import (
    RuntimeConfig,
    MeshConfig,
    FullIDSMConfig,
    PartialIDSMConfig,
    Notebook04Config,
)


def test_runtime_config_defaults():
    """RuntimeConfig 默认值应合理。"""
    cfg = RuntimeConfig()
    assert cfg.use_gpu is False
    assert cfg.gpu_backend == "auto"
    assert cfg.random_seed == 42


def test_runtime_config_from_env_defaults(monkeypatch):
    """from_env() 在无环境变量时应使用默认值。"""
    monkeypatch.delenv("IDSM_USE_GPU", raising=False)
    monkeypatch.delenv("IDSM_GPU_BACKEND", raising=False)
    monkeypatch.delenv("IDSM_SEED", raising=False)
    cfg = RuntimeConfig.from_env()
    assert cfg.use_gpu is False
    assert cfg.gpu_backend == "auto"
    assert cfg.random_seed == 42


def test_runtime_config_from_env_custom(monkeypatch):
    """from_env() 应正确读取环境变量。"""
    monkeypatch.setenv("IDSM_USE_GPU", "1")
    monkeypatch.setenv("IDSM_GPU_BACKEND", "cupy")
    monkeypatch.setenv("IDSM_SEED", "123")
    cfg = RuntimeConfig.from_env()
    assert cfg.use_gpu is True
    assert cfg.gpu_backend == "cupy"
    assert cfg.random_seed == 123


def test_runtime_resolve_device_cpu():
    """GPU 关闭时 resolve_device 应返回 CPU。"""
    cfg = RuntimeConfig(use_gpu=False)
    dev = cfg.resolve_device()
    assert dev["enabled"] is False
    assert dev["backend"] == "cpu"


def test_runtime_resolve_device_gpu_fallback():
    """GPU 开启时应安全回退到 CPU。"""
    cfg = RuntimeConfig(use_gpu=True)
    dev = cfg.resolve_device()
    assert dev["enabled"] is False
    assert dev["backend"] == "cpu"
    assert "CPU" in dev["reason"] or "cpu" in dev["reason"].lower()


def test_mesh_config_defaults():
    """MeshConfig 默认值应合理。"""
    cfg = MeshConfig()
    assert cfg.n_boundary == 500
    assert cfg.n_grid == 201


def test_full_idsm_config_defaults():
    """FullIDSMConfig 默认值应合理。"""
    cfg = FullIDSMConfig()
    assert cfg.sigma_bg == 1.0
    assert cfg.alpha == 1.0
    assert cfg.n_iter == 22
    assert cfg.lowrank_method == "BFG"
    assert cfg.problem_type == "conductivity"
    assert 0.0 < cfg.sigma_range < cfg.sigma_bg
    # sigma_range 是搜索下界（FreeFEM: cB=0.01），非夹杂真值
    assert cfg.sigma_range == 0.01
    # R₀ 初始化指数（FreeFEM L260-263）
    assert cfg.cond_exponent == 0.5
    assert cfg.pot_exponent == 0.0


def test_partial_idsm_config_defaults():
    """PartialIDSMConfig 默认值应合理。"""
    cfg = PartialIDSMConfig()
    assert cfg.alpha_d < cfg.alpha_n  # α_d < α_n（Paper 3 设计）
    assert cfg.p_norm >= 1.0
    assert cfg.gamma_D > 0


def test_notebook04_config_nested():
    """Notebook04Config 应包含嵌套配置。"""
    cfg = Notebook04Config()
    assert isinstance(cfg.mesh, MeshConfig)
    assert isinstance(cfg.full, FullIDSMConfig)
    assert isinstance(cfg.partial, PartialIDSMConfig)
    assert len(cfg.noise_levels) > 0
    assert all(0 <= nl <= 1 for nl in cfg.noise_levels)


def test_config_dataclass_immutable_default():
    """两个 Notebook04Config 实例的列表默认值应独立。"""
    cfg1 = Notebook04Config()
    cfg2 = Notebook04Config()
    cfg1.noise_levels.append(0.99)
    assert 0.99 not in cfg2.noise_levels
