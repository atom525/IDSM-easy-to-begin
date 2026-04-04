"""Centralized runtime and experiment configuration.

This module keeps major hyperparameters in one place to avoid scattered
hard-coded values in scripts/notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Dict, Any, List


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class RuntimeConfig:
    """Runtime flags, including optional GPU toggle."""

    use_gpu: bool = False
    gpu_backend: str = "auto"  # "auto", "cupy", "torch"
    random_seed: int = 42

    @staticmethod
    def from_env() -> "RuntimeConfig":
        return RuntimeConfig(
            use_gpu=_env_flag("IDSM_USE_GPU", False),
            gpu_backend=os.getenv("IDSM_GPU_BACKEND", "auto"),
            random_seed=int(os.getenv("IDSM_SEED", "42")),
        )

    def resolve_device(self) -> Dict[str, Any]:
        """Resolve effective backend.

        Notes
        -----
        Core FEM sparse solves in this project use SciPy sparse linear algebra.
        These are CPU-based in the current implementation. If GPU is requested,
        we keep a deterministic CPU fallback and report it explicitly.
        """
        if not self.use_gpu:
            return {"enabled": False, "backend": "cpu", "reason": "GPU disabled"}

        # Keep strict CPU fallback unless a validated GPU sparse backend is wired.
        return {
            "enabled": False,
            "backend": "cpu",
            "reason": (
                "GPU requested, but current FEM sparse-solver path is CPU-only; "
                "falling back to CPU to preserve numerical consistency."
            ),
        }


@dataclass
class MeshConfig:
    n_boundary: int = 500
    n_grid: int = 201


@dataclass
class FullIDSMConfig:
    sigma_bg: float = 1.0
    potential_bg: float = 1e-10
    sigma_range: float = 0.01
    potential_range: float = 2e-10
    alpha: float = 1.0
    n_iter: int = 22
    lowrank_method: str = "BFG"
    problem_type: str = "conductivity"
    coeff_known: bool = False


@dataclass
class PartialIDSMConfig:
    sigma_bg: float = 1.0
    potential_bg: float = 1e-10
    sigma_range: float = 0.01
    potential_range: float = 2e-10
    alpha_d: float = 0.05
    alpha_n: float = 2.0
    n_iter: int = 22
    lowrank_method: str = "BFG"
    problem_type: str = "conductivity"
    coeff_known: bool = False
    gamma_D: float = 4.0
    epsilon_cutoff: float = 0.02
    p_norm: float = 2.0


@dataclass
class Notebook04Config:
    noise_levels: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.3])
    eps_sweep: List[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    )
    dsm_gamma: float = 0.5
    mesh: MeshConfig = field(default_factory=MeshConfig)
    full: FullIDSMConfig = field(default_factory=FullIDSMConfig)
    partial: PartialIDSMConfig = field(default_factory=PartialIDSMConfig)

