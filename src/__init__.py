"""Educational IDSM package for elliptic inverse problems.

This package follows:
- Ito, Jin, Wang, Zou (2025): IDSM for elliptic inverse problems
- Jin, Wang, Zou (2026): stable IDSM with partial Cauchy data
"""

from .mesh import EllipticMesh, generate_elliptic_mesh, generate_sampling_grid
from .forward_solver import (
    circle_inclusion,
    generate_cauchy_data,
    generate_cauchy_data_general,
    make_conductivity_conductive,
    make_conductivity_example1,
    make_conductivity_single,
    make_potential_example3,
    solve_forward,
    solve_forward_general,
    square_inclusion,
)
from .dsm import compute_dsm_indicator, discretize_laplace_beltrami
from .idsm import run_idsm
from .idsm_partial import run_idsm_partial
from .config import (
    RuntimeConfig,
    MeshConfig,
    FullIDSMConfig,
    PartialIDSMConfig,
    Notebook04Config,
)

__all__ = [
    "EllipticMesh",
    "generate_elliptic_mesh",
    "generate_sampling_grid",
    "circle_inclusion",
    "square_inclusion",
    "make_conductivity_conductive",
    "make_conductivity_example1",
    "make_conductivity_single",
    "make_potential_example3",
    "solve_forward",
    "solve_forward_general",
    "generate_cauchy_data",
    "generate_cauchy_data_general",
    "discretize_laplace_beltrami",
    "compute_dsm_indicator",
    "run_idsm",
    "run_idsm_partial",
    "RuntimeConfig",
    "MeshConfig",
    "FullIDSMConfig",
    "PartialIDSMConfig",
    "Notebook04Config",
]
