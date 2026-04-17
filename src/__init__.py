"""Educational IDSM package for elliptic inverse problems.

This package follows:
- Ito, Jin, Wang, Zou (2025): IDSM for elliptic inverse problems
- Jin, Wang, Zou (2026): stable IDSM with partial Cauchy data
"""

from .mesh import EllipticMesh, generate_elliptic_mesh, generate_sampling_grid
from .fem import (
    assemble_stiffness_matrix,
    assemble_mass_matrix,
    assemble_boundary_mass_matrix,
    assemble_boundary_load,
    assemble_boundary_mean_constraint,
    assemble_partial_boundary_mass_matrix,
    solve_neumann_system,
    solve_robin_system,
    compute_boundary_normal_flux,
    compute_boundary_normal_derivative,
)
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
from .utils import (
    compute_iou,
    compute_iou_from_grid,
    p0_to_grid,
    plot_mesh,
    plot_field,
    plot_p0_field,
    plot_boundary_data,
    EXAMPLE1_BOXES,
)
from .config import (
    RuntimeConfig,
    MeshConfig,
    FullIDSMConfig,
    PartialIDSMConfig,
    Notebook04Config,
)

__all__ = [
    # mesh
    "EllipticMesh",
    "generate_elliptic_mesh",
    "generate_sampling_grid",
    # fem
    "assemble_stiffness_matrix",
    "assemble_mass_matrix",
    "assemble_boundary_mass_matrix",
    "assemble_boundary_load",
    "assemble_boundary_mean_constraint",
    "assemble_partial_boundary_mass_matrix",
    "solve_neumann_system",
    "solve_robin_system",
    "compute_boundary_normal_flux",
    "compute_boundary_normal_derivative",
    # forward_solver
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
    # dsm
    "discretize_laplace_beltrami",
    "compute_dsm_indicator",
    # idsm
    "run_idsm",
    "run_idsm_partial",
    # utils
    "compute_iou",
    "compute_iou_from_grid",
    "p0_to_grid",
    "plot_mesh",
    "plot_field",
    "plot_p0_field",
    "plot_boundary_data",
    "EXAMPLE1_BOXES",
    # config
    "RuntimeConfig",
    "MeshConfig",
    "FullIDSMConfig",
    "PartialIDSMConfig",
    "Notebook04Config",
]
