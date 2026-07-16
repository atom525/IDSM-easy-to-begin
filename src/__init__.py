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
    make_double_example2,
    make_potential_example3,
    solve_forward,
    solve_forward_general,
    square_inclusion,
)
from .dsm import compute_dsm_indicator, discretize_laplace_beltrami
from .idsm import run_idsm
from .idsm_partial import run_idsm_partial
from .mesh import generate_disk_mesh
from .utils import (
    compute_iou,
    compute_iou_from_grid,
    p0_to_grid,
    plot_mesh,
    plot_field,
    plot_p0_field,
    plot_boundary_data,
    EXAMPLE1_BOXES,
    SINGLE_INCLUSION_CIRCLE,
    CMAP_SIGMA,
    CMAP_INDICATOR,
    CMAP_FORWARD,
    CMAP_POTENTIAL,
    CMAP_CLASSIFY,
    SIGMA_VMIN_EX1,
    SIGMA_VMAX_EX1,
    SIGMA_VMIN_EX2,
    SIGMA_VMAX_EX2,
    V_VMIN,
    V_VMAX,
    ETA_VMIN,
    ETA_VMAX,
    TRUTH_RECT_KW,
    TRUTH_CIRCLE_KW,
    add_truth_boxes,
    add_truth_circles,
    plot_sigma_reconstruction,
    plot_indicator_grid,
)
from .config import (
    RuntimeConfig,
    MeshConfig,
    FullIDSMConfig,
    PartialIDSMConfig,
    DoubleIDSMConfig,
    Notebook04Config,
)
from .phaseless_scattering import (
    PhaselessDSMConfig,
    batch_run_examples,
    compute_multi_incidence_indicator,
    compute_phaseless_dsm_indicator,
    corrected_phaseless_data,
    example_specs,
    run_example_dsm,
)
from .phaseless_dsmdl import (
    DatasetConfig,
    TrainingConfig,
    UNetDSMDL,
    build_dataset,
    dsmdl_loss,
    make_dataset_tensors,
    train_unet,
)
from .phaseless_reference import (
    ForwardConfig,
    TrainingConfigLegacy,
    UNet3Ab,
    compute_inputs_from_mat,
    generate_forward_dataset,
    train_unet3ab,
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
    "make_double_example2",
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
    "generate_disk_mesh",
    # utils
    "compute_iou",
    "compute_iou_from_grid",
    "p0_to_grid",
    "plot_mesh",
    "plot_field",
    "plot_p0_field",
    "plot_boundary_data",
    "EXAMPLE1_BOXES",
    "SINGLE_INCLUSION_CIRCLE",
    "CMAP_SIGMA",
    "CMAP_INDICATOR",
    "CMAP_FORWARD",
    "CMAP_POTENTIAL",
    "CMAP_CLASSIFY",
    "SIGMA_VMIN_EX1",
    "SIGMA_VMAX_EX1",
    "SIGMA_VMIN_EX2",
    "SIGMA_VMAX_EX2",
    "V_VMIN",
    "V_VMAX",
    "ETA_VMIN",
    "ETA_VMAX",
    "TRUTH_RECT_KW",
    "TRUTH_CIRCLE_KW",
    "add_truth_boxes",
    "add_truth_circles",
    "plot_sigma_reconstruction",
    "plot_indicator_grid",
    # config
    "RuntimeConfig",
    "MeshConfig",
    "FullIDSMConfig",
    "PartialIDSMConfig",
    "DoubleIDSMConfig",
    "Notebook04Config",
    # phaseless scattering
    "PhaselessDSMConfig",
    "example_specs",
    "corrected_phaseless_data",
    "compute_phaseless_dsm_indicator",
    "compute_multi_incidence_indicator",
    "run_example_dsm",
    "batch_run_examples",
    # phaseless dsmdl
    "DatasetConfig",
    "TrainingConfig",
    "UNetDSMDL",
    "build_dataset",
    "make_dataset_tensors",
    "dsmdl_loss",
    "train_unet",
    # reference phaseless port
    "ForwardConfig",
    "TrainingConfigLegacy",
    "UNet3Ab",
    "compute_inputs_from_mat",
    "generate_forward_dataset",
    "train_unet3ab",
]
