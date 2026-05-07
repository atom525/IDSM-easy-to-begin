# Demystifying Iterative Direct Sampling Methods -- From Theory to Code

An educational Python package implementing Direct Sampling Methods (DSM) and Iterative Direct Sampling Methods (IDSM) for elliptic inverse problems, with comprehensive Jupyter notebook tutorials.

## Project Rationale

The recently developed Iterative Direct Sampling Methods (IDSM) by Jin, Wang, Zou, and Ito represent a significant advance in solving inverse problems like Electrical Impedance Tomography (EIT). They combine the speed of direct methods with the accuracy of iterative ones. However, the mathematical depth of the original papers can be a barrier. This project bridges this gap by providing a clear, well-documented, open-source educational implementation.

**Target Audience**: Graduate students and researchers new to inverse problems or direct sampling methods.

**Core Philosophy**: Clarity over speed. Code is simple, well-commented, and modular. Mathematical steps in the papers are explicitly linked to lines of code.

## Mathematical Background

### Problem Setting

The primary model is the Electrical Impedance Tomography (EIT) problem on an ellipse $\Omega = \{x : x_1^2 + x_2^2/0.64 < 1\}$ (Paper 1, Example 1):

$$-\nabla \cdot (\sigma(x)\nabla y) = 0 \quad \text{in } \Omega, \qquad \sigma \frac{\partial y}{\partial \nu} = f \quad \text{on } \Gamma$$

Given boundary measurements (Cauchy data) $\{(f_\ell, y_\ell^d)\}_{\ell=1}^L$, the goal is to recover the unknown conductivity $\sigma(x)$.

The framework extends to the generalized model with a zeroth-order potential term (Paper 1, Examples 2–3):

$$-\nabla \cdot (\sigma(x)\nabla y) + q(x)y = 0 \quad \text{in } \Omega$$

which covers diffuse optical tomography (DOT, $\sigma=1$, recover $q$) and simultaneous recovery of both $\sigma$ and $q$.

### Methods Implemented

| Method | Reference | Key Idea |
|---|---|---|
| **DSM** (Direct Sampling) | Paper 1, Section 2.2 | Single-pass indicator via Green's function correlation |
| **IDSM** (Full Data) | Paper 1, Algorithm 3.2 | Iterative refinement with regularized DtN map + quasi-Newton |
| **IDSM** (Partial Data) | Paper 3, Algorithm 5.1 | Data completion + HR-DtN + stabilization-correction |
| **IDSM** (Parabolic) | Paper 2, Algorithm 4.1 | Time-segmented inversion of moving inclusions via backward adjoint |

## Installation

### Requirements

- Python 3.8+
- NumPy, SciPy, Matplotlib, MeshPy, scikit-fem

### Setup

```bash
# Create conda environment (recommended)
conda create -n IDSM python=3.10
conda activate IDSM

# Install dependencies
pip install -r requirements.txt

# Verify
python -c "from IDSM.src import run_idsm; print('OK')"
```

## Quick Start

The recommended way to get started is through the Jupyter notebooks, which provide step-by-step theory and code:

```bash
cd IDSM/notebooks
jupyter notebook 01_forward_problem.ipynb   # Phase 1: FEM, mesh, forward solver
```

The five notebooks are designed to be followed in order:

| Notebook | Content | Approx. Time |
|----------|---------|--------------|
| `01_forward_problem.ipynb` | Weak form, P1 FEM, saddle-point system, noise model | ~1 min |
| `02_classical_dsm.ipynb` | Laplace-Beltrami, DSM indicator, limitations | ~2 min |
| `03_iterative_dsm.ipynb` | Regularized DtN map, IDSM Algorithm 3.2, DFP/BFG | ~8 min |
| `04_comparative_study.ipynb` | DSM vs IDSM, partial data, ablation, noise sweep | ~15 min |
| `05_parabolic_idsm.ipynb` | Parabolic IDSM (Algorithm 4.1), Paper §5 Examples 5.1–5.5 (merging / mixed / nonlinear / fading / diminishing) | ~3 min |

Alternatively, you can use the package API directly:

```python
from IDSM.src.mesh import generate_elliptic_mesh
from IDSM.src.forward_solver import make_conductivity_example1, generate_cauchy_data
from IDSM.src.idsm import run_idsm

# Generate mesh and ground truth
mesh = generate_elliptic_mesh(n_boundary=256)
sigma_true, u_true = make_conductivity_example1(mesh)

# Generate synthetic Cauchy data with 10% noise
sources = [lambda x, y: x, lambda x, y: y]
data = generate_cauchy_data(mesh, sigma_true, sources, noise_level=0.1)

# Run IDSM (Algorithm 3.2)
history = run_idsm(mesh, data, n_iter=22, sigma_range=0.01,
                   problem_type="conductivity", lowrank_method="BFG")

# Result: history['sigma_final'] is the reconstructed conductivity (P0 array)
print(f"Residual: {history['residuals'][0]:.4e} -> {history['residuals'][-1]:.4e}")
```

## Repository Structure

```
IDSM/
  src/
    __init__.py          # Package exports
    config.py            # Centralized hyperparameters (RuntimeConfig, etc.)
    mesh.py              # Mesh generation, boundary extraction, coarse mesh
    fem.py               # P1 FEM public API (delegates to scikit-fem backend)
    fem_skfem.py         # scikit-fem backed FEM assembly (default backend)
    fem_legacy.py        # Hand-written P1 FEM (retained for regression testing)
    forward_solver.py    # Forward PDE solver, Cauchy data generation
    dsm.py               # Classical DSM (Laplace-Beltrami, indicator)
    idsm.py              # Full-data IDSM (Algorithm 3.2)
    idsm_partial.py      # Partial-data IDSM (Algorithm 5.1)
    idsm_parabolic.py    # Parabolic IDSM (Algorithm 4.1, Paper 2)
    utils.py             # Visualization, distance, IoU metrics
  notebooks/
    01_forward_problem.ipynb     # Phase 1: FEM, mesh, forward solver
    02_classical_dsm.ipynb       # Phase 2: DSM baseline
    03_iterative_dsm.ipynb       # Phase 3: IDSM with DtN map
    04_comparative_study.ipynb   # Phase 4: Full comparison
    05_parabolic_idsm.ipynb      # Phase 5: Parabolic IDSM (moving inclusions)
  tests/
    test_mesh.py             # Mesh area, boundary, coarsening
    test_fem.py              # Stiffness symmetry, mass, Neumann solver
    test_fem_regression.py   # skfem vs legacy numerical agreement
    test_forward.py          # Forward solver, noise model
    test_dsm.py              # Eigenvalues, indicator
    test_idsm.py             # Box constraints, residual decrease
    test_idsm_partial.py     # Data completion, HR-DtN, lambda
    test_utils.py            # Distance, IoU, grid projection
    test_config.py           # Configuration defaults and env vars
  reference/
    Example1.edp ... Example5.edp   # FreeFEM reference code
  figures/               # Generated figures from notebooks
  requirements.txt
  README.md
```

## Module API Reference

### `src/mesh.py`
- `generate_elliptic_mesh(n_boundary)` -- Create unstructured triangular mesh on the ellipse
- `generate_sampling_grid(n_grid)` -- Regular grid for DSM indicator evaluation
- `generate_coarse_mesh(target_triangles)` -- Coarse mesh for Paper 3 stabilization
- `fine_to_coarse_p0` / `coarse_to_fine_p0` -- P0 inter-mesh projection

### `src/fem.py`

FEM public API, backed by scikit-fem (default) or hand-written legacy implementation (`IDSM_FEM_LEGACY=1`).

- `assemble_stiffness_matrix(mesh, sigma)` -- Stiffness matrix $K_{ij} = \int \sigma \nabla\phi_j \cdot \nabla\phi_i$
- `assemble_mass_matrix(mesh, coeff)` -- Mass matrix $M_{ij} = \int q\,\phi_j \phi_i$
- `assemble_boundary_mass_matrix(mesh)` -- Boundary mass $M_\Gamma$
- `assemble_partial_boundary_mass_matrix(mesh, gamma_d_node_mask)` -- Split $M_\Gamma \to M_D + M_N$ (Paper 3)
- `compute_boundary_normal_flux(mesh, sigma, y)` -- Boundary normal flux $\sigma\,\partial y/\partial n$
- `compute_boundary_normal_derivative(mesh, z, sigma_bg)` -- Generic $\sigma_0\partial z/\partial n$ (geometry-independent)
- `solve_neumann_system(K, b, B)` -- Saddle-point solve $[K,B; B^T,0]$
- `solve_robin_system(mesh, A, alpha, v)` -- Robin BVP for DtN map

### `src/idsm.py`
- `run_idsm(mesh, cauchy_data, ...)` -- Main IDSM loop (Algorithm 3.2)
- `apply_regularized_dtn(mesh, v, A_op, alpha)` -- Double Robin BVP (Eq. 3.20)
- `compute_p0_gradient(mesh, w_list, y_list)` -- Gradient $\zeta_k$ (Eq. 3.17)
- `LowRankPreconditioner` -- DFP/BFG quasi-Newton (Eq. 3.14-3.15)

### `src/idsm_partial.py`
- `run_idsm_partial(mesh, cauchy_data, gamma_d_info, ...)` -- Paper 3 Algorithm 5.1
- `define_accessible_boundary(mesh, theta_range)` -- Define $\Gamma_D$ from angle range
- `complete_data(y_data, y_current, mask)` -- Data completion (Eq. 4.1)
- `apply_hr_dtn(mesh, v, A_op, alpha_d, alpha_n, ...)` -- Heterogeneous DtN (Eq. 4.2)
- `StabilizedLowRankResolver` -- Stabilization-correction scheme (Eq. 4.10-4.16)

### `src/idsm_parabolic.py` (Paper 2: arXiv:2511.08197)
- `run_idsm_parabolic(coarse_mesh, y_data, forward_dt, f_func, g_func, initial_data, cfg, ...)` -- Algorithm 4.1, time-segmented IDSM for moving inclusions; auto-dispatches to σ/V double-field or U-recovery single-field path via `cfg.model`
- `solve_forward_parabolic_segment(...)` -- Crank–Nicolson forward solver per segment (linear σ + V)
- `solve_forward_segment_nonlinear(...)` -- Newton + Crank–Nicolson solver for §5.3 nonlinear $|y|y\cdot U$ source
- `iterate_segment_nonlinear(...)` -- U-recovery inner loop (single-field P0 reconstruction projected to $[u_A, 2u_B]$)
- `edp_cfg_example_5_{1..5}(noise=...)` -- Paper §5 Examples 5.1–5.5 configurations matched to FreeFEM `parabolic_*.edp`
- `solve_backward_adjoint_segment(...)` -- Backward adjoint PDE (Eq. 4.1) with terminal condition
- `compute_local_dual(mesh, y_hist, z_hist, normal_scale)` -- Time-integrated $\zeta_c$ and $\zeta_v$ (Eq. 4.2-4.4)
- `apply_inclusion_projection(eta, n_tri, model, cA, cB, vA, vB)` -- Box projection per model type
- `damp_lowrank_state(R, forget_scale)` -- Inter-segment forgetScale damping
- `inclusion_trajectory_example1(t)` / `inclusion_trajectory_example4(t)` -- Truth trajectories (Examples 5.1, 5.4)
- `ParabolicConfig` -- Parabolic IDSM hyperparameters (defaults match `parabolic_*.edp`)

### `src/dsm.py`
- `compute_dsm_indicator(mesh, cauchy_data, gamma, ...)` -- DSM indicator $\eta(x)$ (Eq. 2.8)
- `discretize_laplace_beltrami(mesh, gamma)` -- $(-\Delta_\Gamma)^\gamma$ eigendecomposition
- `LaplaceBeltramiOperator` -- Discrete $(-\Delta_\Gamma)^\gamma$ operator with `apply()` method
- `compute_scattering_data(cauchy_data)` -- Scattering $y_d^s = y_\emptyset - y_d$
- `compute_dsm_numerator(mesh, scatter, lb_op)` -- Auxiliary PDE solve (Eq. 2.9)
- `compute_dsm_denominator_integral(mesh, points)` -- FreeFEM-style normalization (Eq. 2.10)

### `src/forward_solver.py`
- `solve_forward(mesh, sigma, f_func)` -- EIT forward solve: $\nabla\cdot(\sigma\nabla y)=0$
- `solve_forward_general(mesh, sigma, potential, f_func)` -- Generalized with zeroth-order term
- `generate_cauchy_data(mesh, sigma, sources, noise_level)` -- Noisy Cauchy data pairs
- `generate_cauchy_data_general(mesh, sigma, potential, sources, noise_level)` -- DOT Cauchy data
- `make_conductivity_example1(mesh)` -- Example 1 (two insulating squares)
- `make_conductivity_conductive(mesh)` -- Conductive variant ($\sigma=3.0$)
- `make_conductivity_single(mesh)` -- Single circular inclusion
- `make_double_example2(mesh)` -- Example 2 (simultaneous $\sigma$ + $v$, double type)
- `make_potential_example3(mesh)` -- Example 3 (potential-only, DOT)

### `src/utils.py`
- `compute_iou(u_true, u_recon, mesh)` -- Area-matched Intersection over Union
- `distance_to_boundary(mesh, points)` -- Min distance from points to boundary edges

### `src/config.py`
- `RuntimeConfig` -- GPU/seed/backend settings (from environment variables)
- `MeshConfig` -- Mesh resolution parameters
- `FullIDSMConfig` -- Full-data IDSM hyperparameters (Algorithm 3.2)
- `PartialIDSMConfig` -- Partial-data IDSM hyperparameters (Algorithm 5.1)
- `DoubleIDSMConfig` -- Example 2 (double type) hyperparameters (FreeFEM Example2.edp)
- `Notebook01Config` ... `Notebook04Config` -- Per-notebook configuration dataclasses

## Notebook Guide

| Notebook | Phase | Content |
|---|---|---|
| `01_forward_problem.ipynb` | 1 | Weak form, P1 FEM, saddle-point system, mesh convergence, noise model |
| `02_classical_dsm.ipynb` | 2 | Laplace-Beltrami eigenproblem, DSM indicator, gamma parameter, limitations |
| `03_iterative_dsm.ipynb` | 3 | Regularized DtN map, Algorithm 3.2, DFP/BFG, convergence analysis |
| `04_comparative_study.ipynb` | 4 | DSM vs IDSM, partial data, damping factor, ablation, noise sweep |
| `05_parabolic_idsm.ipynb` | 5 | Crank-Nicolson forward, backward adjoint (Eq. 4.1), Algorithm 4.1, Example 5.1 |

## Configuration

All hyperparameters are centralized in `src/config.py`:

```python
from IDSM.src.config import Notebook04Config, RuntimeConfig

cfg = Notebook04Config()
cfg.full.alpha = 1.0           # Robin regularization
cfg.full.n_iter = 22           # Iterations (FreeFEM: storeNum=22)
cfg.partial.alpha_d = 0.05     # Paper 3 Table 1
cfg.partial.alpha_n = 2.0
```

GPU toggle (currently CPU-only due to sparse solver constraints):
```python
runtime = RuntimeConfig(use_gpu=True)  # Will warn and fall back to CPU
```

## Testing

```bash
cd IDSM
pytest tests/ -v            # Default: scikit-fem backend (85 tests)
IDSM_FEM_LEGACY=1 pytest tests/ -v   # Legacy hand-written FEM backend
```

## FAQ

**Q: Why scikit-fem instead of FEniCSx?**

A: scikit-fem is a pure Python library (pip-installable, no compiled dependencies) that covers all P1 triangular FEM operations needed by this project. FEniCSx requires PETSc/MPI compilation and has heavier installation requirements, which creates unnecessary barriers for an educational package. Both are mathematically equivalent for the P1 case; our regression tests verify numerical agreement to machine precision.

**Q: Can I switch back to the hand-written FEM?**

A: Yes. Set `IDSM_FEM_LEGACY=1` as an environment variable. The adapter layer in `fem.py` will delegate to `fem_legacy.py` instead of `fem_skfem.py`. All tests pass with both backends.

## References

1. K. Ito, B. Jin, F. Wang, J. Zou, "Iterative direct sampling method for elliptic inverse problems with limited Cauchy data," *SIAM J. Imaging Sci.* 18(2), 2025. [arXiv:2503.00423](https://arxiv.org/abs/2503.00423)

2. B. Jin, F. Wang, J. Zou, "A direct sampling method for simultaneously recovering inhomogeneous inclusions of different nature," *SIAM J. Sci. Comput.* 43(3), 2021. [arXiv:2005.05499](https://arxiv.org/abs/2005.05499)

3. B. Jin, F. Wang, J. Zou, "A stable iterative direct sampling method for elliptic inverse problems with partial Cauchy data," *J. Comput. Phys.* 550, 2026. [arXiv:2511.08171](https://arxiv.org/abs/2511.08171)

4. B. Jin, F. Wang, J. Zou, "An iterative direct sampling method for reconstructing moving inhomogeneities in parabolic problems," 2025. [arXiv:2505.06406](https://arxiv.org/abs/2505.06406)

**FreeFEM reference code**:
- Elliptic: [github.com/RaulWangfr/IDSM-elliptic](https://github.com/RaulWangfr/IDSM-elliptic)
- Parabolic: [github.com/RaulWangfr/IDSM-parabolic](https://github.com/RaulWangfr/IDSM-parabolic)

## License

This project is for educational purposes.
