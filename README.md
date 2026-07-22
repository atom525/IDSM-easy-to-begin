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
| **Phaseless DSM + DSM-DL** | Paper 4, Section 3-5 | Corrected phaseless data (Eq. 3.11–3.12) and U-Net for inhomogeneous + impenetrable scatterers |

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
python -c "import sys; sys.path.insert(0,'.'); from src.idsm import run_idsm; print('OK')"
```

## Quick Start

The recommended way to get started is through the Jupyter notebooks, which now use a consistent **theory-to-code** teaching structure:

- paper equation statement and symbols,
- function contract in `src/`,
- low-level step-by-step execution with intermediate variables,
- wrapper equivalence checks,
- final experiment and figure section.

```bash
cd IDSM-easy-to-begin/notebooks
jupyter notebook 01_forward_problem.ipynb   # Phase 1: FEM, mesh, forward solver
```

The six notebooks are designed to be followed in order:

| Notebook | Content | Approx. Time |
|----------|---------|--------------|
| `01_forward_problem.ipynb` | Eq. (2.2) weak form to `K,b,B` assembly, explicit Neumann solve, manual-vs-wrapper Cauchy noise pipeline | ~2–4 min |
| `02_classical_dsm.ipynb` | Eq. (2.6)–(2.10): Laplace-Beltrami, scattering data, numerator/denominator assembly, manual indicator vs `compute_dsm_indicator` | ~4–8 min |
| `03_iterative_dsm.ipynb` | Algorithm 3.2 internals: regularized DtN, P0 gradients, low-rank update, two-step transparent run and full IDSM applications | ~10–18 min |
| `04_comparative_study.ipynb` | Paper 3 unit-disk Algorithm 5.1 chain plus shared-ellipse DSM/IDSM comparisons, stabilization ablation, alpha sweep, and Paper 1 double-coefficient example | ~2–5 min teaching profile; full figure script is longer |
| `05_parabolic_idsm.ipynb` | Algorithm 4.1 segment walkthrough (`ThFine -> Th -> ThCoarse`), adjoint history, finalization/forgetting, nonlinear solver, and cached Examples 5.1–5.5 | ~3–8 min with validated caches |
| `06_phaseless_dsmdl.ipynb` | Theory-first dual track: strict VIE/BIE DSM chain with wrapper assertion, traced U-Net tensors, and physical small-grid MoM mapping to released source lines | ~2 min from cache; full-scale training scripts ~1 h |

Alternatively, you can use the package API directly:

```python
from src.mesh import generate_elliptic_mesh
from src.forward_solver import make_conductivity_example1, generate_cauchy_data
from src.idsm import run_idsm

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
    phaseless_scattering.py  # Helmholtz forward (LU Lippmann-Schwinger) + DSM index (Eq. 3.11)
    phaseless_bie.py     # BIE Dirichlet/Neumann solvers for Section 5.1 impenetrable obstacles
    phaseless_dsmdl.py   # DSM-DL: datasets, U-Net (paper Fig.1), TV+window-SSIM loss, training/eval
    phaseless_paper_figs.py  # Shared assemblers used by the script and Notebook 06 for Fig.6-10
    utils.py             # Visualization, distance, IoU metrics
  notebooks/
    01_forward_problem.ipynb     # Phase 1: FEM, mesh, forward solver
    02_classical_dsm.ipynb       # Phase 2: DSM baseline
    03_iterative_dsm.ipynb       # Phase 3: IDSM with DtN map
    04_comparative_study.ipynb   # Phase 4: Full comparison
    05_parabolic_idsm.ipynb      # Phase 5: Parabolic IDSM (moving inclusions)
    06_phaseless_dsmdl.ipynb     # Phase 6: Phaseless DSM + DSM-DL (Helmholtz, U-Net, Fig.2-10)
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
    test_phaseless_scattering.py  # DSM corrected data shape, noise model, peak diagnostics
    test_phaseless_dsmdl.py       # U-Net forward shape, loss finite, dataset shape
  reference/
    Example1.edp ... Example5.edp   # FreeFEM reference code
  figures/               # Generated figures from notebooks
    parabolic/           # Paper Section 5 multi-frame reproduction figures
    06_phaseless/        # Phaseless DSM + DSM-DL paper figures (Fig.2-10)
  results/
    parabolic/           # Reproducible NPZ caches consumed by figures/05_parabolic
    phaseless/           # paper_summary.json, paper_comparison.md, reference_*.json/md
  scripts/
    run_all_examples.py              # Regenerate Paper Section 5 NPZ caches with ThFine/Th/ThCoarse
    plot_parabolic_paper_figures.py  # Regenerate figures/05_parabolic from NPZ caches
    run_nb04_figures.py              # Regenerate NB04 comparative-study figures (figures/04_comparative/04_*.png)
    run_phaseless_full_repro.py      # Phase 6: paper-scale Helmholtz reproduction (Fig.2-10 + checkpoints)
    run_phaseless_forward_generate.py # Phase 6: pure Python port of ISP_forward/DataMnist.m
    run_phaseless_port_repro.py      # Phase 6: pure Python port of DSMDL_phaseless/main.py
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
- `run_idsm_parabolic(coarse_mesh, fine_mesh, cfg, c_func, v_func, solve_mesh=..., ...)` -- Algorithm 4.1, time-segmented IDSM for moving inclusions. `fine_mesh` is the data mesh `ThFine`, `solve_mesh` is the P1 PDE mesh `Th`, and `coarse_mesh` is the P0 coefficient mesh `ThCoarse`; the function auto-dispatches to σ/V double-field or U-recovery single-field path via `cfg.model`
- `solve_forward_segment(...)` -- Crank–Nicolson forward solver per segment (linear σ + V)
- `solve_forward_segment_nonlinear(...)` -- Newton + Crank–Nicolson solver for §5.3 nonlinear $|y|y\cdot U$ source
- `iterate_segment_nonlinear(...)` -- U-recovery inner loop (single-field P0 reconstruction projected to $[u_A, 2u_B]$)
- `edp_cfg_example_5_{1..5}(noise=...)` preserves the supplied `.edp` parameters. `paper_cfg_example_5_{1..5}(noise=...)` uses the Paper 2 time steps (`0.01` for data, `0.0125` inside each inverse segment), inter-segment damping `0.6`, and the paper noise choices; Example 5.1 also uses tolerance `0.10`.
- `solve_adjoint_segment(...)` -- Backward adjoint PDE (Eq. 4.1) with terminal condition
- `synthesize_full_forward(...)` -- Reference fine-mesh forward pass that produces noisy boundary measurements
- `c_func_example_5_{1..5}` / `v_func_example_5_{1..5}` / `u_func_example_5_3` -- Ground-truth coefficient fields used both for synthesizing data and for IoU evaluation
- `trajectory_example_5_{1..5}` / `radius_example_5_{1,4,5}` -- Inclusion centre/radius trajectories per example
- `ParabolicConfig` -- Parabolic IDSM hyperparameters (defaults match `parabolic_*.edp`)

### Parabolic Paper-Profile Artifacts

Notebook 05 keeps unique diagnostics under `figures/05_parabolic/05_*.png`. The canonical multi-frame galleries are generated once as `05_paper_*.png` from paper-profile three-mesh caches:

```bash
# Main paper-noise caches, including Example 5.1 at 5% and 10% noise.
python scripts/run_all_examples.py --mode paper --include-ex51-n10 --mesh-mode paper --quiet

python scripts/plot_parabolic_paper_figures.py
```

The run command writes only the latest paper-profile caches to `results/parabolic/*.npz`. The plot command refreshes `05_paper_*.png`, `table1.txt`, and the canonical `results/parabolic/summary.json`.

### `src/phaseless_scattering.py` (Paper 4: arXiv:2403.02584)
- `PhaselessBatchSimulator` -- Hybrid forward solver: medium scatterers go through a batched Lippmann-Schwinger LU solve using the paper Eq. (2.4) coefficient contrast $k^2(n-1)$ (`torch.linalg.lu_factor` cached per sample, reused across incidences), while sound-soft scatterers ($u=0$ on $\partial D$, paper Eq. 2.3) are dispatched to the Dirichlet **boundary-integral equation** in `phaseless_bie.py`. Mixed-circle scenes use a Born-superposition approximation (medium VIE field + per-soft-scatterer BIE field). Both `compute_dsm_inputs(labels, ...)` (legacy complex-$n$ surrogate) and `compute_dsm_inputs_with_meta(labels, metas, ...)` (strict BIE+VIE Born) yield either Eq. (3.11) phaseless or Eq. (3.1) phased DSM indices.
- `run_example_dsm_paper(example_key, noise_level, n_incident, ...)` -- Section 5.1 dispatcher: VIE for medium square / close squares / ring, and BIE Dirichlet for sound-soft squares.
- `add_phaseless_noise`, `corrected_phaseless_data`, `compute_phaseless_dsm_indicator` -- Building blocks for the paper's Eq. (3.11)-(3.12).

### `src/phaseless_bie.py`
- `boundary_circle`, `boundary_square`, `stack_boundaries` -- Boundary discretizations used by Section 5.1.
- `solve_sound_soft` -- Single-layer Dirichlet BIE $S\,\varphi = -u_{\text{inc}}$.
- `solve_sound_hard` -- Indirect single-layer Neumann BIE $(-I/2 + K^T)\,\varphi = -\partial_n u_{\text{inc}}$.
- `evaluate_total_at` -- Reconstruct the total field at the receiver ring from the boundary density.

### `src/phaseless_dsmdl.py`
- `DatasetConfig`, `TrainingConfig` -- Strict paper-protocol settings (wavelength, receiver geometry, $N_i$, $n_{\text{soft}}$, batch / epochs / LR schedule).
- `build_dataset` / `build_dataset_with_norm` -- Generate the polygon, MNIST (official train/test split), and mixed-circle labels and the matching DSM indicator inputs.
- `UNetDSMDL(in_channels, out_channels=1, base_channels=64, depth=4)` -- U-Net mirroring paper Fig.1 (64-128-256-512-1024).
- `dsmdl_loss` -- Eq. (4.1) $\text{MSE} + 0.5\,\text{TV}(X) + 0.5(1-\text{SSIM}_{11\times 11}(X,Y))$.
- `BalancedPolygonBatchSampler`, `make_polygon_dataloader` -- 5+5 medium/soft per-batch sampler (paper §5.2.1.1).
- `train_unet`, `predict`, `threshold_to_classes`, `pixel_accuracy`, `relative_l2_error` -- Training loop and Eq. (5.3)/(5.4) metrics.
- `save_model_checkpoint`, `load_model_checkpoint` -- Persist trained networks for fast reuse.

### `src/phaseless_paper_figs.py`
- Shared assemblers for paper Fig.6–Fig.10. Both `scripts/run_phaseless_full_repro.py` and `notebooks/06_phaseless_dsmdl.ipynb` import from here, so the layout stays identical between the offline run and the inline notebook view.
- `load_case_from_checkpoint(path, in_channels, device)` -- Rehydrate a model + dataset metadata from `.pt` for figure rendering.
- `render_and_save_all(ckpt_dir, out_fig, seed, device)` -- Convenience entry that regenerates all DSM-DL paper figures from checkpoints.

### Phase 6 reproduction artifacts

The strict run is driven by a single script and produces every paper output (Fig.2–Fig.10, Table 1, Table 2, Section 5.2.3 metrics, and the trained checkpoints):

```bash
# Strict paper-protocol reproduction. ~60 minutes on an A100.
conda run -n IDSM python scripts/run_phaseless_full_repro.py
```

The reference-code path is also available in pure Python:

```bash
# Optional: regenerate forward data without MATLAB.
conda run -n IDSM python scripts/run_phaseless_forward_generate.py --n-samples 200

# Reproduce DSMDL_phaseless/main.py behavior (U_Net3Ab + legacy loss weights).
conda run -n IDSM python scripts/run_phaseless_port_repro.py
```

It writes:

- `results/phaseless/paper_summary.json` and `results/phaseless/paper_comparison.md` (Table 1, Table 2, mixed-circle metrics vs paper),
- `results/phaseless/checkpoints/{polygon_Ni1, polygon_Ni4, mnist_Ni4, mnist_Ni16, mixed_circle_Ni10}.pt`,
- `figures/06_phaseless/06_paper_*.png` contains the canonical Paper Figs. 2–10 galleries; `06_reference_mnist.png` is the separate released-code path output.

Notebook 06 §6 then renders Fig.6–Fig.10 inline from those checkpoints in well under a minute, without retraining.

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
| `01_forward_problem.ipynb` | 1 | Eq. (2.2) weak form, explicit FEM assembly, Neumann gauge solve, boundary noise formula walkthrough |
| `02_classical_dsm.ipynb` | 2 | Eq. (2.6)–(2.10) DSM chain with low-level numerator/denominator construction and wrapper checks |
| `03_iterative_dsm.ipynb` | 3 | Algorithm 3.2 internals: double-Robin DtN, low-rank preconditioner updates, projected iterations, convergence |
| `04_comparative_study.ipynb` | 4 | Algorithm 5.1 partial-data IDSM on the unit disk, shared-ellipse comparisons, stabilization/HR-DtN ablations, alpha sweep |
| `05_parabolic_idsm.ipynb` | 5 | Algorithm 4.1 segment decomposition and nonlinear extension, then full paper-style moving-inclusion examples |
| `06_phaseless_dsmdl.ipynb` | 6 | Phaseless DSM + DSM-DL: strict paper chain, per-stage network traces, released MoM/source-line mapping, and paper-scale metrics |

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

4. B. Jin, F. Wang, J. Zou, "An iterative direct sampling method for reconstructing moving inhomogeneities in parabolic problems," 2025. [arXiv:2511.08197](https://arxiv.org/abs/2511.08197)

5. J. Ning, F. Han, J. Zou, "A direct sampling method and its integration with deep learning for inverse scattering problems with phaseless data," *SIAM J. Sci. Comput.*, 2025. [arXiv:2403.02584](https://arxiv.org/abs/2403.02584)

**FreeFEM reference code**:
- Elliptic: [github.com/RaulWangfr/IDSM-elliptic](https://github.com/RaulWangfr/IDSM-elliptic)
- Parabolic: [github.com/RaulWangfr/IDSM-parabolic](https://github.com/RaulWangfr/IDSM-parabolic)
- Phaseless DSM-DL: no public reference repository identified; Phase 6 is a paper-faithful re-implementation.

## License

This project is for educational purposes.
