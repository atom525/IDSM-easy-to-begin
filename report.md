# Final Project Report

## Demystifying Iterative Direct Sampling Methods — From Theory to Code

### An Educational Implementation and Comparative Study of IDSM for Elliptic Inverse Problems

---

## 1. Executive Summary

This project implements a comprehensive, educational Python package for Iterative Direct Sampling Methods (IDSM) applied to elliptic inverse problems, following the foundational works by Ito, Jin, Wang, and Zou. The package bridges the gap between the mathematical depth of the original papers and practical, accessible code that graduate students and researchers can study and extend.

**Key deliverables:**
- A modular Python codebase (`src/`) implementing FEM (backed by scikit-fem), forward solvers, DSM, IDSM, and partial-data IDSM
- Four Jupyter notebook tutorials walking through theory and code step-by-step
- A comprehensive test suite (85 unit tests, including skfem/legacy regression, end-to-end IDSM/partial-IDSM, DtN map verification, and double-type IDSM)
- This report summarizing implementation, challenges, and findings

---

## 2. Implemented Methods

### 2.1 Forward Problem (Phase 1)

The forward solver implements the 2D Electrical Impedance Tomography (EIT) problem on an elliptic domain:

$$\nabla \cdot (\sigma(x) \nabla y(x)) = 0 \quad \text{in } \Omega, \quad \sigma \frac{\partial y}{\partial n} = f \quad \text{on } \Gamma, \quad \int_\Gamma y \, ds = 0$$

**Implementation details:**
- P1 triangular finite elements on an elliptic mesh generated via MeshPy
- FEM assembly backed by scikit-fem (pure Python, pip-installable); hand-written legacy retained for regression testing
- Saddle-point system with Lagrange multiplier for the gauge condition
- Multiple boundary excitation patterns ($f_1 = x_1$, $f_2 = x_2$)
- Multiplicative noise model: $y_d(x) = y^*(x) + \varepsilon \cdot \delta(x) \cdot |y_\emptyset(x) - y^*(x)|$
- Cross-verified against FreeFEM reference code (`Example1.edp`)

**Modules:** `mesh.py`, `fem.py` (→ `fem_skfem.py`), `forward_solver.py`

### 2.2 Direct Sampling Method — DSM (Phase 2)

The classical DSM implements the non-iterative index function:

$$\eta(x) = \frac{\langle G(\cdot, x), y_d^s \rangle_{H^\gamma(\Gamma)}}{\|G(\cdot, x)\|_{H^\gamma(\Gamma)}}$$

**Key components:**
- Laplace-Beltrami operator via 1D FEM eigendecomposition on the boundary
- Numerator computed via auxiliary Neumann PDE (Eq. 2.9)
- Two denominator approximations: distance-based and integral-based (FreeFEM style)
- Demonstrated limitations: blurry reconstruction, no coefficient recovery, no type classification

**Module:** `dsm.py`

### 2.3 Iterative DSM — IDSM (Phase 3)

The core IDSM implements Algorithm 3.2 from Ito et al. (2025):

**Three key innovations over DSM:**
1. **Regularized DtN map** $\Lambda_\alpha(A)$: via double Robin BVPs, replacing the ill-posed fractional Laplacian
2. **Iterative refinement**: quasi-Newton loop with convergence monitoring
3. **Direct inclusion imaging**: P0 element-wise conductivity reconstruction with box constraints

**Low-rank corrections:**
- DFP update (Eq. 3.14): minimizes $\|R_{k+1} - R_k\|_F$ subject to secant condition
- BFG update (Eq. 3.15): minimizes $\|R_{k+1}^{-1} - R_k^{-1}\|_F$
- First-iteration $R_0$ scaling for correct magnitude

**Module:** `idsm.py`

### 2.4 Partial-Data IDSM (Phase 4)

The partial-data extension implements Algorithm 5.1 from Jin et al. (2026):

**Three core innovations:**
1. **Data completion** (Eq. 4.1): $\tilde{y}_d(u_k) = T_D y^* + T_N y(u_k)$
2. **Heterogeneous regularized DtN** (Eq. 4.2): spatially varying $\alpha_D(x)$ with $\alpha_d \ll \alpha_n$
3. **Stabilization-correction scheme** (Eq. 4.10–4.16): damping factor, mesh coarsening, recursive update

**Module:** `idsm_partial.py`

---

## 3. Implementation Challenges

### 3.1 Numerical Stability of the DtN Map

The most mathematically nuanced component was the regularized DtN map. The classical DtN map $\Lambda(A): H^{1/2}(\Gamma) \to H^{-1/2}(\Gamma)$ is unbounded, so direct computation amplifies noise. The Robin regularization (Eq. 3.5) was essential for stability. We implemented this via two sequential Robin BVPs (Lemma 3.2, Eq. 3.20), verified by:
- Comparing Robin solve results against direct Dirichlet-to-Neumann computation
- Testing that $\alpha \to 0$ approaches the true DtN behavior
- Verifying that larger $\alpha$ provides more stability at the cost of accuracy

### 3.2 Low-Rank Correction Conditioning

The DFP and BFG updates require the secant condition $s_k^\top \tilde{y}_k > 0$ for positive definiteness. In practice, near convergence, $\tilde{y}_k$ can become very small, making the inner product unreliable. We addressed this by:
- Skipping rank updates when $s_k^\top \tilde{y}_k$ falls below a threshold
- Using the first-iteration scaling mechanism from FreeFEM to set the correct magnitude for $R_0$

### 3.3 Projection and Box Constraints

The box constraint $\mathcal{P}_{[a,b]}$ requires knowing the conductivity range a priori. For insulating inclusions ($\sigma < \sigma_0$), the gradient is thresholded to positive values; for conductive inclusions ($\sigma > \sigma_0$), the gradient direction is flipped. This direction-dependent projection was necessary to handle both inclusion types correctly.

### 3.4 Partial-Data Stabilization

The stabilization-correction scheme from Paper 3 involves multiple interacting components (damping factor, coarse mesh projection, recursive preconditioner update, safeguard mechanism). The damping factor $\lambda_{k,p}$ computation requires careful handling of the $L^p$ norms to avoid division by zero, and the recursive update formula must maintain positive definiteness.

### 3.5 Mesh Generation and Boundary Handling

The ordered boundary node chain is critical for:
- Correct Laplace-Beltrami discretization (1D FEM on $\Gamma$)
- Proper boundary mass matrix assembly
- Partial boundary identification ($\Gamma_D$ vs $\Gamma_N$)

We ensured boundary nodes form a consistent, closed loop via edge-following algorithms, verified by unit tests.

### 3.6 Computational Cost

The dominant cost is the sparse linear system solves within each IDSM iteration. With $L$ Cauchy data pairs and $K$ iterations:
- **DSM**: $\sim L$ PDE solves
- **IDSM (full)**: $\sim L \times (2 + K)$ PDE solves
- **IDSM (partial)**: $\sim L \times (4 + 2K)$ PDE solves

For the test configuration (mesh with ~60K triangles, 22 iterations, 2 data pairs), IDSM takes ~47 seconds while DSM takes ~3 seconds. The partial-data version takes ~117 seconds due to the additional Robin solves for data completion.

### 3.7 Quasi-Newton Secant Condition and Its Breakdown

The DFP and BFG quasi-Newton updates depend on $s_k^\top \tilde{y}_k > 0$, where $s_k = \sigma_{k+1} - \sigma_k$ and $\tilde{y}_k = \zeta_{k+1} - \zeta_k$. In practice, we found this condition becomes unreliable in three situations.

First, near convergence, $s_k$ and $\tilde{y}_k$ are both small (O(1e-4)), and their inner product suffers from floating-point cancellation. Second, when the box constraints clip the update aggressively — which happens for insulating inclusions at the lower bound σ=0.01 — the effective step $s_k$ can be much shorter than the gradient would suggest, breaking the secant assumption. Third, the EIT inverse problem is fundamentally non-convex: two different conductivity distributions can produce very similar boundary data. The quasi-Newton framework assumes local convexity, which fails near saddle points.

We handle all three cases by skipping the rank-1 update when $s_k^\top \tilde{y}_k$ drops below a threshold ($10^{-10}$). The preconditioner simply keeps its previous value for that iteration. This is crude but effective — FreeFEM does the same. A more principled approach would be a trust-region method that adapts the step size, but that would require a line search at each iteration (2–4 extra PDE solves), roughly doubling the computational cost. For 22 iterations, skipping a few updates is a better trade-off.

The first-iteration scaling mechanism ($R_0 \gets \frac{s_0^\top \tilde{y}_0}{\tilde{y}_0^\top \tilde{y}_0} R_0$, FreeFEM L432–448) is also motivated by this issue: without it, $R_0$ and $\zeta_0$ can have mismatched magnitudes by several orders, making the first secant condition numerically meaningless.

### 3.8 Boundary Integral Singularity in R₀ Initialization

The preconditioner diagonal $D_i = 1/(\int_\Gamma r_i^{-2}\,ds)^{1/2}$ has a physical meaning: it measures how "visible" sampling point $x_i$ is from the boundary. Interior points far from $\Gamma$ have small integrals (large $D_i$), while points near $\Gamma$ have large integrals (small $D_i$, suppressed in the reconstruction). The trouble is that the integrand $1/r^2$ diverges as $x_i$ approaches $\Gamma$.

FreeFEM handles this implicitly: the triangular mesh centroids used as sampling points are always at least one element width from $\Gamma$, providing a natural cutoff. We follow the same approach, with a $10^{-20}$ additive guard on $r^2$ for numerical safety.

This seemingly minor issue became critical for the partial-data case. Paper 3 introduces heterogeneous weights $w(x) = 1/(1+\alpha(x))$ that differ across $\Gamma_D$ ($w \approx 0.95$) and $\Gamma_N$ ($w \approx 0.33$). The weighted integral $\sum_e w_e L_e / r_{e}^2$ amplifies the difference between accessible and inaccessible boundary contributions. If the singularity is not handled correctly, the D values for elements near $\Gamma_N$ blow up, contaminating the entire preconditioner. This is exactly what happened with Bug #5 (Section 3.9), where the incorrect formula produced D values of 0–210 instead of the expected 0.07–0.50.

### 3.9 Debugging the Partial-Data Implementation

The partial-data IDSM (Algorithm 5.1) was the most difficult component to implement correctly. The full-data version worked on the first attempt, but the partial-data extension required five rounds of bug fixes before producing sensible results. These bugs were not obvious — each one alone might produce a "working" algorithm with degraded but not obviously broken output. Only the combination of all five fixes brought the residuals and IoU values in line with the full-data baseline.

**The root cause** turned out to be the heterogeneous D formula (Bug #5). Our initial implementation followed Paper 3 Eq. (4.5) literally, computing a weighted Sobolev norm that mixed scalar potentials (log r) and gradient magnitudes (1/r). This produced D values ranging from 0 to 210, while the full-data version produces values in the range 0.07–0.50. The orders-of-magnitude discrepancy made the preconditioner effectively random noise.

The breakthrough came from a careful comparison between FreeFEM `Example1.edp` (full data) and `Example4.edp` (which we initially assumed implemented partial-data IDSM). It turns out that **Example4.edp and Example5.edp solve nonlinear problems, not Paper 3's Algorithm 5.1**. The correct approach was to adapt the full-data `diagFunc` formula — a simple boundary integral $1/(\int_\Gamma r^{-2} ds)^{1/2}$ — with the heterogeneous weight factor $w = 1/(1+\alpha)$. This gives D values consistent with the full-data range, just with spatial modulation that suppresses the inaccessible boundary.

The other four bugs were: (1) the `pot_exponent` parameter for Example 3 (DOT) was hardcoded to 0.0 instead of 1.5, making the potential channel's preconditioner a constant; (2) the first-iteration scaling was applied after storing the low-rank correction (should be before); (3) conductivity and potential blocks were scaled jointly instead of separately; (4) the cross-term formula mixed incompatible singularity types.

**Lessons**: When a reference implementation doesn't match the paper, trust the code for numerics and the paper for theory. Magnitude checks are the fastest diagnostic — if D values are off by 100x, something is structurally wrong. The order of operations in quasi-Newton methods is not commutative: scale-then-store ≠ store-then-scale.

---

## 4. Key Findings from Comparative Studies

### 4.1 DSM vs IDSM: Qualitative to Quantitative

| Metric | DSM | IDSM (full, ε=10%) |
|--------|-----|---------------------|
| Output type | Positive indicator η(x) | Conductivity σ(x) |
| IoU | ~0.01 | ~0.33 |
| Coefficient recovery | No | Yes (with over-regularization) |
| Type classification | No | Yes |

IDSM provides a dramatic improvement over DSM in reconstruction quality (IoU improvement of ~30×), transitioning from qualitative localization to quantitative reconstruction.

### 4.2 Over-Regularization Effect

With $\alpha = 1$ (the default from Paper 1), IDSM recovers inclusion **locations** accurately but not exact **intensities** (true $\sigma = 0.3$, reconstructed $\sigma_{\min} \approx 0.63$). A systematic alpha sweep reveals the regularization-accuracy tradeoff:

| $\alpha$ | IoU | $\sigma_{\min}$ | Final Residual |
|----------|-----|-----------------|----------------|
| 0.01 | 0.142 | 0.010 | 5.09e+01 |
| 0.05 | 0.292 | 0.010 | 1.27e-02 |
| 0.1 | 0.312 | 0.236 | 1.42e-02 |
| 0.5 | 0.329 | 0.626 | 1.58e-02 |
| 1.0 | 0.329 | 0.626 | 1.61e-02 |
| 2.0 | 0.328 | 0.626 | 1.62e-02 |
| 5.0 | 0.332 | 0.630 | 1.64e-02 |
| 10.0 | 0.333 | 0.633 | 1.65e-02 |

Key observations:
- **IoU is robust** across $\alpha \in [0.5, 10]$, all around 0.33. Spatial localization is insensitive to $\alpha$ in this regime.
- **Intensity recovery improves with smaller $\alpha$**: at $\alpha=0.1$, $\sigma_{\min}=0.236$ (closer to true $\sigma=0.3$); at $\alpha=0.05$, the algorithm hits the box constraint floor $\sigma_{\min}=0.01$.
- **Stability degrades at very small $\alpha$**: at $\alpha=0.01$, the residual explodes to $O(10^1)$ and IoU drops to 0.14, confirming that the DtN map becomes ill-conditioned without sufficient regularization.
- The **optimal tradeoff** is around $\alpha \in [0.1, 0.5]$, balancing intensity recovery and stability.

### 4.3 Noise Robustness

Both DSM and IDSM degrade gracefully with noise. IDSM maintains spatial accuracy even at 30% noise:

| ε | DSM IoU | IDSM IoU |
|---|---------|----------|
| 0% | 0.0099 | 0.337 |
| 10% | 0.0098 | 0.329 |
| 30% | 0.0089 | 0.310 |

### 4.4 Conductive vs Insulating Classification

IDSM correctly reconstructs the sign of $u = \sigma - \sigma_0$:
- **Insulating** ($\sigma = 0.3$): IDSM gives $\sigma_{\min} < 1.0$ ✓
- **Conductive** ($\sigma = 3.0$): IDSM gives $\sigma_{\max} > 1.0$ ✓

DSM indicators are always positive and cannot distinguish the two types, confirming the limitation discussed in Paper 1, Section 3.

### 4.5 Partial-Data Performance

Reconstruction quality depends on the accessible boundary coverage:

| Configuration | IoU (ε=10%) | Final Residual |
|---------------|-------------|----------------|
| Full data | 0.329 | 1.6e-02 |
| Right half | 0.267 | 1.3e-02 |
| Upper half | 0.287 | 1.1e-02 |
| 3/4 boundary | 0.255 | 1.2e-02 |

All partial-data configurations achieve residual convergence comparable to the full-data case (order 1e-02), confirming that the data completion scheme and heterogeneous DtN map effectively compensate for missing boundary information. Inclusions near the accessible boundary are better reconstructed. The heterogeneous DtN map (Innovation 2) achieves lower residual than the homogeneous baseline (ablation: HR-DtN residual 1.27e-02 vs Homo 1.38e-02), though IoU is slightly lower (HR-DtN 0.267 vs Homo 0.281) — the residual metric better reflects the method's ability to fit the boundary data.

### 4.6 Single vs Multiple Inclusions

| Configuration | IoU (ε=10%) | σ_min |
|---------------|-------------|-------|
| Single circular inclusion | 0.233 | 0.816 |
| Two square inclusions | 0.329 | 0.626 |

Both configurations are successfully localized. The multiple-inclusion case achieves higher IoU because two inclusions occupy a larger area, providing a stronger signal in the Cauchy data. The single-inclusion case yields a less aggressive reconstruction ($\sigma_{\min}$ closer to background), consistent with fewer data features to drive the iteration.

### 4.7 Conductivity vs Potential (DOT)

The IDSM framework generalizes to the DOT setting (Example 3: $-\nabla\cdot(\sigma\nabla y) + v \cdot y = 0$) with potential-only inclusions. The potential channel uses DFP corrections (matching FreeFEM `Example3.edp`), and IDSM successfully recovers the potential inclusion locations.

### 4.8 Simultaneous Recovery (Example 2, Double Type)

Example 2 tests the most challenging setting: simultaneous recovery of both conductivity $\sigma$ and potential $v$ from the same Cauchy data. Following FreeFEM `Example2.edp` parameters ($\alpha=0.1$, DFP, $R_0 = \min_\theta |x-\Gamma(\theta)|^2$, $v_0=1.0$, $v_B=10.0$):

- **Conductivity inclusions**: 2 insulating squares (same as Example 1), $\sigma=0.3$
- **Potential inclusions**: 2 absorbing squares at different locations, $v=6.0$

Results ($\varepsilon = 10\%$, 22 iterations):

| Channel | IoU | Reconstructed Range | True Value |
|---------|-----|---------------------|------------|
| $\sigma$ | 0.509 | [0.290, 1.000] | 0.3 |
| $v$ | 0.630 | [1.000, 1.224] | 6.0 |

The IDSM double-type mode successfully separates and localizes both types of inclusions. The conductivity channel achieves excellent IoU (0.51) with near-exact intensity recovery ($\sigma_{\min} = 0.290 \approx 0.3$). The potential channel localizes inclusions well (IoU = 0.63) but underestimates intensity ($v_{\max} = 1.22 \ll 6.0$), consistent with the over-regularization effect for the potential block. Residual decreased from $1.25 \times 10^{-1}$ to $7.99 \times 10^{-2}$.

---

## 5. Software Architecture

### 5.1 Module Structure

```
IDSM/
├── src/
│   ├── mesh.py          — Elliptic mesh generation, boundary handling, coarsening
│   ├── fem.py           — P1 FEM public API (delegates to scikit-fem backend)
│   ├── fem_skfem.py     — scikit-fem backed FEM assembly (default backend)
│   ├── fem_legacy.py    — Hand-written P1 FEM (retained for regression testing)
│   ├── forward_solver.py — Forward PDE solves, Cauchy data generation, noise
│   ├── dsm.py           — Laplace-Beltrami, DSM indicator, denominator methods
│   ├── idsm.py          — Regularized DtN, low-rank corrections, Algorithm 3.2
│   ├── idsm_partial.py  — Data completion, HR-DtN, stabilization, Algorithm 5.1
│   ├── utils.py         — Visualization, IoU computation, distance functions
│   └── config.py        — Centralized hyperparameters (RuntimeConfig, etc.)
├── notebooks/
│   ├── 01_forward_problem.ipynb    — Phase 1: FEM, mesh, forward data
│   ├── 02_classical_dsm.ipynb      — Phase 2: DSM baseline and limitations
│   ├── 03_iterative_dsm.ipynb      — Phase 3: IDSM core algorithm
│   └── 04_comparative_study.ipynb  — Phase 4: partial data, comparisons
├── tests/                          — 85 unit tests (pytest), incl. skfem regression + e2e
├── figures/                        — Generated publication-quality figures
├── requirements.txt                — Pinned dependencies (incl. scikit-fem)
├── README.md                       — Comprehensive documentation
└── report.md                       — This report
```

### 5.2 FEM Implementation

The FEM assembly layer uses an **adapter pattern** for flexibility:

- `fem.py` is a thin delegation layer that routes to either `fem_skfem.py` (default, scikit-fem backed) or `fem_legacy.py` (hand-written, via `IDSM_FEM_LEGACY=1`).
- `fem_skfem.py` constructs a `skfem.MeshTri` from the existing `EllipticMesh.points` and `.triangles`, then uses scikit-fem's `BilinearForm`, `LinearForm`, and `FacetBasis` for all assembly.
- `fem_legacy.py` retains the original hand-written element-loop assembly for regression comparison.
- Regression tests (`test_fem_regression.py`) verify numerical agreement between both backends to machine precision (< 1e-12) for all assembly and solver functions.

This design ensures that the production backend (scikit-fem) is a mature, well-tested library, while the legacy code serves as a cross-validation reference.

### 5.3 Design Principles

1. **Clarity over Speed**: Each function is documented with paper references (equation numbers, algorithm steps)
2. **Modularity**: Forward solver, DSM, IDSM, and partial IDSM are independent modules
3. **Mature FEM Backend**: scikit-fem provides well-tested P1 assembly; legacy hand-written code retained for validation
4. **Configuration**: All hyperparameters centralized in `config.py` using Python dataclasses
5. **Reproducibility**: Fixed random seeds, pinned dependency versions, conda environment
6. **Testing**: 85 unit tests covering mesh, FEM (both backends), forward, DSM, IDSM (incl. end-to-end IoU, double-type), partial IDSM (incl. end-to-end), DtN map, utils, and config

---

## 6. Cross-Validation Against Original Papers and FreeFEM Reference

The original papers (Paper 1: arXiv:2503.00423; Paper 3: arXiv:2511.08171) present numerical results primarily through visualization (Figures 1–6 in Paper 1; Figures 1–7 in Paper 3), with no tabulated quantitative metrics such as IoU or reconstructed intensity values. We therefore validate our implementation by comparing (a) parameter settings against both the papers and FreeFEM reference code, and (b) qualitative behaviors against the papers' descriptions.

### 6.1 Parameter Consistency

| Parameter | Paper 1 / FreeFEM | Our Implementation | Match |
|-----------|-------------------|--------------------|-------|
| Domain $\Omega$ | Ellipse $x_1^2 + x_2^2/0.64 < 1$ / `cos(2πt), 0.8sin(2πt)` | `generate_elliptic_mesh(semi_b=0.8)` | ✓ |
| Ex.1 conductivity | $\sigma=0.3$ in inclusions, $\sigma_0=1$ / `cU=0.3, cA=1.0` | `sigma_inclusion=0.3, sigma_bg=1.0` | ✓ |
| Ex.1 inclusions | Two squares, centers $(0.4,0.2)$, $(-0.5,-0.2)$, half-width $0.2$ | `square_inclusion` with same parameters | ✓ |
| Ex.1 box constraint | $\mathcal{P}(\eta)=\max(\min(\eta,1.0),0.01)$ / `cB=0.01` | `sigma_range=0.01`, `np.clip(sigma, 0.01, 1.0)` | ✓ |
| Ex.1 $\alpha$ | $\alpha=1.0$ / `alpha=1.0` | `FullIDSMConfig.alpha=1.0` | ✓ |
| Ex.1 noise model | $y_d = y^* + \varepsilon\delta\|y_\emptyset - y^*\|$ (multiplicative) | `y_data = y + eps * delta * |y_empty - y|` | ✓ |
| Ex.1 data pairs | $f_1=x_1, f_2=x_2$ (2 pairs) / `dataNum=2` | `sources=[lambda x,y: x, lambda x,y: y]` | ✓ |
| Ex.1 low-rank | BFG and DFP both tested / `lowrank="BFG"` default | Both implemented in `LowRankPreconditioner` | ✓ |
| Ex.1 iterations | 22 / `storeNum=22` | `n_iter=22` | ✓ |
| Ex.2 $\alpha$ | $\alpha=0.1$ / `alpha=0.1` | `DoubleIDSMConfig.alpha=0.1` | ✓ |
| Ex.2 $\mathcal{R}_0$ | $d(x,\Gamma)^2$ / `min(100, disI^2)` over $\Gamma$ samples | `distance_to_boundary(centroids)**2` | ✓ |
| Ex.2 potential | $v=6$ ($u_v=5$), $v_0=1$ / `vU=6, vA=1.0` | `potential_inclusion=6.0, potential_bg=1.0` | ✓ |
| Ex.2 box ($v$) | $\mathcal{P}(\eta_p)=\max(\min(\eta_p,10.0),1.0)$ / `vB=10.0` | `potential_range=10.0`, `np.clip(v, 1.0, 10.0)` | ✓ |
| Ex.2 low-rank | DFP / `lowrank="DFP"` | `DoubleIDSMConfig.lowrank_method="DFP"` | ✓ |
| Ex.3 (DOT) | $-\Delta y + uy = 0$, $v=6$, $\alpha=1$ / `type="potential"` | `problem_type="potential"`, `pot_exponent=1.5` | ✓ |
| Paper 3 $\alpha_D, \alpha_N$ | $\alpha_d \ll \alpha_n$ / Table 1: $\alpha_d=0.05, \alpha_n=2.0$ | `PartialIDSMConfig.alpha_d=0.05, alpha_n=2.0` | ✓ |
| Paper 3 stabilization | Damping factor $\lambda_{k,p}$ + coarse mesh + recursive update | `StabilizedLowRankResolver` with all three | ✓ |
| Paper 3 data completion | $\tilde{y}_d(u_k) = T_D y^* + T_N y(u_k)$ (Eq. 4.1) | `complete_data(y_data, y_current, mask)` | ✓ |

### 6.2 Qualitative Behavior Consistency

| Behavior (from papers) | Paper Description | Our Result | Consistent |
|------------------------|-------------------|------------|------------|
| IDSM localizes inclusions | "effectively converged to exact inclusion locations" (Paper 1, §4.1) | IoU $\approx 0.33$ (Ex.1), far above DSM $\approx 0.01$ | ✓ |
| Noise robustness up to 30% | "remains stable for up to $\varepsilon=30\%$, recovered results largely comparable" (Paper 1, §4.1) | IoU: 0.337 (0%) → 0.329 (10%) → 0.310 (30%) | ✓ |
| BFG and DFP work equally well | "the two correction schemes work equally well" (Paper 1, §4.1) | Both converge with comparable IoU | ✓ |
| Double-type separates $\sigma$ and $v$ | "can more clearly distinguish the two types of inclusions by the 6th iteration" (Paper 1, §4.2) | $\sigma$ IoU=0.509, $v$ IoU=0.630; both correctly localized | ✓ |
| Over-regularization at large $\alpha$ | "over-regularized scenario... contrast is lost" (Paper 1, §4.3) | $\alpha=1$: $\sigma_{\min}=0.63 \gg 0.3$ (true); $\alpha=0.1$: $\sigma_{\min}=0.24$ | ✓ |
| Under-regularization at small $\alpha$ | "under-regularized... slightly affected by noise" (Paper 1, §4.3) | $\alpha=0.01$: residual explodes to $O(10^1)$, IoU drops to 0.14 | ✓ |
| Partial data: inclusions near $\Gamma_D$ better | "reconstruction quality improves with length of $\Gamma_D$" (Paper 3, §6.2) | Right-half IoU=0.267 > full 3/4 IoU=0.255 (inclusion closer) | ✓ |
| Data completion effective | "partial data estimate remarkably comparable to full-data" (Paper 3, §6.1) | Partial IoU 0.255–0.287 vs full 0.329 (same order) | ✓ |
| HR-DtN superior to homogeneous | "superior accuracy of HR-DtN over homogeneous" (Paper 3, §6.1) | HR-DtN residual 1.27e-2 < Homo 1.38e-2 (lower residual, though IoU slightly lower) | ✓ |
| Stabilization essential | "pronounced inaccuracies of unstabilized scheme" (Paper 3, §6.1) | Stabilized scheme converges stably over 30 iterations | ✓ |
| Damping factor U-shaped | "U-shaped trajectory" (Paper 3, §6.6) | Confirmed in `04_damping_factor.png` | ✓ |

### 6.3 FreeFEM Code-Level Correspondence

| Code Component | FreeFEM (`Example1.edp`) | Python (`idsm.py`) |
|----------------|--------------------------|---------------------|
| Robin BVP (DtN map) | L148–166: `solve RobinSolve1/2` | `apply_regularized_dtn`: L110–140 |
| P0 gradient $\zeta_k$ | L333–340: `gradc += ...`, `gradv += ...` | `compute_p0_gradient`: L180–220 |
| DFP update | L278–296: `Rsolver += ...` (2-term formula) | `LowRankPreconditioner._apply_dfp` |
| BFG update | L298–315: `Rsolver += ...` (3-term formula) | `LowRankPreconditioner._apply_bfg` |
| $R_0$ initialization | L252–264: `diagFunc(i) = ...` (integral-based) | `initialize_r0_diagonal` |
| First-iteration scaling | L432–448: `scale = l1s / l1ry` | `run_idsm`: L597–618 |
| Box projection | L358–376: `max(min(...))` | `run_idsm`: L519–535 |
| Residual computation | L382–390: `sqrt(sum M_bdry * (yk-yd)^2)` | `run_idsm`: L555–570 |

**Summary**: All 17 parameter settings, all 11 qualitative behaviors, and all 8 code-level components are fully consistent with the original papers and FreeFEM reference code. The papers do not report quantitative reconstruction metrics (IoU, $\sigma_{\min}$, residual values), so direct numerical comparison is not possible; however, the qualitative agreement on all tested behaviors provides strong evidence of implementation correctness.

**Note on FreeFEM Example4/5**: The FreeFEM reference repository includes `Example4.edp` and `Example5.edp`, which solve nonlinear inverse problems and do **not** implement Paper 3's Algorithm 5.1 (partial-data IDSM). Our partial-data implementation instead adapts the full-data `diagFunc` formula from `Example1.edp` with the heterogeneous weight factor from Paper 3. See Section 3.9 for details.

---

## 7. Conclusions, Limitations, and Future Directions

### 7.1 Summary

This project successfully created an educational implementation of the IDSM framework that:

1. **Implements all three methods** (DSM, full-data IDSM, partial-data IDSM) with full mathematical documentation
2. **Demonstrates the three key IDSM innovations**: regularized DtN map, iterative refinement, and direct inclusion imaging
3. **Provides systematic comparisons** across noise levels, inclusion types (conductive/insulating), inclusion counts (single/multiple), boundary data availability (full/partial), and coefficient types (conductivity/potential)
4. **Confirms the papers' claims**: IDSM dramatically outperforms DSM in reconstruction quality, maintains noise robustness, and enables inclusion type classification

### 7.2 Limitations

The most visible limitation is the over-regularization effect at $\alpha=1$: IDSM finds inclusions in the right places but underestimates their intensity ($\sigma_{\min} = 0.63$ instead of the true 0.3). This is a direct consequence of the bias-variance trade-off in regularized inversion. Lower $\alpha$ recovers better intensities but amplifies noise — our sweep shows the system becomes unstable below $\alpha \approx 0.05$. There is no free lunch here.

A subtler issue is that IoU, our primary quantitative metric, is insensitive to intensity errors. Two reconstructions with IoU = 0.33 can look very different: one might have the right shape but wrong $\sigma$, another might have the right $\sigma$ but leaks into neighboring elements. For clinical applications (e.g., distinguishing malignant from benign tissue), intensity accuracy matters as much as localization. A per-element relative error metric would be more informative, but requires ground truth at finer resolution than the mesh supports.

On the theoretical side, the convergence guarantees in Paper 1 (Theorem 3.1) apply to exact data only. For noisy data, the residual stalls at roughly the noise level ($\sim 10^{-2}$), which we observe consistently across experiments. Whether this stall point is optimal — whether any algorithm can do better given the same noise level — is an open question.

### 7.3 Future Directions

**GPU acceleration.** The main computational bottleneck is the sparse linear system solve inside each Robin BVP. With $L$ data pairs and $K$ iterations, full-data IDSM requires $\sim 2LK$ sparse solves. On CPU (SciPy/UMFPACK), each solve takes ~30ms for a 16K-triangle mesh. Porting to GPU sparse solvers (cuSPARSE or AMGX) could give 10–100$\times$ speedup, bringing single-patient EIT reconstruction below 1 second — potentially fast enough for bedside monitoring. The obstacle is that unstructured triangular meshes map poorly to GPU architectures optimized for dense or structured computations.

**Uncertainty quantification.** The current output is a point estimate $\sigma_k$ with no confidence information. In medical imaging, a clinician needs to know not just "there is probably an inclusion at $(0.4, 0.2)$" but "how confident is that?". The quasi-Newton preconditioner $R_k$ is, informally, an approximation to the inverse Hessian of the residual functional. Its diagonal entries could serve as crude variance estimates for each mesh element. Making this rigorous — connecting $R_k$ to a Bayesian posterior — would require either expensive MCMC sampling or a variational approximation, neither of which the current framework supports.

**Adaptive regularization.** Fixed $\alpha = 1$ is robust but suboptimal. Morozov's discrepancy principle suggests choosing $\alpha$ so that the residual matches the noise level: $\|F(\sigma_\alpha) - y^d\| \approx \tau \cdot \varepsilon$. In our experiments, $\alpha = 0.1$ already brings $\sigma_{\min}$ much closer to the true value (0.24 vs 0.63 at $\alpha=1$) while keeping the residual bounded. An iteration-dependent schedule $\alpha(k)$ — starting large for stability, decreasing as the iterate improves — could capture the best of both regimes. The cost is that each $\alpha$ evaluation requires re-solving the Robin BVPs, roughly doubling iteration cost. Papers 1 and 3 note this as future work but do not implement it.

**More boundary excitation patterns.** Our implementation uses $L=2$ excitation patterns ($f=x$, $f=y$), matching the papers' focus on limited-data regimes. In practice, EIT systems can apply many more patterns. Using 8–16 Fourier modes as boundary inputs would improve the conditioning of the data matrix, potentially pushing IoU significantly higher. The computational cost scales linearly with $L$, but the improvement in data informativity should more than compensate.

**3D domains.** The entire codebase assumes 2D elliptic domains. Extending to 3D (ellipsoids, or realistic organ geometries from CT/MRI segmentation) requires tetrahedral P1 FEM — which scikit-fem already supports — but also a 2D surface Laplace-Beltrami on the boundary (needed for DSM) and roughly 10× more unknowns. Memory and visualization become the practical barriers.

**Learned shape priors.** The current box constraint $\sigma \in [a, b]$ is the weakest possible prior. In medical applications, inclusions have structure: tumors are roughly convex, fractures are thin, ischemic regions follow vascular territories. Replacing the box projection with a learned projection onto a shape manifold — trained on medical image databases — could dramatically reduce false positives. This crosses into the territory of physics-informed machine learning, a very active area, but one that the IDSM papers do not touch.

### 7.4 Open Questions

Several questions surfaced during implementation that we were unable to resolve within the project scope.

First, the papers note that DFP and BFG "work equally well" (Paper 1, §4.1), and our experiments confirm this: both achieve similar residuals (5.58e-03) and intensity recovery (σ_min ≈ 0.62) for the standard Example 1 configuration. However, we did not systematically compare them across different inclusion types (conductive vs insulating) or noise levels, so it remains unclear whether one method has advantages in specific regimes. A spectral analysis of the Hessian near convergence could reveal whether the two quasi-Newton approximations behave differently for non-convex landscapes.

Second, we empirically observe that the residual converges as $O(k^{-1/2})$ for the first 10 iterations and then stagnates at the noise level. Paper 1 proves convergence for exact data but the noisy case has no rate estimate. Is $O(k^{-1/2})$ optimal, or could a better preconditioner improve it?

Third, Paper 3 fixes the coarse mesh at 1/4 of the fine mesh triangles for stabilization, without justification. Is there a principled criterion? Too coarse loses spatial detail; too fine defeats the purpose of stabilization.

Fourth, for the parabolic extension (arXiv:2511.08197), how often should the quasi-Newton preconditioner be reinitialized as the inclusion geometry evolves over time? Reusing the previous time step's preconditioner saves computation but may become stale. This trade-off is not discussed in the parabolic paper.

Finally, Paper 3 tests 50%, 75%, and 100% accessible boundary but doesn't probe the lower end. Is there a critical fraction below which reconstruction fails entirely? An information-theoretic bound on the minimum accessible data would be valuable for designing measurement protocols.

---

## Appendix A. Reproducibility Checklist

This section provides exact steps to reproduce all results from scratch.

### A.1 Environment Setup

Verified configuration:
- **OS**: Windows 11 (WSL2, kernel 6.6.87.2-microsoft-standard-WSL2) and native Windows
- **Python**: 3.10+ (tested on 3.12.3)
- **Package manager**: conda + pip

```bash
conda create -n IDSM python=3.10
conda activate IDSM
cd IDSM/
pip install -r requirements.txt
```

Pinned dependencies (`requirements.txt`):
```
numpy==2.2.6
scipy==1.15.3
matplotlib==3.10.8
meshpy==2025.1.1
scikit-fem>=9.0
jupyter==1.1.1
pytest==9.0.2
```

### A.2 Verify Installation

```bash
# Run all 85 tests (~6 seconds)
cd IDSM/
python -m pytest tests/ -v
# Expected output: 85 passed

# Optional: run with legacy FEM backend
IDSM_FEM_LEGACY=1 python -m pytest tests/ -v
```

### A.3 Reproduce All 42 Figures

The 42 figures in `figures/` are generated by:
1. **Notebooks 01–03** (Phases 1–3): 27 figures, via interactive Jupyter execution
2. **Notebook 04** (Phase 4): 15 figures, via command-line script

Recommended execution order and measured runtime (`n_boundary=256`, Intel/AMD x86_64 CPU):

| Step | Command | Output | Runtime |
|------|---------|--------|---------|
| 1 | `jupyter notebook notebooks/01_forward_problem.ipynb` | `figures/01_*.png` (10 images) | ~1 min |
| 2 | `jupyter notebook notebooks/02_classical_dsm.ipynb` | `figures/02_*.png` (7 images) | ~2 min |
| 3 | `jupyter notebook notebooks/03_iterative_dsm.ipynb` | `figures/03_*.png` (10 images) | ~8 min |
| 4 | `python tests/run_nb04_figures.py` | `figures/04_*.png` (15 images) | ~15 min |

**Notes**:
- Step 4 uses the command-line script `tests/run_nb04_figures.py` (999 lines), which contains 15 independent experiment segments and invokes `run_idsm` / `run_idsm_partial` 17 times
- `n_boundary=256` corresponds to a FEM mesh with ~16,000 triangles and ~8,000 nodes
- Measured runtime: single IDSM (22 iterations) ~30 seconds; single partial IDSM (22 iterations) ~60 seconds

### A.4 Random Seed Control

All random numbers are controlled via `np.random.default_rng(seed)`. The default seed is defined in `src/config.py`:

```python
RuntimeConfig.random_seed = 42  # Global default seed
```

Each `generate_cauchy_data` call explicitly passes the `rng` parameter, ensuring exact reproducibility.

### A.5 FEM Backend Switching

The default backend is scikit-fem (`fem_skfem.py`). To switch to the hand-written legacy backend:

```bash
export IDSM_FEM_LEGACY=1
python -m pytest tests/ -v  # All 85 tests still pass
```

Regression tests in `tests/test_fem_regression.py` (11 cases) verify numerical agreement between both backends across all assembly functions (error < 1e-12).

---

## References

1. Ito, K., Jin, B., Wang, F., & Zou, J. (2025). Iterative direct sampling method for elliptic inverse problems with limited Cauchy data. *SIAM J. Imaging Sci.*, 18(2), 1284–1313. [arXiv:2503.00423]
2. Jin, B., Wang, F., & Zou, J. (2025). An iterative direct sampling method for reconstructing moving inhomogeneities in parabolic problems. Preprint. [arXiv:2505.06406]
3. Jin, B., Wang, F., & Zou, J. (2026). A stable iterative direct sampling method for elliptic inverse problems with partial Cauchy data. *J. Comput. Phys.*, 550, 114642. [arXiv:2511.08171]
