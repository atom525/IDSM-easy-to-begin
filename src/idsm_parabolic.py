"""
idsm_parabolic.py
=================

Iterative Direct Sampling Method (IDSM) for the parabolic inverse problem of
recovering moving inhomogeneities.

Implements Algorithm 4.1 of:
  Bangti Jin, Fengru Wang, Jun Zou. "An Iterative Direct Sampling Method
  for Reconstructing Moving Inhomogeneities in Parabolic Problems",
  arXiv:2511.08197 (2025).

This module is the parabolic analogue of `src/idsm.py` (elliptic IDSM,
Paper 1) and `src/idsm_partial.py` (Paper 3, partial Cauchy data).

Forward problem (Eq. 2.1-2.3):
    ∂_t y - Δy + N(y)u = f       in Ω × (0, T]
    ∂_n y = g                    on Γ × (0, T]
    y(·, 0) = h                  in Ω

The unknown u(x,t) = Σ_j p_j(t) χ_{ω_j(t)}(x) represents moving inclusions.
Three model variants (Eq. 2.4-2.6):
  - 'conductivity': N(y)u = -∇·(u ∇y)
  - 'potential'   : N(y)u = u·y                    (linear potential, p=2)
  - 'double'      : both blocks active simultaneously

We use a Crank-Nicolson scheme matching the FreeFEM reference repository
https://github.com/RaulWangfr/IDSM-parabolic (cloned to reference/parabolic_*.edp)
and implement the segment-wise iteration with DFP/BFG low-rank updates and
the inter-segment forgetScale damping.

Cross-references:
  - Algorithm 4.1: paper §4 / parabolic_PotentialFading.edp L353-654
  - Forward CN step: parabolic_*.edp L370-378 ("EmptyEquation") and L438-448
  - Backward adjoint Eq. 4.1: parabolic_*.edp L392-398 ("DualEquation")
  - DFP/BFG Eq. 4.6-4.7: parabolic_*.edp L295-336 ("Rsolver")
  - diagonal R0 (paper §4.5): parabolic_*.edp L262-281 ("diagFunc")
  - inter-segment damping: parabolic_*.edp L489-495
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import factorized, spsolve

from .fem_skfem import (
    assemble_stiffness_matrix,
    assemble_mass_matrix,
    assemble_boundary_load,
    assemble_boundary_mass_matrix,
)
from .idsm import LowRankPreconditioner


# ============================================================
# Configuration
# ============================================================

@dataclass
class ParabolicConfig:
    """Parameters for the parabolic IDSM solver.

    Defaults match parabolic_ConductivityMerging.edp L7-31.
    """
    total_time: float = 10.21
    delta_t: float = 0.1                 # outer segment length δt
    delta_t_split: int = 6               # in-segment substeps (Δt = δt / split)
    forget_scale: float = 0.7            # inter-segment damping λ (paper §4.5 default 0.6; .edp L20 default 0.7 — both reported, we follow .edp)
    tolerance: float = 0.08              # ε_tol for inner residual stop
    save_num: int = 10                   # storage capacity for low-rank corrections
    max_inner: int = 8                   # cap on inner iterations per segment (paper §4.3 max=5; .edp L19 default 8 — we follow .edp)
    cA: float = 1.0                      # background conductivity
    cB: float = 0.1                      # inclusion conductivity (lower → contrast)
    vA: float = 1e-10                    # background potential
    vB: float = 2e-10                    # inclusion potential (placeholder for cond.)
    kappa: float = 1e10                  # Dirichlet penalty for transition step
    lowrank: str = 'BFG'                 # 'BFG' or 'DFP'
    model: str = 'conductivity'          # 'conductivity', 'potential', or 'double'
    diag_exponent: float = 0.7           # exponent on minDis (= d²) ⇒ effective Euclidean d^1.4 (paper §4.5; .edp L274)
    diag_cutoff: float = 0.01            # zero-out threshold near boundary

    @property
    def inverse_dt(self) -> float:
        return self.delta_t / float(self.delta_t_split)

    @property
    def n_segments(self) -> int:
        return int(np.floor(self.total_time / self.delta_t))


# ============================================================
# Forward parabolic solver (Crank-Nicolson)
# ============================================================

def _assemble_cn_operator(M, K, M_v, dt: float):
    """Assemble Crank-Nicolson operator A = M/dt + 0.5*K + 0.5*M_v.

    Mirrors parabolic_*.edp L338-341:
      varf Stiffness(u, v) = int2d(0.5*Grad(u).Grad(v)*cA)
                           + int2d(0.5*u*v*vA)
                           + int2d(u*v/inverseDeltat);
      Amatrix = Stiffness(P1solve, P1solve, solver=Cholesky, factorize=true);

    The .edp version uses cA, vA (constant), so A is fixed throughout the run.
    For variable coefficients (e.g. inside the inner-iteration forward solve),
    K and M_v are reassembled per call.
    """
    return (M / dt) + 0.5 * K + 0.5 * M_v


def solve_forward_parabolic_segment(
    mesh,
    M, K_bg, M_v_bg,
    sigma_p0, v_p0,
    y_init: np.ndarray,
    t_begin: float,
    t_end: float,
    n_substeps: int,
    f_func: Callable[[float], Callable[[np.ndarray, np.ndarray], np.ndarray]],
    g_func: Callable[[float], Callable[[np.ndarray, np.ndarray], np.ndarray]],
):
    """Solve the forward parabolic PDE on (t_begin, t_end] by Crank-Nicolson.

    Weak CN formulation (parabolic_*.edp L438-448):
      0.5 σ ∇y_{n+1} ∇v + 0.5 σ ∇y_n ∇v
    + 0.5 V y_{n+1} v   + 0.5 V y_n v
    + (y_{n+1} - y_n) v / Δt
    = ∫_Ω f v dx + ∫_Γ g v ds          (evaluated at t_n + Δt/2, midpoint rule)

    Linear system per substep:
      A · y_{n+1} = (M/Δt - 0.5 K - 0.5 M_v) y_n + (M_f + b_g)
    where A = M/Δt + 0.5 K + 0.5 M_v.

    Parameters
    ----------
    mesh : EllipticMesh (or unit disk via generate_disk_mesh)
    M  : sparse (N,N) — mass matrix on Ω (no coefficient).
    K_bg, M_v_bg : sparse (N,N) — stiffness with sigma_p0 and mass with v_p0.
        (Caller assembles; inside the inner iteration these depend on guesses.)
    sigma_p0 : array (M,) — current conductivity field (P0).
        Used only for the source/load contribution shape; the matrices already
        encode it via K_bg/M_v_bg.
    v_p0 : array (M,) — current potential field (P0).
    y_init : array (N,) — initial condition y(·, t_begin).
    t_begin, t_end : float — segment endpoints.
    n_substeps : int — number of CN steps inside the segment (= delta_t_split).
    f_func : callable t → (callable (x,y) → array)
        Time-dependent interior source.
    g_func : callable t → (callable (x,y) → array)
        Time-dependent Neumann data (∂_n y = g).

    Returns
    -------
    y_history : array (n_substeps + 1, N)
        y_history[0] = y_init; y_history[-1] = y(·, t_end).
    """
    dt = (t_end - t_begin) / n_substeps
    A = _assemble_cn_operator(M, K_bg, M_v_bg, dt)
    rhs_op = (M / dt) - 0.5 * K_bg - 0.5 * M_v_bg

    A_solve = factorized(A.tocsc())

    n_dof = mesh.n_points
    y_history = np.zeros((n_substeps + 1, n_dof))
    y_history[0] = y_init.copy()

    for j in range(n_substeps):
        t_mid = t_begin + (j + 0.5) * dt
        f_at = f_func(t_mid)
        g_at = g_func(t_mid)

        # interior load: ∫_Ω f φ dx (use P1 mass-matrix-times-nodal-evaluation
        # for the simplest representation — equivalent to lumped quadrature).
        f_node = f_at(mesh.points[:, 0], mesh.points[:, 1])
        b_interior = M @ f_node

        b_boundary = assemble_boundary_load(mesh, g_at)

        rhs = rhs_op @ y_history[j] + b_interior + b_boundary
        y_history[j + 1] = A_solve(rhs)

    return y_history


def solve_forward_parabolic_nonlinear_segment(
    mesh,
    M, K_bg,
    sigma_p0, u_p0,
    y_init: np.ndarray,
    t_begin: float,
    t_end: float,
    n_substeps: int,
    f_func: Callable[[float], Callable[[np.ndarray, np.ndarray], np.ndarray]],
    g_func: Callable[[float], Callable[[np.ndarray, np.ndarray], np.ndarray]],
    vA: float = 1e-10,
    newton_tol: float = 1e-8,
    newton_max_iter: int = 30,
):
    """Forward parabolic PDE with nonlinear potential N(y)u = u·y·|y| (Eq. 2.5, p=3).

    Reference: parabolic_Nonlinear.edp L142-168 (Newton iteration).
    Linearisation: (y_k + δy)·|y_k + δy| ≈ y_k·|y_k| + 2|y_k|·δy.

    Per CN substep we solve for δy:
        (M/Δt + ½K_bg + ½(M_v + 2 M_{u|y_k|})) δy
        =  M/Δt (y_n - y_k) - ½K_bg (y_k + y_n) - ½M_v (y_k + y_n)
           - ½M_u (y_k|y_k| + y_n|y_n|) + b_f + b_g
    iterate y_k ← y_k + δy until ‖δy‖ < newton_tol.

    Parameters
    ----------
    mesh, M, K_bg : same as `solve_forward_parabolic_segment`.
    sigma_p0 : array (M,) — conductivity (constant cA in nonlinear example).
    u_p0     : array (M,) — nonlinear-potential strength field.
    vA       : float — background linear potential (for completeness).
    newton_tol, newton_max_iter — inner Newton stopping criteria.
    """
    dt = (t_end - t_begin) / n_substeps
    rhs_static = (M / dt) - 0.5 * K_bg - 0.5 * assemble_mass_matrix(
        mesh, np.full(mesh.n_triangles, vA))

    n_dof = mesh.n_points
    y_history = np.zeros((n_substeps + 1, n_dof))
    y_history[0] = y_init.copy()
    y_n = y_init.copy()

    for j in range(n_substeps):
        t_mid = t_begin + (j + 0.5) * dt
        f_at = f_func(t_mid)
        g_at = g_func(t_mid)
        f_node = f_at(mesh.points[:, 0], mesh.points[:, 1])
        b_interior = M @ f_node
        b_boundary = assemble_boundary_load(mesh, g_at)
        b_const = b_interior + b_boundary

        # Newton iterates around y_k (start from y_n)
        y_k = y_n.copy()
        for _ in range(newton_max_iter):
            # Project nodal |y_k| → P0 via centroidal average
            abs_yk_p0 = _node_to_p0_abs(mesh, y_k)
            M_lin = assemble_mass_matrix(mesh, 2.0 * u_p0 * abs_yk_p0)
            A_k = (M / dt) + 0.5 * K_bg + \
                  0.5 * assemble_mass_matrix(mesh, np.full(mesh.n_triangles, vA)) + \
                  0.5 * M_lin

            # nonlinear residual term: u·y·|y| at y_k and y_n
            ynyn = y_n * np.abs(y_n)
            ykyk = y_k * np.abs(y_k)
            M_u = assemble_mass_matrix(mesh, u_p0)
            nonlinear_rhs = -0.5 * M_u @ (ykyk + ynyn)

            # full RHS
            rhs = (rhs_static @ y_n) - (0.5 * K_bg @ y_k) - \
                  (0.5 * assemble_mass_matrix(
                      mesh, np.full(mesh.n_triangles, vA)) @ y_k) + \
                  nonlinear_rhs + b_const - (M / dt) @ y_k + (M / dt) @ y_n

            # Above expression telescopes into  (M/Δt + ½K + ½M_v + ½M_lin) δy = ...
            # but to keep the formulation transparent we solve directly:
            delta_y = spsolve(A_k.tocsc(),
                              (M / dt) @ (y_n - y_k)
                              - 0.5 * K_bg @ (y_k + y_n)
                              - 0.5 * assemble_mass_matrix(
                                  mesh, np.full(mesh.n_triangles, vA)
                              ) @ (y_k + y_n)
                              - 0.5 * M_u @ (ykyk + ynyn)
                              + b_const)
            y_k = y_k + delta_y
            if np.linalg.norm(delta_y) < newton_tol:
                break

        y_history[j + 1] = y_k
        y_n = y_k

    return y_history


def _node_to_p0_abs(mesh, y_node: np.ndarray) -> np.ndarray:
    """Per-triangle |y| via P1→P0 centroidal averaging."""
    return np.abs(y_node[mesh.triangles].mean(axis=1))


def solve_backward_adjoint_segment(
    mesh,
    M, K_bg, M_v_bg,
    measurement_residuals: np.ndarray,
    n_substeps: int,
    dt: float,
):
    """Solve the backward adjoint Eq. 4.1 on a single segment.

    Eq. 4.1 (paper §4.1):
      -∂_t z - Δz + V z = 0    in Ω × (t_n, t_{n+1})
      ∂_n z = y_s^{n,d}        on Γ × (t_n, t_{n+1})
      z(·, t_{n+1}) = 0        terminal condition

    Reverse-time Crank-Nicolson (parabolic_*.edp L383-399):
    Substituting τ = t_{n+1} - t turns this into a forward CN with the same
    operator A as the primal problem. Stepping from j = n_substeps down to 1:

      A · z^{j-1} = (M/Δt - 0.5 K - 0.5 M_v) z^j + b_meas(j)

    where b_meas(j) = ∫_Γ measurement_residuals[j-1] · v ds (Neumann load).

    The terminal condition z^{n_substeps} = 0 makes z^{n_substeps - 1} purely
    driven by the last measurement; subsequent steps cumulate.

    Parameters
    ----------
    measurement_residuals : array (n_substeps, N)
        Per-substep residual field on the boundary (the (yEmpty - measurement)/2
        sum at substep j is mapped to the j-th entry; matches L387-389).
    n_substeps : int — number of substeps (= delta_t_split).
    dt : float — substep size (= inverseDeltat).

    Returns
    -------
    z_history : array (n_substeps + 1, N)
        z_history[n_substeps] = 0 (terminal); z_history[0] = z(·, t_begin).
    normal_scale : float
        1 / Σ_j ‖measurement_residuals[j]‖²_{L²(Γ)} for use as weight.
    """
    A = _assemble_cn_operator(M, K_bg, M_v_bg, dt)
    rhs_op = (M / dt) - 0.5 * K_bg - 0.5 * M_v_bg
    A_solve = factorized(A.tocsc())

    M_bdry = assemble_boundary_mass_matrix(mesh)

    n_dof = mesh.n_points
    z_history = np.zeros((n_substeps + 1, n_dof))

    normal_scale_sum = 0.0
    # measurement_residuals[j] correspond to substep boundary midpoint
    for j in range(n_substeps, 0, -1):
        meas_j = measurement_residuals[j - 1]
        # contribution ∫_Γ meas² ds via the boundary mass matrix
        normal_scale_sum += float(meas_j @ (M_bdry @ meas_j))

        b_meas = M_bdry @ meas_j
        rhs = rhs_op @ z_history[j] + b_meas
        z_history[j - 1] = A_solve(rhs)

    if normal_scale_sum > 0.0:
        normal_scale = 1.0 / normal_scale_sum
    else:
        normal_scale = 0.0

    return z_history, normal_scale


# ============================================================
# Local dual function (ζ_c, ζ_v computation)
# ============================================================

def compute_local_dual(
    mesh,
    y_history: np.ndarray,
    z_history: np.ndarray,
    normal_scale: float,
):
    """Compute the local dual functions ζ_c (conductivity) and ζ_v (potential).

    Paper §4.2 / parabolic_*.edp L407-409:
      ζ_c = ∫_τ ½ (∇y_{j+1} + ∇y_j) · ∇z_j · normalScale  dt
      ζ_v = ∫_τ ½ (y_{j+1} + y_j) · z_j · normalScale     dt

    Implemented as midpoint-rule on each substep, with the midpoint of y at
    [j, j+1] paired with z at [j] (matching the .edp convention where yU runs
    forward 0..deltaTsplit and yDual runs backward in the same indexing).

    Note: in the .edp the loop body sums over the entire segment by accumulating
    P0 fields per substep. Here we sum them and return the time-integrated P0
    fields directly.

    Parameters
    ----------
    y_history : array (n_substeps + 1, N) — primal y over the segment.
    z_history : array (n_substeps + 1, N) — adjoint z over the segment.
    normal_scale : float — weight 1/‖measurement‖² (multiplied into ζ).

    Returns
    -------
    zeta_c : array (M,) — P0 conductivity dual (per triangle).
    zeta_v : array (M,) — P0 potential dual.
    """
    n_substeps = y_history.shape[0] - 1
    n_tri = mesh.n_triangles

    zeta_c = np.zeros(n_tri)
    zeta_v = np.zeros(n_tri)

    grad_phi = mesh.grad_phi  # (M, 3, 2)
    tri = mesh.triangles      # (M, 3)

    for j in range(n_substeps):
        # midpoint of y across the substep
        y_mid = 0.5 * (y_history[j + 1] + y_history[j])
        # adjoint sample at the substep boundary (matches edp pairing)
        z_j = z_history[j]

        # ∇y_mid and ∇z_j on each triangle (P0 from P1)
        grad_y = (grad_phi[:, 0] * y_mid[tri[:, 0], None]
                  + grad_phi[:, 1] * y_mid[tri[:, 1], None]
                  + grad_phi[:, 2] * y_mid[tri[:, 2], None])  # (M, 2)
        grad_z = (grad_phi[:, 0] * z_j[tri[:, 0], None]
                  + grad_phi[:, 1] * z_j[tri[:, 1], None]
                  + grad_phi[:, 2] * z_j[tri[:, 2], None])

        zeta_c += np.einsum('mi,mi->m', grad_y, grad_z)

        # P0 average of y_mid and z_j on each triangle
        y_avg = (y_mid[tri[:, 0]] + y_mid[tri[:, 1]] + y_mid[tri[:, 2]]) / 3.0
        z_avg = (z_j[tri[:, 0]] + z_j[tri[:, 1]] + z_j[tri[:, 2]]) / 3.0
        zeta_v += y_avg * z_avg

    zeta_c *= normal_scale
    zeta_v *= normal_scale
    return zeta_c, zeta_v


# ============================================================
# Initial diagonal preconditioner R₀ (parabolic version)
# ============================================================

def initialize_r0_parabolic(mesh, exponent: float = 0.7, cutoff: float = 0.01):
    """Initialize the diagonal R₀ preconditioner for the parabolic inverse problem.

    Reference: parabolic_*.edp L262-281 (diagFunc):
      diagFunc(i)        = minDis^0.7
      diagFunc(i + ndof) = minDis^0.7
      if minDis < 0.01: both blocks set to 0

    Important: in the .edp, ``minDis`` is the *squared* Euclidean distance to Γ
    (it is computed as ``min((cx-bx)^2 + (cy-by)^2)``, no sqrt). Therefore
    ``minDis^0.7 = d(x,Γ)^{1.4}``, which matches paper §4.5 explicitly:
    ``D(x) = d(x,Γ)^{1.4} · χ_{Ω_ε}(x)``. The argument name `exponent`
    refers to the squared-distance exponent (0.7), not the Euclidean one (1.4).

    The two blocks (conductivity, potential) share the same template; the
    iteration's projection step (`apply_inclusion_projection`) zeroes out the
    block not used by the chosen `model`.

    Parameters
    ----------
    mesh : EllipticMesh (unit disk)
    exponent : float — default 0.7 (acts on squared distance d²; corresponds
        to Euclidean d^{1.4} as in paper §4.5 / .edp L274).
    cutoff : float — squared-distance threshold for boundary masking
        (i.e. exclude triangles with d(x,Γ)² < 0.01 ⇔ d(x,Γ) < 0.1).

    Returns
    -------
    diag : array (2 * M,) — concatenated [conductivity_block, potential_block].
    """
    centroids = mesh.centroids  # (M, 2)
    n_tri = mesh.n_triangles

    # squared distance to the unit-circle boundary; .edp computes
    # min over 200 boundary samples, we use the exact distance for the
    # unit disk: minDis = (1 - r)², where r = ||c||.
    r = np.sqrt(np.sum(centroids ** 2, axis=1))
    min_dis = (1.0 - r) ** 2  # squared distance, matching edp's `(.. - x)^2 + (.. - y)^2`

    block = np.power(np.maximum(min_dis, 0.0), exponent)
    block[min_dis < cutoff] = 0.0

    return np.concatenate([block, block])


# ============================================================
# Inclusion projection (paper §3, FreeFEM L416-420)
# ============================================================

def apply_inclusion_projection(
    eta: np.ndarray, n_tri: int, model: str,
    cA: float, cB: float, vA: float, vB: float,
):
    """Apply the projection P (Algorithm 4.1 step 4).

    parabolic_*.edp L416-420:
      cGuess = max(min(cGrad, 0), -0.99/|cA-cB|) * |cA-cB| + cA
      vGuess = max(min(vGrad, 2),  0           ) * |vA-vB| + vA

    The bounds reflect known a-priori contrast: conductivity drops below
    background (so cGrad ∈ [-0.99/Δc, 0]); potential lies in [0, 2*|Δv|].

    Returns
    -------
    sigma : array (M,) — projected conductivity field σ ∈ [cB, cA] (or cA if
        model excludes conductivity).
    v_pot : array (M,) — projected potential field v ∈ [vA, vA + 2|Δv|].
    """
    c_grad = eta[:n_tri]
    v_grad = eta[n_tri:]

    delta_c = abs(cA - cB)
    delta_v = abs(vA - vB)

    if model in ('conductivity', 'double'):
        c_clip = np.clip(c_grad, -0.99 / delta_c if delta_c > 0 else -np.inf, 0.0)
        sigma = c_clip * delta_c + cA
    else:
        sigma = np.full(n_tri, cA)

    if model in ('potential', 'double'):
        v_clip = np.clip(v_grad, 0.0, 2.0)
        v_pot = v_clip * delta_v + vA
    else:
        v_pot = np.full(n_tri, vA)

    return sigma, v_pot


# ============================================================
# Inter-segment damping (forgetScale)
# ============================================================

def damp_lowrank_state(R: LowRankPreconditioner, forget_scale: float):
    """Apply forgetScale damping to all stored low-rank corrections.

    parabolic_*.edp L489-495:
      if(localLoop == 0){
        for(int j = 0; j < min(storeCount, saveNum); j++){
          for(int i = 0; i < P0solve.ndof*2; i++){
            sStore(i, j) *= forgetScale;
            ryStore(i, j) *= forgetScale;
          }
        }
      }

    Note: only sStore and ryStore are damped (not yStore). This is because the
    low-rank update formula is bilinear in (s, ry) and linear in y; damping
    s and ry by λ scales each correction by λ², matching the operator-norm
    intuition that "old" updates lose weight quadratically.
    """
    for j in range(min(R.count, R.max_store)):
        R.s_store[j] *= forget_scale
        R.ry_store[j] *= forget_scale


__all__ = [
    'ParabolicConfig',
    'solve_forward_parabolic_segment',
    'solve_backward_adjoint_segment',
    'compute_local_dual',
    'initialize_r0_parabolic',
    'apply_inclusion_projection',
    'damp_lowrank_state',
    'interpolate_boundary_data',
    'inclusion_trajectory_example1',
    'inclusion_trajectory_example4',
    'circle_indicator_p0',
    'iterate_within_segment',
    'run_idsm_parabolic',
]


# ============================================================
# Time-interpolation of boundary measurements
# ============================================================

def interpolate_boundary_data(
    y_data_fine: np.ndarray, t: float, forward_dt: float,
):
    """Linearly interpolate boundary measurement at time t.

    parabolic_*.edp L242-260 (BoundaryData):
      n1 = ceil(t/forwardDeltat); n2 = floor(t/forwardDeltat)
      lambda = (t/forwardDeltat - n2)
      result = yData[n1] * lambda + yData[n2] * (1 - lambda)

    Parameters
    ----------
    y_data_fine : array (T_f, N) — fine time-grid noisy boundary data.
    t : float — target time.
    forward_dt : float — fine-grid spacing.

    Returns
    -------
    array (N,) — interpolated boundary field.
    """
    n_fine = y_data_fine.shape[0]
    s = t / forward_dt
    n2 = int(np.floor(s))
    n1 = int(np.ceil(s))
    n2 = max(0, min(n_fine - 1, n2))
    n1 = max(0, min(n_fine - 1, n1))
    if n1 == n2:
        return y_data_fine[n1].copy()
    lam = s - n2
    return lam * y_data_fine[n1] + (1.0 - lam) * y_data_fine[n2]


# ============================================================
# Inclusion trajectories (ground truth for benchmarking)
# ============================================================

def inclusion_trajectory_example1(t: float):
    """Example 5.1 (ConductivityMerging): 4 inclusion trajectories.

    parabolic_ConductivityMerging.edp L47-101:
      - traj 0, 1: two inclusions on opposite-y branches that merge at t=3
        and split / reverse direction at t=6.
      - traj 2, 3: ghost (out-of-domain) inclusions, never visible.

    Returns
    -------
    centers : array (4, 2) — inclusion centers at time t.
    radii   : array (4, 2) — semi-axes (rx, ry).
    active  : array (4,) bool — whether each inclusion is inside the unit disk.
    """
    centers = np.zeros((4, 2))
    radii = np.array([
        [0.2, 0.2],
        [0.2, 0.2],
        [1e-10, 1e-10],
        [1e-10, 1e-10],
    ])

    # traj 0
    if t < 3.0:
        centers[0, 1] = 0.6 * np.cos(4 * t * np.pi / 24)
        centers[0, 0] = -0.7 * np.sin(4 * t * np.pi / 24)
    else:
        centers[0, 1] = -0.6 * np.cos(4 * t * np.pi / 24)
        centers[0, 0] = -0.7 * np.sin(4 * t * np.pi / 24)

    # traj 1
    if t < 3.0:
        centers[1, 1] = -0.6 * np.cos(4 * t * np.pi / 24)
        centers[1, 0] = -0.7 * np.sin(4 * t * np.pi / 24)
    else:
        if t < 6.0:
            centers[1, 1] = -0.6 * np.cos(4 * t * np.pi / 24)
            centers[1, 0] = -0.7 * np.sin(4 * t * np.pi / 24)
        else:
            centers[1, 1] = -0.6 * np.cos(4 * (12.0 - t) * np.pi / 24)
            centers[1, 0] = -0.7 * np.sin(4 * (12.0 - t) * np.pi / 24)

    centers[2:] = 2.0  # ghost
    active = np.array([True, True, False, False])
    return centers, radii, active


def inclusion_trajectory_example4(t: float):
    """Example 5.4 (PotentialFading): 2 active potential inclusions.

    parabolic_PotentialFading.edp L47-87:
      - traj 0, 1: ghost (no conductivity contrast).
      - traj 2: circular orbit, fading potential vB → vA over t=0..6.
      - traj 3: opposite circular orbit, growing potential vA → vB.

    Returns
    -------
    centers, radii, active (4,2)/(4,2)/(4,)
    plus
    pot_vals : array (4,) — current potential value at each inclusion.
    """
    vA, vB = 1e-10, 15.0
    centers = np.zeros((4, 2))
    radii = np.array([
        [1e-10, 1e-10],
        [1e-10, 1e-10],
        [0.2, 0.2],
        [0.2, 0.2],
    ])
    centers[2, 0] = 0.7 * np.cos(3 * t * np.pi / 24)
    centers[2, 1] = 0.6 * np.sin(3 * t * np.pi / 24)
    centers[3, 0] = 0.5 * np.cos(3 * t * np.pi / 24 + 4 * np.pi / 5)
    centers[3, 1] = 0.6 * np.sin(3 * t * np.pi / 24 + 4 * np.pi / 5)
    centers[:2] = 2.0  # ghost
    active = np.array([False, False, True, True])
    pot_vals = np.array([vA, vA,
                         max(vB + t * (vA - vB) / 6.0, vA),
                         min(vA + t * (vB - vA) / 6.0, vB)])
    return centers, radii, active, pot_vals


def inclusion_trajectory_example2(t: float):
    """Example 5.2 (MixedMoving): mixed conductivity + potential inclusions.

    parabolic_MixedMoving.edp L47-87:
      - traj 0, 1: conductivity inclusions on circular arcs (cB strength)
      - traj 2  : potential inclusion on a circular arc (vB strength)
      - traj 3  : ghost (off-domain at (2.0, 2.0))

    All circles have fixed radius 0.2.

    Returns
    -------
    centers : array (4, 2)
    radii   : array (4, 2) — all (0.2, 0.2) except ghosts
    active  : array (4,) bool — traj 0,1,2 active; traj 3 ghost
    kind    : array (4,) of {'c', 'v', '-'} — block tag (cond / pot / ghost)
    """
    centers = np.zeros((4, 2))
    radii = np.full((4, 2), 0.2)

    centers[0, 0] = 0.65 * np.cos(t * np.pi / 8 - 7 * np.pi / 6)
    centers[0, 1] = 0.65 * np.sin(t * np.pi / 8 - 7 * np.pi / 6)

    centers[1, 0] = 0.6 * np.cos(t * np.pi / 8 - 2 * np.pi / 6)
    centers[1, 1] = 0.7 * np.sin(t * np.pi / 8 - 2 * np.pi / 6)

    centers[2, 0] = 0.65 * np.cos(t * np.pi / 8 - 2 * np.pi / 6)
    centers[2, 1] = 0.65 * np.sin(t * np.pi / 8 - 2 * np.pi / 6)

    centers[3] = 2.0  # ghost
    radii[3] = 1e-10

    active = np.array([True, True, True, False])
    kind = np.array(['c', 'c', 'v', '-'])
    return centers, radii, active, kind


def inclusion_trajectory_example3(t: float):
    """Example 5.3 (Nonlinear): single nonlinear-potential inclusion.

    parabolic_Nonlinear.edp L49-63:
      - traj 0: orbit (0.5 cos(4πt/24+π/4), 0.7 sin(4πt/24+π/4))
      - inclusion strength uB = 20.0 (potential block, p=3 nonlinearity in N(y)u)

    Returns
    -------
    centers, radii, active : as for other examples (1 inclusion)
    pot_strength : float — uB
    """
    centers = np.zeros((1, 2))
    centers[0, 0] = 0.5 * np.cos(4 * t * np.pi / 24 + np.pi / 4)
    centers[0, 1] = 0.7 * np.sin(4 * t * np.pi / 24 + np.pi / 4)
    radii = np.array([[0.2, 0.2]])
    active = np.array([True])
    return centers, radii, active, 20.0


def inclusion_trajectory_example5(t: float):
    """Example 5.5 (ConductivityDiminishing): inclusion radius shrinks to zero.

    parabolic_ConductivityDiminishing.edp L48-88:
      - traj 0: circular orbit, fixed radius 0.2
      - traj 1: opposite orbit, radius shrinks as max(0.3 - 0.03·t, 1e-10)
      - traj 2, 3: ghost (off-domain)

    Returns
    -------
    centers, radii, active (4,2)/(4,2)/(4,)
    """
    centers = np.zeros((4, 2))
    centers[0, 0] = 0.7 * np.cos(4 * t * np.pi / 24 + np.pi / 3)
    centers[0, 1] = 0.6 * np.sin(4 * t * np.pi / 24 + np.pi / 3)
    centers[1, 0] = 0.6 * np.cos(4 * t * np.pi / 24 - 2 * np.pi / 3)
    centers[1, 1] = 0.5 * np.sin(4 * t * np.pi / 24 - 2 * np.pi / 3)
    centers[2:] = 2.0

    r1 = max(0.3 - 0.03 * t, 1e-10)
    radii = np.array([
        [0.2, 0.2],
        [r1, r1],
        [1e-10, 1e-10],
        [1e-10, 1e-10],
    ])

    # an inclusion is "active" (visible) only when its radius exceeds a threshold
    active = np.array([True, r1 > 1e-3, False, False])
    return centers, radii, active


def circle_indicator_p0(mesh, centers: np.ndarray, radii: np.ndarray,
                         active: np.ndarray) -> np.ndarray:
    """Build a P0 indicator (per triangle) for the union of active circles.

    centroid c falls inside circle j iff
      (c.x - centers[j,0])²/radii[j,0]² + (c.y - centers[j,1])²/radii[j,1]² < 1
    """
    cents = mesh.centroids
    ind = np.zeros(mesh.n_triangles, dtype=bool)
    for j in range(len(active)):
        if not active[j]:
            continue
        dx = (cents[:, 0] - centers[j, 0]) / radii[j, 0]
        dy = (cents[:, 1] - centers[j, 1]) / radii[j, 1]
        ind |= (dx * dx + dy * dy) < 1.0
    return ind


# ============================================================
# Inner iteration (Algorithm 4.1 steps 5-22)
# ============================================================

def iterate_within_segment(
    mesh,
    M, K_const, M_v_const,
    M_bdry,
    R: LowRankPreconditioner,
    cfg: ParabolicConfig,
    y_last: np.ndarray,
    y_empty_history: np.ndarray,
    y_data_fine: np.ndarray,
    forward_dt: float,
    t_begin: float,
    t_end: float,
    f_func: Callable, g_func: Callable,
    sigma_state: dict,
):
    """Run the inner iteration for one IDSM segment (Algorithm 4.1 steps 5-22).

    Mirrors parabolic_*.edp L403-580 (the `while(true)` block).

    Parameters
    ----------
    M, K_const, M_v_const : sparse — constant-coefficient FE matrices, used in
        the cached CN operator A and the adjoint solve. (cA, vA in the .edp.)
    M_bdry : sparse — boundary mass matrix (∫_Γ φ_i φ_j ds).
    R : LowRankPreconditioner — current preconditioner state.
    sigma_state : dict — mutable container holding {'sigma': ..., 'v_pot': ...}
        carrying the latest (cGuess, vGuess) from the previous segment for
        warm-starting (matches `cGuess[tIndex - 1]` blending in .edp L432-436).

    Returns
    -------
    sigma : array (M,) — final conductivity guess for this segment.
    v_pot : array (M,) — final potential guess.
    y_history : array (n_substeps + 1, N) — final forward solution.
    n_inner : int — number of inner iterations performed.
    residuals : list — per-iteration boundary residual.
    """
    n_tri = mesh.n_triangles
    n_dof = mesh.n_points
    n_sub = cfg.delta_t_split
    inv_dt = cfg.inverse_dt

    # ---- backward adjoint with measurement vs background ----------------
    meas_residuals = np.zeros((n_sub, n_dof))
    for j in range(n_sub):
        t_mid = t_begin + (j + 0.5) * (cfg.delta_t / n_sub)
        meas_t = interpolate_boundary_data(y_data_fine, t_mid, forward_dt)
        # 0.5 (yEmpty[j+1] + yEmpty[j]) - measurement (parabolic_*.edp L388-389)
        meas_residuals[j] = 0.5 * (y_empty_history[j + 1] + y_empty_history[j]) - meas_t

    z_history, normal_scale = solve_backward_adjoint_segment(
        mesh, M, K_const, M_v_const, meas_residuals,
        n_substeps=n_sub, dt=inv_dt,
    )

    sigma = sigma_state.get('sigma', np.full(n_tri, cfg.cA)).copy()
    v_pot = sigma_state.get('v_pot', np.full(n_tri, cfg.vA)).copy()

    residuals: List[float] = []
    y_history = y_empty_history.copy()
    sigma_prev = sigma_state.get('sigma_prev', None)
    v_pot_prev = sigma_state.get('v_pot_prev', None)

    n_inner = 0
    tol = cfg.tolerance

    while n_inner < cfg.max_inner:
        # --- Step 5: ζ from current y_history vs adjoint, then project --------
        zeta_c, zeta_v = compute_local_dual(mesh, y_history, z_history,
                                             normal_scale)
        eta = R.apply(np.concatenate([zeta_c, zeta_v]))
        sigma, v_pot = apply_inclusion_projection(
            eta, n_tri, cfg.model, cfg.cA, cfg.cB, cfg.vA, cfg.vB,
        )

        # --- Step 6: forward with the new (σ, V), Eq. 4.5 ---------------------
        # blend with previous segment's guess for time-smoothness, .edp L432-436
        K_guess = assemble_stiffness_matrix(mesh, sigma)
        M_v_guess = assemble_mass_matrix(mesh, v_pot)
        y_history = solve_forward_parabolic_segment(
            mesh, M, K_guess, M_v_guess, sigma, v_pot,
            y_last, t_begin=t_begin, t_end=t_end, n_substeps=n_sub,
            f_func=f_func, g_func=g_func,
        )

        # --- Step 7: backward adjoint w.r.t. residual yEmpty - yU ------------
        meas_residuals_tilde = np.zeros((n_sub, n_dof))
        for j in range(n_sub):
            meas_residuals_tilde[j] = (
                0.5 * (y_empty_history[j + 1] + y_empty_history[j])
                - 0.5 * (y_history[j + 1] + y_history[j])
            )
        z_tilde_history, _ = solve_backward_adjoint_segment(
            mesh, M, K_const, M_v_const, meas_residuals_tilde,
            n_substeps=n_sub, dt=inv_dt,
        )

        zeta_c_tilde, zeta_v_tilde = compute_local_dual(
            mesh, y_history, z_tilde_history, normal_scale,
        )

        # --- Step 8: error fields w.r.t. background --------------------------
        delta_c = abs(cfg.cA - cfg.cB)
        delta_v = abs(cfg.vA - cfg.vB)
        c_err = np.zeros(n_tri)
        v_err = np.zeros(n_tri)
        if cfg.model in ('conductivity', 'double'):
            c_err = (sigma - cfg.cA) / delta_c
        if cfg.model in ('potential', 'double'):
            v_err = (v_pot - cfg.vA) / delta_v

        # --- Step 9: inter-segment damping (only on first inner iter) --------
        if n_inner == 0:
            damp_lowrank_state(R, cfg.forget_scale)

        # --- Step 10: scale R₀ diagonal blocks -------------------------------
        if n_inner == 0 or R.count == 0:
            dyk_c = zeta_c_tilde * R.diag[:n_tri]
            dyk_v = zeta_v_tilde * R.diag[n_tri:]
            scale1 = 0.0
            scale2 = 0.0
            if cfg.model in ('conductivity', 'double'):
                num = float(np.sum(np.abs(c_err) * mesh.areas))
                den = float(np.sum(np.abs(dyk_c) * mesh.areas))
                if den > 0:
                    scale1 = num / den
            if cfg.model in ('potential', 'double'):
                num = float(np.sum(np.abs(v_err) * mesh.areas))
                den = float(np.sum(np.abs(dyk_v) * mesh.areas))
                if den > 0:
                    scale2 = num / den
            if scale1 != 0:
                R.diag[:n_tri] *= scale1
            if scale2 != 0:
                R.diag[n_tri:] *= scale2

        # --- Step 11: store low-rank update ----------------------------------
        yk = np.concatenate([zeta_c_tilde, zeta_v_tilde])
        ryk = R.apply(yk)
        # boundary saturation correction (parabolic_*.edp L534-551)
        for i in range(n_tri):
            if cfg.model in ('conductivity', 'double'):
                if c_err[i] == 0.0:
                    c_err[i] = max(ryk[i], 0.0)
                if delta_c > 0 and c_err[i] == -0.99 / delta_c:
                    c_err[i] = min(ryk[i], -0.99 / 0.90)
            if cfg.model in ('potential', 'double'):
                if v_err[i] == 2.0:
                    v_err[i] = max(ryk[n_tri + i], 2.0)
                if v_err[i] == 0.0:
                    v_err[i] = min(ryk[n_tri + i], 0.0)

        sk = np.concatenate([c_err, v_err])
        if float(sk @ yk) > 0.0:
            R.update(sk, yk, ryk)

        # --- Step 12: residual & termination ---------------------------------
        meas_end = interpolate_boundary_data(y_data_fine, t_end, forward_dt)
        diff = y_history[-1] - meas_end
        num = float(diff @ (M_bdry @ diff))
        den = float(meas_end @ (M_bdry @ meas_end))
        rel_err = np.sqrt(max(num, 0.0) / max(den, 1e-300))
        residuals.append(rel_err)

        n_inner += 1
        if rel_err < tol:
            break
        # parabolic_*.edp L578: tolerance loosens every saveNum iterations
        if n_inner % cfg.save_num == 0:
            R.count = 0  # reset store; matches `storeCount = 0`
            tol *= 1.2

    sigma_state['sigma_prev'] = sigma_state.get('sigma', np.full(n_tri, cfg.cA))
    sigma_state['v_pot_prev'] = sigma_state.get('v_pot', np.full(n_tri, cfg.vA))
    sigma_state['sigma'] = sigma
    sigma_state['v_pot'] = v_pot
    return sigma, v_pot, y_history, n_inner, residuals


# ============================================================
# Forward solver with Dirichlet penalty (segment transition)
# ============================================================

def solve_forward_with_dirichlet_penalty(
    mesh,
    M, M_bdry, sigma_p0, v_p0, y_init: np.ndarray,
    t_begin: float, t_end: float, n_substeps: int,
    f_func: Callable, g_func: Callable,
    measurement_times_data: List[Tuple[float, np.ndarray]],
    kappa: float = 1e10,
):
    """Forward CN with a Dirichlet penalty term on the boundary, matching the
    final transition step in parabolic_*.edp L595-642.

    Adds  + kappa ∫_Γ u v ds  -  kappa ∫_Γ d v ds  to the system, where d is
    the boundary measurement at the substep midpoint. This drives y → measurement
    on Γ at high penalty, providing the next-segment initial value.

    Parameters
    ----------
    measurement_times_data : list of (t_mid, dirichlet_data) per substep.
    """
    K = assemble_stiffness_matrix(mesh, sigma_p0)
    M_v = assemble_mass_matrix(mesh, v_p0)
    dt = (t_end - t_begin) / n_substeps
    A_base = _assemble_cn_operator(M, K, M_v, dt)
    A_pen = A_base + kappa * M_bdry
    rhs_op = (M / dt) - 0.5 * K - 0.5 * M_v
    A_solve = factorized(A_pen.tocsc())

    n_dof = mesh.n_points
    y_history = np.zeros((n_substeps + 1, n_dof))
    y_history[0] = y_init.copy()

    for j in range(n_substeps):
        t_mid, diri_data = measurement_times_data[j]
        f_at = f_func(t_mid)
        g_at = g_func(t_mid)

        f_node = f_at(mesh.points[:, 0], mesh.points[:, 1])
        b_interior = M @ f_node
        b_boundary = assemble_boundary_load(mesh, g_at)
        b_diri = kappa * (M_bdry @ diri_data)

        rhs = rhs_op @ y_history[j] + b_interior + b_boundary + b_diri
        y_history[j + 1] = A_solve(rhs)

    return y_history


# ============================================================
# Top-level driver
# ============================================================

def run_idsm_parabolic(
    mesh,
    y_data_fine: np.ndarray,
    forward_dt: float,
    f_func: Callable, g_func: Callable,
    initial_data: np.ndarray,
    cfg: Optional[ParabolicConfig] = None,
    truth_traj_func: Optional[Callable] = None,
    save_segments: Optional[Sequence[int]] = None,
    verbose: bool = False,
):
    """Run the full segmented IDSM iteration over (0, total_time].

    Implements parabolic_*.edp L342-654 (outer time loop).

    Parameters
    ----------
    mesh : EllipticMesh — the solver mesh (typically `generate_disk_mesh`).
    y_data_fine : array (T_f, N) — synthetic boundary data on the fine grid.
        Each row is the noisy measurement on all P1 nodes at one fine timestep.
        Required entries: T_f = ceil(total_time / forward_dt) + 1.
    forward_dt : float — fine-grid spacing for `y_data_fine`.
    f_func : callable t → (callable (x, y) → array)  — interior source.
    g_func : callable t → (callable (x, y) → array)  — Neumann boundary data.
    initial_data : array (N,) — initial condition y(·, 0).
    cfg : ParabolicConfig — solver parameters.
    truth_traj_func : optional callable t → (centers, radii, active)
        For per-segment IoU evaluation against ground-truth inclusions.
    save_segments : iterable of segment indices to retain full y_history for
        plotting. Defaults to a few sample times.

    Returns
    -------
    dict with keys:
        sigma_history : array (n_segments, M)
        v_pot_history : array (n_segments, M)
        residuals_per_segment : list of lists
        n_inner_per_segment : array (n_segments,)
        y_history_saved : dict {segment_idx: array(n_substeps+1, N)}
        iou_history : array (n_segments,) or None
        runtime_seconds : float
    """
    import time
    if cfg is None:
        cfg = ParabolicConfig()
    n_seg = cfg.n_segments
    n_sub = cfg.delta_t_split
    n_tri = mesh.n_triangles
    save_set = set(save_segments) if save_segments is not None else set(
        np.linspace(0, max(n_seg - 1, 0), 10).astype(int).tolist()
    )

    # cached constant-coefficient FE matrices (cA, vA), .edp L338-341
    M = assemble_mass_matrix(mesh)
    M_bdry = assemble_boundary_mass_matrix(mesh)
    K_const = assemble_stiffness_matrix(
        mesh, np.full(n_tri, cfg.cA)
    )
    M_v_const = assemble_mass_matrix(
        mesh, np.full(n_tri, cfg.vA)
    )

    # initial R₀
    diag_r0 = initialize_r0_parabolic(mesh, exponent=cfg.diag_exponent,
                                       cutoff=cfg.diag_cutoff)
    R = LowRankPreconditioner(diag_r0, method=cfg.lowrank,
                               max_store=cfg.save_num)

    sigma_history = np.zeros((n_seg, n_tri))
    v_pot_history = np.zeros((n_seg, n_tri))
    n_inner_per_segment = np.zeros(n_seg, dtype=int)
    residuals_per_segment: List[List[float]] = []
    y_history_saved: dict = {}
    iou_history: List[float] = []

    sigma_state = {}
    y_last = initial_data.copy()

    t_start = time.time()

    for tIndex in range(n_seg):
        t_begin = tIndex * cfg.delta_t
        t_end = t_begin + cfg.delta_t

        # --- Step 1: background (empty) forward solution -----------------
        y_empty_history = solve_forward_parabolic_segment(
            mesh, M, K_const, M_v_const,
            np.full(n_tri, cfg.cA), np.full(n_tri, cfg.vA),
            y_last, t_begin=t_begin, t_end=t_end, n_substeps=n_sub,
            f_func=f_func, g_func=g_func,
        )

        # --- Step 2: inner iteration -------------------------------------
        sigma, v_pot, y_history, n_inner, residuals = iterate_within_segment(
            mesh, M, K_const, M_v_const, M_bdry, R, cfg,
            y_last=y_last,
            y_empty_history=y_empty_history,
            y_data_fine=y_data_fine,
            forward_dt=forward_dt,
            t_begin=t_begin, t_end=t_end,
            f_func=f_func, g_func=g_func,
            sigma_state=sigma_state,
        )

        # --- Step 3: transition (Dirichlet penalty solve) ---------------
        meas_pairs: List[Tuple[float, np.ndarray]] = []
        for j in range(n_sub):
            t_mid = t_begin + (j + 0.5) * (cfg.delta_t / n_sub)
            meas_pairs.append((t_mid, interpolate_boundary_data(
                y_data_fine, t_mid, forward_dt)))
        y_trans = solve_forward_with_dirichlet_penalty(
            mesh, M, M_bdry, sigma, v_pot, y_last,
            t_begin=t_begin, t_end=t_end, n_substeps=n_sub,
            f_func=f_func, g_func=g_func,
            measurement_times_data=meas_pairs, kappa=cfg.kappa,
        )

        sigma_history[tIndex] = sigma
        v_pot_history[tIndex] = v_pot
        n_inner_per_segment[tIndex] = n_inner
        residuals_per_segment.append(residuals)

        if tIndex in save_set:
            y_history_saved[tIndex] = y_history.copy()

        # IoU at segment end
        if truth_traj_func is not None:
            traj_out = truth_traj_func(t_end)
            centers, radii, active = traj_out[:3]
            truth_ind = circle_indicator_p0(mesh, centers, radii, active)
            if cfg.model == 'conductivity':
                pred_ind = sigma < 0.5 * (cfg.cA + cfg.cB)
            elif cfg.model == 'potential':
                pred_ind = v_pot > 0.5 * (cfg.vA + cfg.vB)
            else:  # double
                pred_ind = (sigma < 0.5 * (cfg.cA + cfg.cB)) | (
                    v_pot > 0.5 * (cfg.vA + cfg.vB)
                )
            inter = float(np.sum((pred_ind & truth_ind) * mesh.areas))
            union = float(np.sum((pred_ind | truth_ind) * mesh.areas))
            iou = inter / union if union > 0 else 0.0
            iou_history.append(iou)

        # next-segment initial condition
        y_last = y_trans[-1]

        if verbose and (tIndex % 10 == 0):
            print(f"  segment {tIndex+1}/{n_seg}: t={t_end:.2f}, "
                  f"inner={n_inner}, residual={residuals[-1]:.4f}")

    runtime = time.time() - t_start
    return {
        'sigma_history': sigma_history,
        'v_pot_history': v_pot_history,
        'residuals_per_segment': residuals_per_segment,
        'n_inner_per_segment': n_inner_per_segment,
        'y_history_saved': y_history_saved,
        'iou_history': np.array(iou_history) if iou_history else None,
        'runtime_seconds': runtime,
    }
