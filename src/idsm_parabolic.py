"""
idsm_parabolic.py — Iterative Direct Sampling Method for parabolic inverse problems.

This module mirrors the parabolic FreeFEM reference programs in ``reference/``
with NumPy/SciPy/scikit-fem building blocks:

1. synthesize noisy forward boundary data with a Crank-Nicolson heat solve,
2. solve the empty-background and adjoint segment problems,
3. assemble the local dual indicator,
4. apply the DFP/BFGS low-rank resolver, and
5. project the indicator to P0 coefficient guesses for each time segment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .mesh import EllipticMesh, coarse_to_fine_p0, fine_to_coarse_p0
from .fem_skfem import (
    assemble_stiffness_matrix,
    assemble_mass_matrix,
    assemble_boundary_mass_matrix,
    assemble_boundary_load,
)
from .idsm import LowRankPreconditioner


# ============================================================
# 1. ParabolicConfig
# ============================================================

@dataclass
class ParabolicConfig:
    """Algorithm parameters shared by the Section 5 parabolic examples.

    Defaults follow ``reference/parabolic_ConductivityMerging.edp``. Individual
    ``edp_cfg_example_5_*`` and ``paper_cfg_example_5_*`` helpers override only
    the values that differ for each example or for notebook-scale runs.
    """

    # Background and inclusion coefficient values.
    cA: float = 1.0
    cB: float = 0.1
    vA: float = 1e-10
    vB: float = 2e-10         # inclusion potential
    model: str = 'conductivity'  # 'conductivity' / 'potential' / 'double'

    # Forward and inverse time grids.
    total_time: float = 10.21
    forward_dt: float = 0.02
    delta_t: float = 0.1
    delta_t_split: int = 6

    # Mesh-resolution knobs kept for parity with getARGV defaults.
    n_solve: int = 80
    n_coarse: int = 80

    # IDSM iteration and synthetic-noise controls.
    save_num: int = 10
    tolerance: float = 0.08
    forget_scale: float = 0.7
    noise_level: float = 0.2
    lowrank: str = 'BFG'       # 'DFP' / 'BFG'

    data_num: int = 1

    # Boundary penalty used in the segment forward solve.
    kappa: float = 1e10

    max_inner: int = 80

    @property
    def inverse_dt(self) -> float:
        """Time step for the sub-steps inside one inverse segment."""
        return self.delta_t / self.delta_t_split

    @property
    def n_segments(self) -> int:
        """Number of inverse segments used by the FreeFEM main loop.

        The reference code sets ``inverseTimeNum = floor(totalTime / deltaT)``
        and then iterates ``tIndex < inverseTimeNum - 1``.  Keeping that
        off-by-one convention matters when comparing per-segment histories and
        Table 1 solve counts against the original ``parabolic_*.edp`` files.
        """
        return max(0, int(np.floor(self.total_time / self.delta_t)) - 1)


# ============================================================
# 2. Trajectory & Source (Example 5.1: ConductivityMerging)
# ============================================================

def trajectory_example_5_1(t: float, traj_index: int) -> np.ndarray:
    """Moving centers for Example 5.1, copied from ``Traj`` in FreeFEM."""
    result = np.zeros(2)
    if traj_index == 0:
        if t < 3.0:
            result[1] = 0.6 * np.cos(4 * t * np.pi / 24)
            result[0] = -0.7 * np.sin(4 * t * np.pi / 24)
        else:
            result[1] = -0.6 * np.cos(4 * t * np.pi / 24)
            result[0] = -0.7 * np.sin(4 * t * np.pi / 24)
    elif traj_index == 1:
        if t < 3.0:
            result[1] = -0.6 * np.cos(4 * t * np.pi / 24)
            result[0] = -0.7 * np.sin(4 * t * np.pi / 24)
        else:
            if t < 6.0:
                result[1] = -0.6 * np.cos(4 * t * np.pi / 24)
                result[0] = -0.7 * np.sin(4 * t * np.pi / 24)
            else:
                result[1] = -0.6 * np.cos(4 * (12.0 - t) * np.pi / 24)
                result[0] = -0.7 * np.sin(4 * (12.0 - t) * np.pi / 24)
    elif traj_index == 2:
        result[0] = 2.0
        result[1] = 2.0
    elif traj_index == 3:
        result[0] = 2.0
        result[1] = 2.0
    return result


def radius_example_5_1(traj_index: int) -> np.ndarray:
    """Ellipse radii for Example 5.1; only trajectories 0 and 1 are active."""
    result = np.zeros(2)
    if traj_index in (0, 1):
        result[0] = 0.2
        result[1] = 0.2
    elif traj_index in (2, 3):
        result[0] = 1e-10
        result[1] = 1e-10
    return result


def c_func_example_5_1(t: float, x: np.ndarray, y: np.ndarray, cfg: ParabolicConfig) -> np.ndarray:
    """Conductivity field for Example 5.1: ``cB`` inside either disk, else ``cA``."""
    x = np.asarray(x); y = np.asarray(y)
    cp1 = trajectory_example_5_1(t, 0); cp2 = trajectory_example_5_1(t, 1)
    r1 = radius_example_5_1(0); r2 = radius_example_5_1(1)
    dis1 = np.sqrt(((x - cp1[0]) / r1[0]) ** 2 + ((y - cp1[1]) / r1[1]) ** 2)
    dis2 = np.sqrt(((x - cp2[0]) / r2[0]) ** 2 + ((y - cp2[1]) / r2[1]) ** 2)
    dis = np.minimum(dis1, dis2)
    return np.where(dis < 1.0, cfg.cB, cfg.cA)


def v_func_example_5_1(t: float, x: np.ndarray, y: np.ndarray, cfg: ParabolicConfig) -> np.ndarray:
    """Potential field for Example 5.1; this conductivity-only case keeps ``vA``."""
    x = np.asarray(x); y = np.asarray(y)
    cp1 = trajectory_example_5_1(t, 2); cp2 = trajectory_example_5_1(t, 3)
    r1 = radius_example_5_1(2); r2 = radius_example_5_1(3)
    dis1 = np.sqrt(((x - cp1[0]) / r1[0]) ** 2 + ((y - cp1[1]) / r1[1]) ** 2)
    dis2 = np.sqrt(((x - cp2[0]) / r2[0]) ** 2 + ((y - cp2[1]) / r2[1]) ** 2)
    dis = np.minimum(dis1, dis2)
    return np.where(dis < 1.0, cfg.vA, cfg.vA)


def rg_source(t: float, x: np.ndarray, y: np.ndarray, data_index: int) -> np.ndarray:
    """Interior source ``RgSource`` used to generate parabolic boundary data."""
    x = np.asarray(x); y = np.asarray(y)
    if data_index == 0:
        return np.sin(t * np.pi / 4) * 25 * np.sin(3 * x) * np.cos(4 * y)
    if data_index == 1:
        return np.sin(t * np.pi / 4) * 25 * np.cos(4 * x) * np.sin(3 * y)
    if data_index == 2:
        return np.sin(t * np.pi / 4) * 32 * np.sin(4 * x) * np.sin(4 * y)
    raise ValueError(f"data_index must be 0/1/2, got {data_index}")


def bd_source(
    t: float,
    x: np.ndarray,
    y: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    data_index: int,
) -> np.ndarray:
    """Neumann boundary source ``BdSource`` evaluated on boundary nodes."""
    x = np.asarray(x); y = np.asarray(y)
    nx = np.asarray(nx); ny = np.asarray(ny)
    if data_index == 0:
        return np.cos(t * np.pi / 6) * (
            3 * np.cos(3 * x) * np.cos(4 * y) * nx
            - 4 * np.sin(3 * x) * np.sin(4 * y) * ny
        )
    if data_index == 1:
        return np.cos(t * np.pi / 6) * (
            -4 * np.sin(4 * x) * np.sin(3 * y) * nx
            + 3 * np.cos(4 * x) * np.cos(3 * y) * ny
        )
    if data_index == 2:
        return np.cos(t * np.pi / 6) * (
            4 * np.cos(4 * x) * np.sin(4 * y) * nx
            + 4 * np.sin(4 * x) * np.cos(4 * y) * ny
        )
    raise ValueError(f"data_index must be 0/1/2, got {data_index}")


def initial_data(x: np.ndarray, y: np.ndarray, data_index: int) -> np.ndarray:
    """Initial condition ``InitialData`` for each illumination index."""
    x = np.asarray(x); y = np.asarray(y)
    if data_index == 0:
        return 3.0 + np.sin(3 * x) * np.cos(4 * y)
    if data_index == 1:
        return 3.0 + np.cos(4 * x) * np.sin(3 * y)
    if data_index == 2:
        return 3.0 + np.sin(4 * x) * np.sin(4 * y)
    raise ValueError(f"data_index must be 0/1/2, got {data_index}")




# ============================================================
# Helpers: P0 projection of P1 products & const operators
# ============================================================

def _project_p1_grad_dot_grad(mesh: EllipticMesh, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Per-triangle P0 value of (grad u)·(grad v), u, v ∈ P1.

    On each triangle T, ∇u and ∇v are constant; returns shape (n_tri,).
    """
    tri = mesh.triangles
    g = mesh.grad_phi   # (n_tri, 3, 2)
    grad_u = (u[tri][:, :, None] * g).sum(axis=1)  # (n_tri, 2)
    grad_v = (v[tri][:, :, None] * g).sum(axis=1)  # (n_tri, 2)
    return (grad_u * grad_v).sum(axis=1)


def _project_p1_product(mesh: EllipticMesh, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Per-triangle P0 value of u*v, u, v ∈ P1.

    Closed-form: ∫_T u v dx = |T|/12 * (Σ_i u_i v_i + Σ_{i<j}(u_i v_j + u_j v_i) + ...)
    More compactly, P0_T(uv) = ∫_T u v / |T| = (Σ_i u_i v_i + 9 * u_avg * v_avg) / 12.
    """
    tri = mesh.triangles
    u_loc = u[tri]; v_loc = v[tri]   # (n_tri, 3)
    dot = (u_loc * v_loc).sum(axis=1)
    u_avg = u_loc.mean(axis=1); v_avg = v_loc.mean(axis=1)
    return (dot + 9.0 * u_avg * v_avg) / 12.0


def _boundary_normals(mesh: EllipticMesh, radius: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Matches the corresponding parabolic FreeFEM reference block."""
    n_pts = mesh.n_points
    nx_full = np.zeros(n_pts); ny_full = np.zeros(n_pts)
    bn = mesh.boundary_nodes
    bx = mesh.points[bn, 0]; by = mesh.points[bn, 1]
    br = np.sqrt(bx ** 2 + by ** 2)
    nx_full[bn] = bx / np.maximum(br, 1e-15)
    ny_full[bn] = by / np.maximum(br, 1e-15)
    return nx_full, ny_full


def project_p1_fine_to_coarse(
    fine_mesh: EllipticMesh,
    coarse_mesh: EllipticMesh,
    field_fine: np.ndarray,
) -> np.ndarray:
    """Matches the corresponding parabolic FreeFEM reference block.


    Parameters
    ----------
    field_fine  : ndarray (fine_mesh.n_points,)

    Returns
    -------
    field_coarse : ndarray (coarse_mesh.n_points,)
    """
    import matplotlib.tri as mtri
    if field_fine.shape != (fine_mesh.n_points,):
        raise ValueError(
            f"field_fine shape {field_fine.shape} != ({fine_mesh.n_points},)"
        )
    tri = mtri.Triangulation(
        fine_mesh.points[:, 0], fine_mesh.points[:, 1], fine_mesh.triangles,
    )
    interp = mtri.LinearTriInterpolator(tri, field_fine)
    cx = coarse_mesh.points[:, 0].copy()
    cy = coarse_mesh.points[:, 1].copy()
    out_masked = interp(cx, cy)
    out = np.asarray(out_masked.filled(np.nan))
    bad = ~np.isfinite(out)

  
    if bad.any():
        eps = 1e-9
        for _ in range(4):
            r = np.hypot(cx[bad], cy[bad])
            scale = np.where(r > 0, (r - eps) / np.maximum(r, 1e-300), 1.0)
            cx_b = cx[bad] * scale
            cy_b = cy[bad] * scale
            out2 = np.asarray(interp(cx_b, cy_b).filled(np.nan))
            ok = np.isfinite(out2)
            idx_bad = np.flatnonzero(bad)
            out[idx_bad[ok]] = out2[ok]
            bad = ~np.isfinite(out)
            if not bad.any():
                break
            eps *= 50.0

  
    if bad.any():
        from scipy.spatial import cKDTree
        tree = cKDTree(fine_mesh.points)
        _, nn_idx = tree.query(np.c_[
            coarse_mesh.points[bad, 0], coarse_mesh.points[bad, 1]
        ])
        out[bad] = field_fine[nn_idx]

    return out


def _same_mesh(mesh_a: EllipticMesh, mesh_b: EllipticMesh) -> bool:
    """Return True when two mesh references have identical P0/P1 dimensions."""
    return (
        mesh_a is mesh_b
        or (
            mesh_a.n_points == mesh_b.n_points
            and mesh_a.n_triangles == mesh_b.n_triangles
            and np.allclose(mesh_a.points, mesh_b.points)
            and np.array_equal(mesh_a.triangles, mesh_b.triangles)
        )
    )


def _coeff_to_solve_p0(
    solve_mesh: EllipticMesh,
    coeff_mesh: EllipticMesh,
    values: np.ndarray,
) -> np.ndarray:
    """Evaluate coarse P0 coefficients on the solve mesh by centroid matching."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape == (solve_mesh.n_triangles,):
        return values
    if values.shape != (coeff_mesh.n_triangles,):
        raise ValueError(
            f"P0 coefficient length {values.shape[0]} does not match "
            f"solve mesh {solve_mesh.n_triangles} or coeff mesh {coeff_mesh.n_triangles}"
        )
    if _same_mesh(solve_mesh, coeff_mesh):
        return values
    return coarse_to_fine_p0(solve_mesh, coeff_mesh, values)


def _solve_to_coeff_p0(
    solve_mesh: EllipticMesh,
    coeff_mesh: EllipticMesh,
    values: np.ndarray,
) -> np.ndarray:
    """Average solve-mesh P0 values onto the coefficient mesh."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape == (coeff_mesh.n_triangles,):
        return values
    if values.shape != (solve_mesh.n_triangles,):
        raise ValueError(
            f"P0 solve value length {values.shape[0]} does not match "
            f"solve mesh {solve_mesh.n_triangles} or coeff mesh {coeff_mesh.n_triangles}"
        )
    if _same_mesh(solve_mesh, coeff_mesh):
        return values
    return fine_to_coarse_p0(solve_mesh, coeff_mesh, values)


@dataclass
class ConstOperators:
    """Pre-assembled time-invariant operators (.edp L352-355)."""
    M: sparse.spmatrix          # P1 mass
    K_cA: sparse.spmatrix       # 0.5 K with σ=cA
    M_vA: sparse.spmatrix       # 0.5 M with V=vA
    M_bdry: sparse.spmatrix     # boundary mass
    A_lhs_csc: sparse.spmatrix  # M/dt + 0.5 K_cA + 0.5 M_vA
    A_lhs_solver: object        # splu factor of A_lhs (csc)
    nx_full: np.ndarray         # (n_pts,) boundary x-normal, 0 inside
    ny_full: np.ndarray         # (n_pts,) boundary y-normal, 0 inside


def assemble_const_operators(mesh: EllipticMesh, cfg: ParabolicConfig) -> ConstOperators:
    """Assemble Amatrix + ancillary operators with single LU factorization (.edp L352-355).

    Amatrix = M/inverseDeltat + 0.5 K(cA) + 0.5 M(vA),  factorize once and reuse for
    Empty/Adjoint solves throughout the run.
    """
    from scipy.sparse.linalg import splu

    M = assemble_mass_matrix(mesh).tocsc()
    K_cA = (0.5 * assemble_stiffness_matrix(mesh, cfg.cA)).tocsc()
    M_vA = (0.5 * assemble_mass_matrix(mesh, cfg.vA)).tocsc()
    M_bdry = assemble_boundary_mass_matrix(mesh).tocsc()
    inv_dt = cfg.inverse_dt
    A_lhs = (M / inv_dt) + K_cA + M_vA
    A_lhs_csc = A_lhs.tocsc()
    A_lhs_solver = splu(A_lhs_csc)
    nx_full, ny_full = _boundary_normals(mesh)
    return ConstOperators(M=M, K_cA=K_cA, M_vA=M_vA, M_bdry=M_bdry,
                          A_lhs_csc=A_lhs_csc, A_lhs_solver=A_lhs_solver,
                          nx_full=nx_full, ny_full=ny_full)


# ============================================================
# 3. Synthesize Forward (.edp L196-253)
# ============================================================

def synthesize_full_forward(
    fine_mesh: EllipticMesh,
    cfg: ParabolicConfig,
    c_func: Callable[[float, np.ndarray, np.ndarray, ParabolicConfig], np.ndarray],
    v_func: Callable[[float, np.ndarray, np.ndarray, ParabolicConfig], np.ndarray],
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Matches the corresponding parabolic FreeFEM reference block.

        (M/dt + 0.5 K_σ + 0.5 M_V) u^{n+1}
        = (M/dt - 0.5 K_σ - 0.5 M_V) u^n + b(t_mid),

        yData[i] = u_fine(node_i) + noise * |u_fine(node_i)|,
        noise ~ Uniform[-noiseLevel, +noiseLevel].

    Returns
    -------
    y_data : ndarray (forward_time_step, data_num, n_pts_fine)
    y_omega_clean : ndarray (forward_time_step, data_num, n_pts_fine)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_pts = fine_mesh.n_points
    n_tri = fine_mesh.n_triangles
    forward_dt = cfg.forward_dt
    n_steps = int(np.ceil(cfg.total_time / forward_dt)) + 1  # .edp L196

  
    centroids = (fine_mesh.points[fine_mesh.triangles[:, 0]]
                 + fine_mesh.points[fine_mesh.triangles[:, 1]]
                 + fine_mesh.points[fine_mesh.triangles[:, 2]]) / 3.0
    cx = centroids[:, 0]; cy = centroids[:, 1]

    M = assemble_mass_matrix(fine_mesh)
    M_csc = M.tocsc()

  
    y_clean = np.zeros((n_steps, cfg.data_num, n_pts))
    y_data = np.zeros((n_steps, cfg.data_num, n_pts))

  
    px = fine_mesh.points[:, 0]; py = fine_mesh.points[:, 1]

  
    bn = fine_mesh.boundary_nodes
    bx = px[bn]; by = py[bn]
    br = np.sqrt(bx ** 2 + by ** 2)
    nx_full = np.zeros(n_pts); ny_full = np.zeros(n_pts)
    nx_full[bn] = bx / np.maximum(br, 1e-15)
    ny_full[bn] = by / np.maximum(br, 1e-15)

  
    M_bdry = assemble_boundary_mass_matrix(fine_mesh)
    M_bdry_csc = M_bdry.tocsc()

    nonlinear = (cfg.model == 'nonlinear')
  
    K_cA_const = assemble_stiffness_matrix(fine_mesh, np.full(n_tri, cfg.cA)) if nonlinear else None
    M_vA_const = assemble_mass_matrix(fine_mesh, np.full(n_tri, cfg.vA)) if nonlinear else None
    K_cA_const_csc = K_cA_const.tocsc() if nonlinear else None
    M_vA_const_csc = M_vA_const.tocsc() if nonlinear else None
    tri = fine_mesh.triangles  # (n_tri, 3) for centroid evaluation of P1 → P0

    for k in range(cfg.data_num):
      
        u0 = initial_data(px, py, k)
        y_clean[0, k] = u0
        y_data[0, k] = u0

        for tIndex in range(n_steps - 1):
            t_mid = tIndex * forward_dt + 0.5 * forward_dt  # .edp L210
            u_prev = y_clean[tIndex, k]

          
            f_vec = rg_source(t_mid, px, py, k)
            rhs_vol = M_csc @ f_vec
            g_vec = bd_source(t_mid, px, py, nx_full, ny_full, k)
            rhs_bdry = M_bdry_csc @ g_vec

            if nonlinear:
                # ---------- Newton + Crank-Nicolson on |y|y*U term (.edp L138-167) ----------
                u_mid_p0 = v_func(t_mid, cx, cy, cfg)  # U(t_mid) evaluated at P0 centroids
                # Pre-evaluate y_n (= u_prev) at centroid → P0
                y_n_p0 = (u_prev[tri[:, 0]] + u_prev[tri[:, 1]] + u_prev[tri[:, 2]]) / 3.0
                abs_yn_p0 = np.abs(y_n_p0)
                M_ynabs_u = assemble_mass_matrix(fine_mesh, abs_yn_p0 * u_mid_p0).tocsc()

                # rhs_const — terms that don't depend on yTmp during Newton (use y_n only)
                #   −[ 0.5 K_cA y_n + 0.5 vA M y_n + 0.5 y_n|y_n| U M − M y_n / dt ]
                #   + rhs_vol + rhs_bdry
                rhs_const = (
                    - 0.5 * (K_cA_const_csc @ u_prev)
                    - 0.5 * (M_vA_const_csc @ u_prev)
                    - 0.5 * (M_ynabs_u @ u_prev)
                    + (M_csc @ u_prev) / forward_dt
                    + rhs_vol + rhs_bdry
                )

                yTmp = u_prev.copy()
                solve_err_tol = 1e-8
                max_newton = 50
                for _newton_it in range(max_newton):
                    yT_p0 = (yTmp[tri[:, 0]] + yTmp[tri[:, 1]] + yTmp[tri[:, 2]]) / 3.0
                    absyT_p0 = np.abs(yT_p0)
                    # LHS A_xi = M/dt + 0.5 K_cA + 0.5 M_vA + M_{|yTmp|*U}
                    M_yTabs_u = assemble_mass_matrix(fine_mesh, absyT_p0 * u_mid_p0).tocsc()
                    A_xi = (M_csc / forward_dt) + 0.5 * K_cA_const_csc + 0.5 * M_vA_const_csc + M_yTabs_u
                    # residual r = -[ A_yT @ yTmp + (M/dt) yTmp ] + rhs_const
                    # where A_yT = 0.5 K_cA + 0.5 M_vA + 0.5 M_{|yTmp|*U}  (note 0.5 not 1.0!)
                    r = (
                        - 0.5 * (K_cA_const_csc @ yTmp)
                        - 0.5 * (M_vA_const_csc @ yTmp)
                        - 0.5 * (M_yTabs_u @ yTmp)
                        - (M_csc @ yTmp) / forward_dt
                        + rhs_const
                    )
                    xi = spsolve(A_xi.tocsr(), r)
                    yTmp = yTmp + xi
                    step_err = float(np.sqrt(xi @ (M_csc @ xi)))  # ‖ξ‖_M (.edp L162 sqrt int (ξ)^2)
                    if step_err < solve_err_tol:
                        break
                u_next = yTmp
            else:
                # ---------- Linear Crank-Nicolson (Ex 5.1/5.2/5.4/5.5) ----------
                sigma_mid = c_func(t_mid, cx, cy, cfg)
                v_mid = v_func(t_mid, cx, cy, cfg)
                K_sig = assemble_stiffness_matrix(fine_mesh, sigma_mid)
                M_v = assemble_mass_matrix(fine_mesh, v_mid)
                A_lhs = (M_csc / forward_dt) + 0.5 * K_sig.tocsc() + 0.5 * M_v.tocsc()
                A_rhs = (M_csc / forward_dt) - 0.5 * K_sig.tocsc() - 0.5 * M_v.tocsc()
                rhs = A_rhs @ u_prev + rhs_vol + rhs_bdry
                u_next = spsolve(A_lhs.tocsr(), rhs)

            y_clean[tIndex + 1, k] = u_next

          
            noise = (2.0 * rng.random(n_pts) - 1.0) * cfg.noise_level
            y_data[tIndex + 1, k] = u_next + noise * np.abs(u_next)

    return y_data, y_clean


# ============================================================
# 4. Boundary Data Interp (.edp L256-271)
# ============================================================

def boundary_data_at(
    t: float,
    data_index: int,
    y_data: np.ndarray,
    forward_dt: float,
) -> np.ndarray:
    """Matches the corresponding parabolic FreeFEM reference block.

    Parameters
    ----------
    y_data : ndarray (n_steps, data_num, n_pts_fine)
    forward_dt : float

    Returns
    -------
    """
    n_steps = y_data.shape[0]
    raw = t / forward_dt
    n2 = int(np.floor(raw))
    n1 = int(np.ceil(raw))
    lam = raw - n2
    n1 = min(n1, n_steps - 1)
    n2 = min(n2, n_steps - 1)
    if n1 == n2:
        return y_data[n1, data_index].copy()
    return y_data[n1, data_index] * lam + y_data[n2, data_index] * (1.0 - lam)


# ============================================================
# 5. R0 diagFunc Init (.edp L275-294)
# ============================================================

def init_diag_func(
    coarse_mesh: EllipticMesh,
    exponent: float = 0.7,
    cutoff: float = 0.01,
    n_boundary_samples: int = 200,
    radius: float = 1.0,
) -> np.ndarray:
    """Matches the corresponding parabolic FreeFEM reference block.


    Parameters
    ----------
    coarse_mesh : EllipticMesh
    exponent : float = 0.7
    cutoff : float = 0.01
    n_boundary_samples : int = 200
    radius : float = 1.0

    Returns
    -------
    diag : ndarray (2*n_tri,)
    """
    centroids = (coarse_mesh.points[coarse_mesh.triangles[:, 0]]
                 + coarse_mesh.points[coarse_mesh.triangles[:, 1]]
                 + coarse_mesh.points[coarse_mesh.triangles[:, 2]]) / 3.0
    cx = centroids[:, 0]; cy = centroids[:, 1]
    n_tri = coarse_mesh.n_triangles

  
    theta = np.arange(n_boundary_samples) * 2.0 * np.pi / n_boundary_samples
    sx = radius * np.cos(theta)
    sy = radius * np.sin(theta)

  
    d2 = (cx[:, None] - sx[None, :]) ** 2 + (cy[:, None] - sy[None, :]) ** 2
    min_d2 = d2.min(axis=1)

    diag_block = np.power(min_d2, exponent)
    diag_block = np.where(min_d2 < cutoff, 0.0, diag_block)

    return np.concatenate([diag_block, diag_block])


# ============================================================
# 6. Empty Forward Segment (.edp L379-389)
# ============================================================

def solve_empty_segment(
    coarse_mesh: EllipticMesh,
    ops: ConstOperators,
    y_last: np.ndarray,
    t_begin: float,
    cfg: ParabolicConfig,
    data_index: int,
) -> np.ndarray:
    """Matches the corresponding parabolic FreeFEM reference block.

    Crank-Nicolson:
        Amatrix u^{j+1} = (M/dt - 0.5 K_cA - 0.5 M_vA) u^j + M f(t_mid) + M_bdry g(t_mid)

    Returns
    -------
    y_empty : ndarray (deltaTsplit + 1, n_pts)
    """
    n_pts = coarse_mesh.n_points
    n_sub = cfg.delta_t_split
    inv_dt = cfg.inverse_dt
    px = coarse_mesh.points[:, 0]; py = coarse_mesh.points[:, 1]

    y = np.zeros((n_sub + 1, n_pts))
    y[0] = y_last
    A_rhs_op = (ops.M / inv_dt) - ops.K_cA - ops.M_vA  # csc
    for j in range(n_sub):
        t_mid = t_begin + (j + 0.5) * inv_dt
        f_vec = rg_source(t_mid, px, py, data_index)
        g_vec = bd_source(t_mid, px, py, ops.nx_full, ops.ny_full, data_index)
        rhs = A_rhs_op @ y[j] + ops.M @ f_vec + ops.M_bdry @ g_vec
        y[j + 1] = ops.A_lhs_solver.solve(rhs)
    return y


# ============================================================
# 7. Backward Adjoint Segment (.edp L394-413)
# ============================================================

def solve_adjoint_segment(
    coarse_mesh: EllipticMesh,
    ops: ConstOperators,
    y_residual_history: np.ndarray,
    measurement_history: np.ndarray,
    cfg: ParabolicConfig,
) -> Tuple[np.ndarray, float]:
    """Matches the corresponding parabolic FreeFEM reference block.

        Amatrix * yDual_new = (M/dt - 0.5 K_cA - 0.5 M_vA) * yDual + M_bdry * yEmptyHistory[j]
    normalScale = 1 / Σ_j ∫_∂ residual[j]^2 ds.

    In the FreeFEM code the variable named ``measurement`` is overwritten by
    ``0.5*(yEmpty[j]+yEmpty[j-1]) - BoundaryData`` before ``normalScale`` is
    accumulated, so the normalization uses the residual energy.

    Parameters
    ----------
    measurement_history : ndarray (deltaTsplit, n_pts)
        Original boundary data kept for traceability with the caller.  The
        FreeFEM-compatible normalization below uses ``y_residual_history``.

    Returns
    -------
    y_dual : ndarray (n_pts,)
    normal_scale : float
    """
    n_sub = cfg.delta_t_split
    n_pts = coarse_mesh.n_points
    inv_dt = cfg.inverse_dt
    if measurement_history.shape != y_residual_history.shape:
        raise ValueError(
            f"measurement_history shape {measurement_history.shape} != "
            f"y_residual_history shape {y_residual_history.shape}"
        )

    A_rhs_op = (ops.M / inv_dt) - ops.K_cA - ops.M_vA
    y_dual = np.zeros(n_pts)
    norm_acc = 0.0

    for j in range(n_sub, 0, -1):
        meas_resid = y_residual_history[j - 1]
        norm_acc += float(meas_resid @ (ops.M_bdry @ meas_resid))
        rhs = A_rhs_op @ y_dual + ops.M_bdry @ meas_resid
        y_dual = ops.A_lhs_solver.solve(rhs)

    normal_scale = 1.0 / max(norm_acc, 1e-300)
    return y_dual, normal_scale


# ============================================================
# 8. Forward Segment with σ/V interp (.edp L443-466, L613-640)
# ============================================================

def solve_forward_segment(
    coarse_mesh: EllipticMesh,
    ops: ConstOperators,
    sigma_prev: np.ndarray,
    sigma_curr: np.ndarray,
    v_prev: np.ndarray,
    v_curr: np.ndarray,
    y_last: np.ndarray,
    t_begin: float,
    cfg: ParabolicConfig,
    data_index: int,
    *,
    is_first_segment: bool,
    dirichlet_data: Optional[Sequence[np.ndarray]] = None,
    coeff_mesh: Optional[EllipticMesh] = None,
) -> np.ndarray:
    """Matches the corresponding parabolic FreeFEM reference block.

                                                  + sigma_curr*(t-t_begin)/(t_end-t_begin)

      LHS += int1d(kappa * u * v),     RHS += int1d(kappa * diriData * v)

    Returns
    -------
    y_history : ndarray (deltaTsplit + 1, n_pts)
    """
    from scipy.sparse.linalg import spsolve as _spsolve

    n_pts = coarse_mesh.n_points
    coeff_mesh = coarse_mesh if coeff_mesh is None else coeff_mesh
    n_sub = cfg.delta_t_split
    inv_dt = cfg.inverse_dt
    delta_t = cfg.delta_t
    t_end = t_begin + delta_t
    px = coarse_mesh.points[:, 0]; py = coarse_mesh.points[:, 1]

    y = np.zeros((n_sub + 1, n_pts))
    y[0] = y_last

    M_csc = ops.M.tocsc()
    M_bdry_csc = ops.M_bdry.tocsc()
    use_kappa = dirichlet_data is not None
    kappa = cfg.kappa if use_kappa else 0.0
    lhs_kappa = (kappa * M_bdry_csc) if use_kappa else None
    sigma_prev_solve = _coeff_to_solve_p0(coarse_mesh, coeff_mesh, sigma_prev)
    sigma_curr_solve = _coeff_to_solve_p0(coarse_mesh, coeff_mesh, sigma_curr)
    v_prev_solve = _coeff_to_solve_p0(coarse_mesh, coeff_mesh, v_prev)
    v_curr_solve = _coeff_to_solve_p0(coarse_mesh, coeff_mesh, v_curr)

    for j in range(n_sub):
        t_mid = t_begin + (j + 0.5) * inv_dt
        if is_first_segment:
            sigma_at = sigma_curr_solve
            v_at = v_curr_solve
        else:
            # FreeFEM uses cGuess[t-1]*(timeNow-timeBegin)/dt
            #       + cGuess[t]*(timeNow-timeEnd)/(-dt).
            w_prev = (t_mid - t_begin) / (t_end - t_begin)
            w_curr = (t_mid - t_end) / (t_begin - t_end)
            sigma_at = sigma_prev_solve * w_prev + sigma_curr_solve * w_curr
            v_at = v_prev_solve * w_prev + v_curr_solve * w_curr

        K_sig = (0.5 * assemble_stiffness_matrix(coarse_mesh, sigma_at)).tocsc()
        M_v = (0.5 * assemble_mass_matrix(coarse_mesh, v_at)).tocsc()
        A_lhs = (M_csc / inv_dt) + K_sig + M_v
        A_rhs = (M_csc / inv_dt) - K_sig - M_v

        f_vec = rg_source(t_mid, px, py, data_index)
        g_vec = bd_source(t_mid, px, py, ops.nx_full, ops.ny_full, data_index)

        rhs = A_rhs @ y[j] + M_csc @ f_vec + M_bdry_csc @ g_vec
        if use_kappa:
            A_lhs = A_lhs + lhs_kappa
            rhs = rhs + kappa * (M_bdry_csc @ dirichlet_data[j])

        y[j + 1] = _spsolve(A_lhs.tocsr(), rhs)
    return y


# ============================================================
# 9. Inclusion Projection (.edp L427-435)
# ============================================================

def apply_inclusion_projection(
    eta: np.ndarray,
    n_tri: int,
    cfg: ParabolicConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Matches the corresponding parabolic FreeFEM reference block.

        [cGrad[], vGrad[]] = eta;
        if(type == "double" || type == "conductivity")
            cGuess[tIndex] = max(min(cGrad, 0.0), -0.99/abs(cA-cB)) * abs(cA-cB) + cA;
        if(type == "double" || type == "potential")
            vGuess[tIndex] = max(min(vGrad, 2.0), 0.0)    * abs(vA-vB) + vA;


    Parameters
    ----------

    Returns
    -------
    v_pot : ndarray (n_tri,)  v_k+1 ∈ [vA, vA + 2·sign(vB-vA)·|vB-vA|]
    """
    if eta.shape[0] != 2 * n_tri:
        raise ValueError(f"eta dim mismatch: got {eta.shape[0]}, expected {2*n_tri}")

    c_grad = eta[:n_tri]
    v_grad = eta[n_tri:]

    delta_c = abs(cfg.cA - cfg.cB)
    delta_v = abs(cfg.vA - cfg.vB)

    if cfg.model in ('double', 'conductivity'):
        if delta_c == 0.0:
            sigma = np.full(n_tri, cfg.cA, dtype=float)
        else:
            c_clip = np.clip(c_grad, -0.99 / delta_c, 0.0)
            sigma = c_clip * delta_c + cfg.cA
    else:
        sigma = np.full(n_tri, cfg.cA, dtype=float)

    if cfg.model in ('double', 'potential'):
        if delta_v == 0.0:
            v_pot = np.full(n_tri, cfg.vA, dtype=float)
        else:
            v_clip = np.clip(v_grad, 0.0, 2.0)
            v_pot = v_clip * delta_v + cfg.vA
    else:
        v_pot = np.full(n_tri, cfg.vA, dtype=float)

    return sigma, v_pot


# ============================================================
# 10. compute_zeta_p0 (.edp L417-419)
# ============================================================

def compute_zeta_p0(
    coarse_mesh: EllipticMesh,
    y_curr: np.ndarray,
    y_last: np.ndarray,
    y_dual: np.ndarray,
    normal_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Matches the corresponding parabolic FreeFEM reference block.

        zetac = 0.5*(Grad(yGuess[k]) + Grad(yLast[k]))'* Grad(yDual[k]) * normalScale[k];
        zetav = 0.5*(yGuess[k] + yLast[k])               * yDual[k]     * normalScale[k];

                             ``_project_p1_product`` (∫ uv / area = (∑u_i v_i + 9·ū·v̄)/12)

    Parameters
    ----------
    coarse_mesh : EllipticMesh
    normal_scale: float             1 / Σ ∫∂ meas² ds

    Returns
    -------
    zeta_c : ndarray (n_tri,)
    zeta_v : ndarray (n_tri,)
    """
    n_pts = coarse_mesh.n_points
    for name, arr in (('y_curr', y_curr), ('y_last', y_last), ('y_dual', y_dual)):
        if arr.shape != (n_pts,):
            raise ValueError(f"{name} shape {arr.shape} != ({n_pts},)")

  
    grad_curr_dot_dual = _project_p1_grad_dot_grad(coarse_mesh, y_curr, y_dual)
    grad_last_dot_dual = _project_p1_grad_dot_grad(coarse_mesh, y_last, y_dual)
    zeta_c = 0.5 * (grad_curr_dot_dual + grad_last_dot_dual) * normal_scale

  
    proj_curr = _project_p1_product(coarse_mesh, y_curr, y_dual)
    proj_last = _project_p1_product(coarse_mesh, y_last, y_dual)
    zeta_v = 0.5 * (proj_curr + proj_last) * normal_scale

    return zeta_c, zeta_v


# ============================================================
# 11. iterate_segment (.edp L416-590)
# ============================================================

def iterate_segment(
    coarse_mesh: EllipticMesh,
    ops: ConstOperators,
    R: LowRankPreconditioner,
    cfg: ParabolicConfig,
    seg_index: int,
    y_last_per_data: Sequence[np.ndarray],
    y_data: np.ndarray,
    forward_dt: float,
    sigma_prev: np.ndarray,
    v_prev: np.ndarray,
    *,
    state: dict,
    coeff_mesh: Optional[EllipticMesh] = None,
) -> dict:
    """Matches the corresponding parabolic FreeFEM reference block.

    Parameters
    ----------
    cfg           : ParabolicConfig
    seg_index     : tIndex (0-based)
    y_last_per_data : list[np.ndarray (n_pts,)] = yLast[k] = yQGuess[tIndex] (.edp L380)

    Returns
    -------
    dict {
        'normal_scale_per_data' : list[float]
    }
    """
    coeff_mesh = coarse_mesh if coeff_mesh is None else coeff_mesh
    n_pts = coarse_mesh.n_points
    n_tri = coeff_mesh.n_triangles
    n_sub = cfg.delta_t_split
    inv_dt = cfg.inverse_dt
    delta_t = cfg.delta_t
    t_begin = seg_index * delta_t
    t_end = t_begin + delta_t
    is_first = (seg_index == 0)

    if y_data.ndim != 3:
        raise ValueError(f"y_data must be (n_steps, data_num, n_pts); got {y_data.shape}")
    if y_data.shape[2] != n_pts:
        raise ValueError(f"y_data n_pts mismatch: {y_data.shape[2]} vs {n_pts}")
    if y_data.shape[1] != cfg.data_num:
        raise ValueError(f"y_data data_num mismatch: {y_data.shape[1]} vs {cfg.data_num}")

  
    y_empty_per_data: list = []
    y_dual_per_data: list = []
    normal_scale_per_data: list = []
    y_guess_per_data: list = []

    for k in range(cfg.data_num):
        y_empty_k = solve_empty_segment(
            coarse_mesh, ops, y_last_per_data[k], t_begin, cfg, data_index=k,
        )
        y_empty_per_data.append(y_empty_k)
      
        y_guess_per_data.append(y_empty_k[n_sub].copy())

    # ---------- Step 2: adjoint with empty residuals (.edp L394-415) -----------
    measurement_history_per_data: list = []
    for k in range(cfg.data_num):
        y_empty_k = y_empty_per_data[k]
        meas_resid = np.empty((n_sub, n_pts))
        meas_hist = np.empty((n_sub, n_pts))
        for j in range(n_sub):
            t_mid = t_begin + (j + 0.5) * inv_dt   # .edp j_edp = j+1: timeNow = (tIndex+(j+0.5)/n_sub)*deltaT
            meas_t = boundary_data_at(t_mid, k, y_data, forward_dt)
            meas_hist[j] = meas_t
            meas_resid[j] = 0.5 * (y_empty_k[j + 1] + y_empty_k[j]) - meas_t
        measurement_history_per_data.append(meas_hist)
        y_dual_k, ns_k = solve_adjoint_segment(
            coarse_mesh, ops, meas_resid, meas_hist, cfg,
        )
        y_dual_per_data.append(y_dual_k)
        normal_scale_per_data.append(ns_k)

  
    sigma_curr = np.full(n_tri, cfg.cA)
    v_curr = np.full(n_tri, cfg.vA)
    residuals: list = []
    local_loop = 0
    save_num = cfg.save_num
    tolerance_save = cfg.tolerance

    while True:
      
        zetac = np.zeros(n_tri); zetav = np.zeros(n_tri)
        for k in range(cfg.data_num):
            zc_solve, zv_solve = compute_zeta_p0(
                coarse_mesh, y_guess_per_data[k], y_last_per_data[k],
                y_dual_per_data[k], normal_scale_per_data[k],
            )
            zc_k = _solve_to_coeff_p0(coarse_mesh, coeff_mesh, zc_solve)
            zv_k = _solve_to_coeff_p0(coarse_mesh, coeff_mesh, zv_solve)
            zetac += zc_k; zetav += zv_k

      
        eta = R.apply(np.concatenate([zetac, zetav]))
        sigma_curr, v_curr = apply_inclusion_projection(eta, n_tri, cfg)

        # ----- 3c: forward + adjoint tilde for each k (.edp L436-486) ---------
        y_dual_tilde_per_data: list = []
        for k in range(cfg.data_num):
            y_hist_k = solve_forward_segment(
                coarse_mesh, ops,
                sigma_prev=sigma_prev, sigma_curr=sigma_curr,
                v_prev=v_prev, v_curr=v_curr,
                y_last=y_last_per_data[k],
                t_begin=t_begin, cfg=cfg, data_index=k,
                is_first_segment=is_first,
                coeff_mesh=coeff_mesh,
            )
            y_guess_per_data[k] = y_hist_k[n_sub].copy()  # .edp L465: yGuess[k] = yU[deltaTsplit]

            # adjoint tilde: meas_tilde[j] = 0.5*(yEmpty[j+1]+yEmpty[j]) - 0.5*(yU[j+1]+yU[j])
            meas_resid_tilde = np.empty((n_sub, n_pts))
            y_empty_k = y_empty_per_data[k]
            for j in range(n_sub):
                meas_resid_tilde[j] = 0.5 * (y_empty_k[j + 1] + y_empty_k[j]) \
                                    - 0.5 * (y_hist_k[j + 1] + y_hist_k[j])
            y_dual_tilde_k, _ = solve_adjoint_segment(
                coarse_mesh, ops, meas_resid_tilde,
                measurement_history_per_data[k], cfg,
            )
            y_dual_tilde_per_data.append(y_dual_tilde_k)

      
        tilde_zc = np.zeros(n_tri); tilde_zv = np.zeros(n_tri)
        for k in range(cfg.data_num):
            zc_solve, zv_solve = compute_zeta_p0(
                coarse_mesh, y_guess_per_data[k], y_last_per_data[k],
                y_dual_tilde_per_data[k], normal_scale_per_data[k],
            )
            zc_k = _solve_to_coeff_p0(coarse_mesh, coeff_mesh, zc_solve)
            zv_k = _solve_to_coeff_p0(coarse_mesh, coeff_mesh, zv_solve)
            tilde_zc += zc_k; tilde_zv += zv_k

        # ----- 3e: cErr/vErr (.edp L487-495) ----------------------------------
        delta_c = abs(cfg.cA - cfg.cB)
        delta_v = abs(cfg.vA - cfg.vB)
        if cfg.model in ('double', 'conductivity'):
            c_err = (sigma_curr - cfg.cA) / delta_c if delta_c > 0 else np.zeros(n_tri)
        else:
            c_err = np.zeros(n_tri)
        if cfg.model in ('double', 'potential'):
            v_err = (v_curr - cfg.vA) / delta_v if delta_v > 0 else np.zeros(n_tri)
        else:
            v_err = np.zeros(n_tri)

      
        store_count = state['store_count']
        slot = store_count % save_num
      
        if slot < len(R.s_store):
            R.s_store[slot] = np.zeros_like(R.s_store[slot])
            R.ry_store[slot] = np.zeros_like(R.ry_store[slot])

        # ----- 3g: forget_scale at localLoop==0 (.edp L504-509) ---------------
        if local_loop == 0:
            n_existing = min(store_count, save_num)
            for j in range(n_existing):
                if j < len(R.s_store):
                    R.s_store[j] *= cfg.forget_scale
                    R.ry_store[j] *= cfg.forget_scale

        # ----- 3h: diagonal scale (.edp L511-553) -----------------------------
        if (local_loop == 0) or (store_count == 0):
            dyk_c = tilde_zc * R.diag[:n_tri]
            dyk_v = tilde_zv * R.diag[n_tri:]
            scale1 = 0.0
            scale2 = 0.0
            areas = coeff_mesh.areas
            if cfg.model in ('double', 'conductivity'):
                num1 = float(np.sum(np.abs(c_err) * areas))
                den1 = float(np.sum(np.abs(dyk_c) * areas))
                if den1 > 0.0:
                    scale1 = num1 / den1
            if cfg.model in ('double', 'potential'):
                num2 = float(np.sum(np.abs(v_err) * areas))
                den2 = float(np.sum(np.abs(dyk_v) * areas))
                if den2 > 0.0:
                    scale2 = num2 / den2
            if scale1 != 0.0:
                R.diag[:n_tri] *= scale1
            if scale2 != 0.0:
                R.diag[n_tri:] *= scale2

        # ----- 3i: yk, ryk; sk via clip-replacement (.edp L554-565) -----------
        yk = np.concatenate([tilde_zc, tilde_zv])
        ryk = R.apply(yk)

        if cfg.model in ('double', 'conductivity'):
            mask_zero = (c_err == 0.0)
            c_err = np.where(mask_zero, np.maximum(ryk[:n_tri], 0.0), c_err)
            if delta_c > 0:
                lim = -0.99 / delta_c
                mask_lim = (c_err == lim)
                c_err = np.where(mask_lim, np.minimum(ryk[:n_tri], -0.99 / 0.90), c_err)
        if cfg.model in ('double', 'potential'):
            mask_two = (v_err == 2.0)
            v_err = np.where(mask_two, np.maximum(ryk[n_tri:], 2.0), v_err)
            mask_zero_v = (v_err == 0.0)
            v_err = np.where(mask_zero_v, np.minimum(ryk[n_tri:], 0.0), v_err)

        sk = np.concatenate([c_err, v_err])

        # ----- 3j: R.update if sk·yk > 0 (.edp L567-572) ----------------------
        if float(sk @ yk) > 0.0:
            R.update(sk, yk, ryk)
            state['store_count'] = R.count

        # ----- 3k: residual check (.edp L573-589) -----------------------------
        err = 0.0
        for k in range(cfg.data_num):
            meas_end = boundary_data_at(t_end, k, y_data, forward_dt)
            diff = y_guess_per_data[k] - meas_end
            num = float(diff @ (ops.M_bdry @ diff))
            den = float(meas_end @ (ops.M_bdry @ meas_end))
            err_k = np.sqrt(max(num, 0.0) / max(den, 1e-300))
            if err_k > err:
                err = err_k
        residuals.append(err)
        local_loop += 1

        if err < state['tolerance']:
            state['tolerance'] = tolerance_save
            break

      
        if local_loop % save_num == 0:
            state['store_count'] = 0
            R.count = 0
            state['tolerance'] *= 1.2

        if local_loop >= cfg.max_inner:
            break

    return {
        'sigma': sigma_curr,
        'v_pot': v_curr,
        'y_guess_per_data': y_guess_per_data,
        'y_dual_per_data': y_dual_per_data,
        'normal_scale_per_data': normal_scale_per_data,
        'y_empty_per_data': y_empty_per_data,
        'residuals': residuals,
        'n_inner': local_loop,
    }


# ============================================================
# 12. finalize_segment (.edp L592-642)
# ============================================================

def finalize_segment(
    coarse_mesh: EllipticMesh,
    ops: ConstOperators,
    R: LowRankPreconditioner,
    cfg: ParabolicConfig,
    seg_index: int,
    y_last_per_data: Sequence[np.ndarray],
    y_data: np.ndarray,
    forward_dt: float,
    sigma_prev: np.ndarray,
    v_prev: np.ndarray,
    y_dual_per_data: Sequence[np.ndarray],
    normal_scale_per_data: Sequence[float],
    y_guess_per_data: Sequence[np.ndarray],
    coeff_mesh: Optional[EllipticMesh] = None,
) -> dict:
    """Matches the corresponding parabolic FreeFEM reference block.


    Returns
    -------
    dict {
        'sigma'            : (n_tri,)
        'v_pot'            : (n_tri,)
        'y_guess_per_data' : list[(n_pts,)]
    }
    """
    coeff_mesh = coarse_mesh if coeff_mesh is None else coeff_mesh
    n_tri = coeff_mesh.n_triangles
    n_sub = cfg.delta_t_split
    inv_dt = cfg.inverse_dt
    delta_t = cfg.delta_t
    t_begin = seg_index * delta_t
    is_first = (seg_index == 0)

  
    zetac = np.zeros(n_tri); zetav = np.zeros(n_tri)
    for k in range(cfg.data_num):
        zc_solve, zv_solve = compute_zeta_p0(
            coarse_mesh, y_guess_per_data[k], y_last_per_data[k],
            y_dual_per_data[k], normal_scale_per_data[k],
        )
        zc_k = _solve_to_coeff_p0(coarse_mesh, coeff_mesh, zc_solve)
        zv_k = _solve_to_coeff_p0(coarse_mesh, coeff_mesh, zv_solve)
        zetac += zc_k; zetav += zv_k

    # ----- step 2: eta -> projection -----
    eta = R.apply(np.concatenate([zetac, zetav]))
    sigma_curr, v_curr = apply_inclusion_projection(eta, n_tri, cfg)

    # ----- step 3: forward with kappa Dirichlet (.edp L612-640) -----
    new_y_guess: list = []
    for k in range(cfg.data_num):
      
        dir_data = []
        for j in range(n_sub):
            t_mid = t_begin + (j + 0.5) * inv_dt
            dir_data.append(boundary_data_at(t_mid, k, y_data, forward_dt))

        y_hist = solve_forward_segment(
            coarse_mesh, ops,
            sigma_prev=sigma_prev, sigma_curr=sigma_curr,
            v_prev=v_prev, v_curr=v_curr,
            y_last=y_last_per_data[k],
            t_begin=t_begin, cfg=cfg, data_index=k,
            is_first_segment=is_first,
            dirichlet_data=dir_data,
            coeff_mesh=coeff_mesh,
        )
        new_y_guess.append(y_hist[n_sub].copy())

    return {
        'sigma': sigma_curr,
        'v_pot': v_curr,
        'y_guess_per_data': new_y_guess,
    }


# ============================================================
# ============================================================

def compute_zeta_u_p0(
    coarse_mesh: EllipticMesh,
    y_curr: np.ndarray,
    y_last: np.ndarray,
    y_dual: np.ndarray,
    normal_scale: float,
) -> np.ndarray:
    """Matches the corresponding parabolic FreeFEM reference block.

        zetau = 0.5*(abs(yGuess[k])*yGuess[k] + abs(yLast[k])*yLast[k]) * yDual[k] * normalScale[k];

    """
    n_pts = coarse_mesh.n_points
    for name, arr in (('y_curr', y_curr), ('y_last', y_last), ('y_dual', y_dual)):
        if arr.shape != (n_pts,):
            raise ValueError(f"{name} shape {arr.shape} != ({n_pts},)")
    psi_curr = np.abs(y_curr) * y_curr
    psi_last = np.abs(y_last) * y_last
    proj_curr = _project_p1_product(coarse_mesh, psi_curr, y_dual)
    proj_last = _project_p1_product(coarse_mesh, psi_last, y_dual)
    return 0.5 * (proj_curr + proj_last) * normal_scale


def apply_inclusion_projection_u(
    eta_u: np.ndarray,
    cfg: ParabolicConfig,
) -> np.ndarray:
    """Matches the corresponding parabolic FreeFEM reference block.

    """
    uA = cfg.vA
    uB = cfg.vB
    return np.clip(eta_u, uA, 2.0 * uB)


def init_diag_func_u(
    coarse_mesh: EllipticMesh,
    exponent: float = 0.7,
    cutoff: float = 0.01,
    n_boundary_samples: int = 200,
    radius: float = 1.0,
) -> np.ndarray:
    """Matches the corresponding parabolic FreeFEM reference block."""
    diag_full = init_diag_func(coarse_mesh, exponent=exponent, cutoff=cutoff,
                                n_boundary_samples=n_boundary_samples, radius=radius)
    return diag_full[:coarse_mesh.n_triangles].copy()


def solve_forward_segment_nonlinear(
    coarse_mesh: EllipticMesh,
    ops: ConstOperators,
    y_last: np.ndarray,
    t_begin: float,
    cfg: ParabolicConfig,
    data_index: int,
    *,
    u_prev_p0: np.ndarray,
    u_curr_p0: np.ndarray,
    is_first_segment: bool,
    dirichlet_data: Optional[Sequence[np.ndarray]] = None,
    coeff_mesh: Optional[EllipticMesh] = None,
) -> np.ndarray:
    """Matches the corresponding parabolic FreeFEM reference block.

    """
    n_pts = coarse_mesh.n_points
    n_tri = coarse_mesh.n_triangles
    coeff_mesh = coarse_mesh if coeff_mesh is None else coeff_mesh
    n_sub = cfg.delta_t_split
    inv_dt = cfg.inverse_dt
    delta_t = cfg.delta_t
    t_end = t_begin + delta_t
    px = coarse_mesh.points[:, 0]; py_ = coarse_mesh.points[:, 1]
    tri = coarse_mesh.triangles

    y = np.zeros((n_sub + 1, n_pts))
    y[0] = y_last.copy()

    M_csc = ops.M.tocsc()
    M_bdry_csc = ops.M_bdry.tocsc()
    use_kappa = dirichlet_data is not None
    kappa = cfg.kappa if use_kappa else 0.0
    lhs_kappa = (kappa * M_bdry_csc) if use_kappa else None
    u_prev_solve = _coeff_to_solve_p0(coarse_mesh, coeff_mesh, u_prev_p0)
    u_curr_solve = _coeff_to_solve_p0(coarse_mesh, coeff_mesh, u_curr_p0)

  
    K_cA_const = assemble_stiffness_matrix(coarse_mesh, np.full(n_tri, cfg.cA)).tocsc()
    M_vA_const = assemble_mass_matrix(coarse_mesh, np.full(n_tri, cfg.vA)).tocsc()

    solve_err_tol = 1e-8
    max_newton = 50

    for j in range(n_sub):
        t_mid = t_begin + (j + 0.5) * inv_dt
        if is_first_segment:
            u_at = u_curr_solve
        else:
            w_prev = (t_mid - t_begin) / (t_end - t_begin)
            w_curr = (t_mid - t_end) / (t_begin - t_end)
            u_at = u_prev_solve * w_prev + u_curr_solve * w_curr

        f_vec = rg_source(t_mid, px, py_, data_index)
        g_vec = bd_source(t_mid, px, py_, ops.nx_full, ops.ny_full, data_index)
        rhs_vol = M_csc @ f_vec
        rhs_bdry = M_bdry_csc @ g_vec

        # y_n = y[j], evaluate at centroid -> P0
        yn = y[j]
        yn_p0 = (yn[tri[:, 0]] + yn[tri[:, 1]] + yn[tri[:, 2]]) / 3.0
        abs_yn_p0 = np.abs(yn_p0)
        M_ynabs_u = assemble_mass_matrix(coarse_mesh, abs_yn_p0 * u_at).tocsc()

        # rhs_const — terms not depending on yTmp during Newton
        rhs_const = (
            - 0.5 * (K_cA_const @ yn)
            - 0.5 * (M_vA_const @ yn)
            - 0.5 * (M_ynabs_u @ yn)
            + (M_csc @ yn) / inv_dt
            + rhs_vol + rhs_bdry
        )

        if use_kappa:
            rhs_const = rhs_const + kappa * (M_bdry_csc @ dirichlet_data[j])

        yTmp = yn.copy()
        for _it in range(max_newton):
            yT_p0 = (yTmp[tri[:, 0]] + yTmp[tri[:, 1]] + yTmp[tri[:, 2]]) / 3.0
            absyT_p0 = np.abs(yT_p0)
            M_yTabs_u = assemble_mass_matrix(coarse_mesh, absyT_p0 * u_at).tocsc()
            A_xi = (M_csc / inv_dt) + 0.5 * K_cA_const + 0.5 * M_vA_const + M_yTabs_u
            if use_kappa:
                A_xi = A_xi + lhs_kappa
            r = (
                - 0.5 * (K_cA_const @ yTmp)
                - 0.5 * (M_vA_const @ yTmp)
                - 0.5 * (M_yTabs_u @ yTmp)
                - (M_csc @ yTmp) / inv_dt
                + rhs_const
            )
            if use_kappa:
                r = r - kappa * (M_bdry_csc @ yTmp)
            xi = spsolve(A_xi.tocsr(), r)
            yTmp = yTmp + xi
            step_err = float(np.sqrt(max(xi @ (M_csc @ xi), 0.0)))
            if step_err < solve_err_tol:
                break
        y[j + 1] = yTmp
    return y


def iterate_segment_nonlinear(
    coarse_mesh: EllipticMesh,
    ops: ConstOperators,
    R: LowRankPreconditioner,
    cfg: ParabolicConfig,
    seg_index: int,
    y_last_per_data: Sequence[np.ndarray],
    y_data: np.ndarray,
    forward_dt: float,
    u_prev: np.ndarray,
    *,
    state: dict,
    coeff_mesh: Optional[EllipticMesh] = None,
) -> dict:
    """Matches the corresponding parabolic FreeFEM reference block.

    Returns
    -------
    dict {'u_curr', 'y_guess_per_data', 'y_dual_per_data', 'normal_scale_per_data',
          'y_empty_per_data', 'residuals', 'n_inner'}
    """
    coeff_mesh = coarse_mesh if coeff_mesh is None else coeff_mesh
    n_pts = coarse_mesh.n_points
    n_tri = coeff_mesh.n_triangles
    n_sub = cfg.delta_t_split
    inv_dt = cfg.inverse_dt
    delta_t = cfg.delta_t
    t_begin = seg_index * delta_t
    t_end = t_begin + delta_t
    is_first = (seg_index == 0)

    if y_data.ndim != 3 or y_data.shape[2] != n_pts or y_data.shape[1] != cfg.data_num:
        raise ValueError(f"y_data shape mismatch: got {y_data.shape}")

  
    y_empty_per_data: list = []
    y_dual_per_data: list = []
    normal_scale_per_data: list = []
    y_guess_per_data: list = []
    for k in range(cfg.data_num):
        y_empty_k = solve_empty_segment(
            coarse_mesh, ops, y_last_per_data[k], t_begin, cfg, data_index=k,
        )
        y_empty_per_data.append(y_empty_k)
        y_guess_per_data.append(y_empty_k[n_sub].copy())

    # ---------- step 2: adjoint w/ empty residuals ----------
    measurement_history_per_data: list = []
    for k in range(cfg.data_num):
        y_empty_k = y_empty_per_data[k]
        meas_resid = np.empty((n_sub, n_pts))
        meas_hist = np.empty((n_sub, n_pts))
        for j in range(n_sub):
            t_mid = t_begin + (j + 0.5) * inv_dt
            meas_t = boundary_data_at(t_mid, k, y_data, forward_dt)
            meas_hist[j] = meas_t
            meas_resid[j] = 0.5 * (y_empty_k[j + 1] + y_empty_k[j]) - meas_t
        measurement_history_per_data.append(meas_hist)
        y_dual_k, ns_k = solve_adjoint_segment(
            coarse_mesh, ops, meas_resid, meas_hist, cfg,
        )
        y_dual_per_data.append(y_dual_k)
        normal_scale_per_data.append(ns_k)

  
    u_curr = np.full(n_tri, cfg.vA)
    residuals: list = []
    local_loop = 0
    save_num = cfg.save_num
    tolerance_save = cfg.tolerance

    uA = cfg.vA
    uB = cfg.vB

    while True:
        # 3a: zetau via current yGuess + yLast + yDual
        zetau = np.zeros(n_tri)
        for k in range(cfg.data_num):
            zu_solve = compute_zeta_u_p0(
                coarse_mesh, y_guess_per_data[k], y_last_per_data[k],
                y_dual_per_data[k], normal_scale_per_data[k],
            )
            zetau += _solve_to_coeff_p0(coarse_mesh, coeff_mesh, zu_solve)

        # 3b: eta -> projection
        eta = R.apply(zetau)
        u_curr = apply_inclusion_projection_u(eta, cfg)

        # 3c: forward (Newton+CN) + adjoint tilde for each k
        y_dual_tilde_per_data: list = []
        for k in range(cfg.data_num):
            y_hist_k = solve_forward_segment_nonlinear(
                coarse_mesh, ops,
                y_last=y_last_per_data[k], t_begin=t_begin, cfg=cfg, data_index=k,
                u_prev_p0=u_prev, u_curr_p0=u_curr,
                is_first_segment=is_first,
                coeff_mesh=coeff_mesh,
            )
            y_guess_per_data[k] = y_hist_k[n_sub].copy()

            meas_resid_tilde = np.empty((n_sub, n_pts))
            y_empty_k = y_empty_per_data[k]
            for j in range(n_sub):
                meas_resid_tilde[j] = 0.5 * (y_empty_k[j + 1] + y_empty_k[j]) \
                                    - 0.5 * (y_hist_k[j + 1] + y_hist_k[j])
            y_dual_tilde_k, _ = solve_adjoint_segment(
                coarse_mesh, ops, meas_resid_tilde,
                measurement_history_per_data[k], cfg,
            )
            y_dual_tilde_per_data.append(y_dual_tilde_k)

        # 3d: tildeZetau
        tilde_zu = np.zeros(n_tri)
        for k in range(cfg.data_num):
            zu_solve = compute_zeta_u_p0(
                coarse_mesh, y_guess_per_data[k], y_last_per_data[k],
                y_dual_tilde_per_data[k], normal_scale_per_data[k],
            )
            tilde_zu += _solve_to_coeff_p0(coarse_mesh, coeff_mesh, zu_solve)

      
        u_err = u_curr.copy()

      
        store_count = state['store_count']
        slot = store_count % save_num
        if slot < len(R.s_store):
            R.s_store[slot] = np.zeros_like(R.s_store[slot])
            R.ry_store[slot] = np.zeros_like(R.ry_store[slot])

        # 3g: forget_scale at localLoop==0
        if local_loop == 0:
            n_existing = min(store_count, save_num)
            for j in range(n_existing):
                if j < len(R.s_store):
                    R.s_store[j] *= cfg.forget_scale
                    R.ry_store[j] *= cfg.forget_scale

        # 3h: diag scale (.edp Nonlinear L444-456)
        if (local_loop == 0) or (store_count == 0):
            ryku = tilde_zu * R.diag
            areas = coeff_mesh.areas
            num = float(np.sum(np.abs(u_err) * areas))
            den = float(np.sum(np.abs(ryku) * areas))
            scale = (num / den) if den > 0.0 else 0.0
            if scale != 0.0:
                R.diag *= scale

        # 3i: yk, ryk; sk via clip-replacement (.edp L460-478)
        yk = tilde_zu.copy()
        ryk = R.apply(yk)

      
        #   uErr[i]==uA: uErr[i] = min(ryk[i], uA)
        #   uErr[i]==2*uB: uErr[i] = max(ryk[i], 2*uB)
        mask_lo = (u_err == uA)
        u_err = np.where(mask_lo, np.minimum(ryk, uA), u_err)
        mask_hi = (u_err == 2.0 * uB)
        u_err = np.where(mask_hi, np.maximum(ryk, 2.0 * uB), u_err)

        sk = u_err

        # 3j: R.update if sk·yk > 0
        if float(sk @ yk) > 0.0:
            R.update(sk, yk, ryk)
            state['store_count'] = R.count

        # 3k: residual check
        err = 0.0
        for k in range(cfg.data_num):
            meas_end = boundary_data_at(t_end, k, y_data, forward_dt)
            diff = y_guess_per_data[k] - meas_end
            num = float(diff @ (ops.M_bdry @ diff))
            den = float(meas_end @ (ops.M_bdry @ meas_end))
            err_k = np.sqrt(max(num, 0.0) / max(den, 1e-300))
            if err_k > err:
                err = err_k
        residuals.append(err)
        local_loop += 1

        if err < state['tolerance']:
            state['tolerance'] = tolerance_save
            break
        if local_loop % save_num == 0:
            state['store_count'] = 0
            R.count = 0
            state['tolerance'] *= 1.2
        if local_loop >= cfg.max_inner:
            break

    return {
        'u_curr': u_curr,
        'y_guess_per_data': y_guess_per_data,
        'y_dual_per_data': y_dual_per_data,
        'normal_scale_per_data': normal_scale_per_data,
        'y_empty_per_data': y_empty_per_data,
        'residuals': residuals,
        'n_inner': local_loop,
    }


def finalize_segment_nonlinear(
    coarse_mesh: EllipticMesh,
    ops: ConstOperators,
    R: LowRankPreconditioner,
    cfg: ParabolicConfig,
    seg_index: int,
    y_last_per_data: Sequence[np.ndarray],
    y_data: np.ndarray,
    forward_dt: float,
    u_prev: np.ndarray,
    y_dual_per_data: Sequence[np.ndarray],
    normal_scale_per_data: Sequence[float],
    y_guess_per_data: Sequence[np.ndarray],
    coeff_mesh: Optional[EllipticMesh] = None,
) -> dict:
    """post-while final refinement for U-recovery (.edp Nonlinear L508-560).

    """
    coeff_mesh = coarse_mesh if coeff_mesh is None else coeff_mesh
    n_tri = coeff_mesh.n_triangles
    n_sub = cfg.delta_t_split
    inv_dt = cfg.inverse_dt
    delta_t = cfg.delta_t
    t_begin = seg_index * delta_t
    is_first = (seg_index == 0)

    zetau = np.zeros(n_tri)
    for k in range(cfg.data_num):
        zu_solve = compute_zeta_u_p0(
            coarse_mesh, y_guess_per_data[k], y_last_per_data[k],
            y_dual_per_data[k], normal_scale_per_data[k],
        )
        zetau += _solve_to_coeff_p0(coarse_mesh, coeff_mesh, zu_solve)
    eta = R.apply(zetau)
    u_curr = apply_inclusion_projection_u(eta, cfg)

    new_y_guess: list = []
    for k in range(cfg.data_num):
        dir_data = []
        for j in range(n_sub):
            t_mid = t_begin + (j + 0.5) * inv_dt
            dir_data.append(boundary_data_at(t_mid, k, y_data, forward_dt))

        y_hist = solve_forward_segment_nonlinear(
            coarse_mesh, ops,
            y_last=y_last_per_data[k], t_begin=t_begin, cfg=cfg, data_index=k,
            u_prev_p0=u_prev, u_curr_p0=u_curr,
            is_first_segment=is_first,
            dirichlet_data=dir_data,
            coeff_mesh=coeff_mesh,
        )
        new_y_guess.append(y_hist[n_sub].copy())

    return {
        'u_curr': u_curr,
        'y_guess_per_data': new_y_guess,
    }




# ============================================================
# 13. run_idsm_parabolic (.edp L367-668, top-level driver)
# ============================================================

def run_idsm_parabolic(
    coarse_mesh: EllipticMesh,
    fine_mesh: EllipticMesh,
    cfg: ParabolicConfig,
    c_func: Callable,
    v_func: Callable,
    truth_traj_func: Optional[Callable] = None,
    *,
    seed: int = 42,
    verbose: bool = False,
    solve_mesh: Optional[EllipticMesh] = None,
) -> dict:
    """Matches the corresponding parabolic FreeFEM reference block.

    Parameters
    ----------
    cfg           : ParabolicConfig

    Returns
    -------
    dict {
        'residuals_per_segment': list[list[float]]
        'n_inner_per_segment'  : list[int]
        'y_data'               : (n_steps, data_num, n_pts_coarse)
    }
    """
    coeff_mesh = coarse_mesh
    solve_mesh = coeff_mesh if solve_mesh is None else solve_mesh
    rng = np.random.default_rng(seed)

  
    if verbose:
        print(
            f"[run_idsm_parabolic] data={fine_mesh.n_points} pts / "
            f"solve={solve_mesh.n_points} pts / coeff={coeff_mesh.n_triangles} tri"
        )
        print(f"  total_time={cfg.total_time} delta_t={cfg.delta_t} forward_dt={cfg.forward_dt}")
    y_data_fine, y_clean_fine = synthesize_full_forward(
        fine_mesh, cfg, c_func, v_func, rng=rng,
    )
    n_steps = y_data_fine.shape[0]
    if verbose:
        print(f"  synthesize done: y_data shape={y_data_fine.shape}"
              f" range=[{y_data_fine.min():.3f},{y_data_fine.max():.3f}]")

  
    n_pts_c = solve_mesh.n_points
    y_data = np.zeros((n_steps, cfg.data_num, n_pts_c))
    for k in range(cfg.data_num):
        for i in range(n_steps):
            y_data[i, k] = project_p1_fine_to_coarse(
                fine_mesh, solve_mesh, y_data_fine[i, k],
            )

    # ===== 3. ConstOperators + R₀ =====
    ops = assemble_const_operators(solve_mesh, cfg)
    is_nonlinear = (getattr(cfg, 'model', None) == 'nonlinear')
    if is_nonlinear:
        diag = init_diag_func_u(coeff_mesh)
    else:
        diag = init_diag_func(coeff_mesh)
    R = LowRankPreconditioner(diag, method=cfg.lowrank, max_store=cfg.save_num)

  
    px = solve_mesh.points[:, 0]; py = solve_mesh.points[:, 1]
    y_quote_history: list = []
    y_init_per_data = []
    for k in range(cfg.data_num):
        y_init_per_data.append(initial_data(px, py, k))
    y_quote_history.append(y_init_per_data[0].copy())  # data 0 used for history slot

  
    n_seg = cfg.n_segments
    sigma_history: list = []
    v_history: list = []
    residuals_per_segment: list = []
    n_inner_per_segment: list = []
    iou_history: list = []

    state = {'store_count': 0, 'tolerance': cfg.tolerance}
    sigma_prev = np.full(coeff_mesh.n_triangles, cfg.cA)
    v_prev = np.full(coeff_mesh.n_triangles, cfg.vA)
    u_prev = np.full(coeff_mesh.n_triangles, cfg.vA)  # U-recovery prev (Ex 5.3 only)

    centers = (coeff_mesh.points[coeff_mesh.triangles[:, 0]]
               + coeff_mesh.points[coeff_mesh.triangles[:, 1]]
               + coeff_mesh.points[coeff_mesh.triangles[:, 2]]) / 3.0
    cx = centers[:, 0]; cy = centers[:, 1]
    areas = coeff_mesh.areas

  
    y_last_per_data = [y_init_per_data[k].copy() for k in range(cfg.data_num)]

    for tIndex in range(n_seg):
        import os as _os
        if is_nonlinear:
            iter_res = iterate_segment_nonlinear(
                solve_mesh, ops, R, cfg, tIndex,
                y_last_per_data=y_last_per_data,
                y_data=y_data,
                forward_dt=cfg.forward_dt,
                u_prev=u_prev,
                state=state,
                coeff_mesh=coeff_mesh,
            )
            residuals_per_segment.append(iter_res['residuals'])
            n_inner_per_segment.append(iter_res['n_inner'])
            if _os.environ.get('IDSM_SKIP_FINALIZE', '0') == '1':
                u_curr = iter_res['u_curr']
                y_guess_per_data = iter_res['y_guess_per_data']
            else:
                final_res = finalize_segment_nonlinear(
                    solve_mesh, ops, R, cfg, tIndex,
                    y_last_per_data=y_last_per_data,
                    y_data=y_data,
                    forward_dt=cfg.forward_dt,
                    u_prev=u_prev,
                    y_dual_per_data=iter_res['y_dual_per_data'],
                    normal_scale_per_data=iter_res['normal_scale_per_data'],
                    y_guess_per_data=iter_res['y_guess_per_data'],
                    coeff_mesh=coeff_mesh,
                )
                u_curr = final_res['u_curr']
                y_guess_per_data = final_res['y_guess_per_data']
          
            sigma_curr = np.full(coeff_mesh.n_triangles, cfg.cA)
            v_curr = u_curr
        else:
            # iterate_segment
            iter_res = iterate_segment(
                solve_mesh, ops, R, cfg, tIndex,
                y_last_per_data=y_last_per_data,
                y_data=y_data,
                forward_dt=cfg.forward_dt,
                sigma_prev=sigma_prev, v_prev=v_prev,
                state=state,
                coeff_mesh=coeff_mesh,
            )
            residuals_per_segment.append(iter_res['residuals'])
            n_inner_per_segment.append(iter_res['n_inner'])

            # finalize_segment (post-while final refinement)
          
            if _os.environ.get('IDSM_SKIP_FINALIZE', '0') == '1':
                sigma_curr = iter_res['sigma']
                v_curr = iter_res['v_pot']
                y_guess_per_data = iter_res['y_guess_per_data']
            else:
                final_res = finalize_segment(
                    solve_mesh, ops, R, cfg, tIndex,
                    y_last_per_data=y_last_per_data,
                    y_data=y_data,
                    forward_dt=cfg.forward_dt,
                    sigma_prev=sigma_prev, v_prev=v_prev,
                    y_dual_per_data=iter_res['y_dual_per_data'],
                    normal_scale_per_data=iter_res['normal_scale_per_data'],
                    y_guess_per_data=iter_res['y_guess_per_data'],
                    coeff_mesh=coeff_mesh,
                )
                sigma_curr = final_res['sigma']
                v_curr = final_res['v_pot']
                y_guess_per_data = final_res['y_guess_per_data']

        sigma_history.append(sigma_curr.copy())
        v_history.append(v_curr.copy())
        y_quote_history.append(y_guess_per_data[0].copy())  # data 0 history slot

        # IoU vs truth at t_end
      
      
      
      
        t_end = (tIndex + 1) * cfg.delta_t
        if cfg.model == 'potential':
            v_true = v_func(t_end, cx, cy, cfg)
            v_thr = 0.5 * (cfg.vA + cfg.vB)
            true_hi = v_true > v_thr
            pred_hi = v_curr > v_thr
            inter = float(np.sum((true_hi & pred_hi) * areas))
            union = float(np.sum((true_hi | pred_hi) * areas))
        elif cfg.model == 'nonlinear':
            # Ex 5.3: sigma is constant; the reconstructed U field is stored in
            # v_curr so the common plotting/result pipeline can be reused.
            u_true = v_func(t_end, cx, cy, cfg)
            u_thr = 0.5 * (cfg.vA + cfg.vB)
            true_hi = u_true > u_thr
            pred_hi = v_curr > u_thr
            inter = float(np.sum((true_hi & pred_hi) * areas))
            union = float(np.sum((true_hi | pred_hi) * areas))
        else:
            sig_true = c_func(t_end, cx, cy, cfg)
            true_low = sig_true < 0.5 * (cfg.cA + cfg.cB)
            pred_low = sigma_curr < 0.5 * (cfg.cA + cfg.cB)
            inter = float(np.sum((true_low & pred_low) * areas))
            union = float(np.sum((true_low | pred_low) * areas))
        iou = inter / max(union, 1e-300)
        iou_history.append(iou)

        if verbose:
            n_in = iter_res['n_inner']
            res_last = iter_res['residuals'][-1] if iter_res['residuals'] else float('nan')
            print(f"  seg {tIndex:3d} t={t_end:6.3f} n_inner={n_in:3d} "
                  f"resid={res_last:.4f} σ∈[{sigma_curr.min():.3f},{sigma_curr.max():.3f}] "
                  f"IoU={iou:.3f}")

      
        if is_nonlinear:
            u_prev = u_curr
        else:
            sigma_prev = sigma_curr
            v_prev = v_curr
        y_last_per_data = [y_guess_per_data[k].copy() for k in range(cfg.data_num)]

    return {
        'sigma_history': sigma_history,
        'v_history': v_history,
        'y_quote_history': y_quote_history,
        'residuals_per_segment': residuals_per_segment,
        'n_inner_per_segment': n_inner_per_segment,
        'iou_history': iou_history,
        'y_data': y_data,
    }


# ============================================================
# ============================================================

def edp_cfg_example_5_1(noise: float = 0.2) -> ParabolicConfig:
    """ConductivityMerging defaults (.edp L7-31)."""
    return ParabolicConfig(
        cA=1.0, cB=0.1, vA=1e-10, vB=2e-10,
        model='conductivity',
        total_time=10.21, forward_dt=0.02, delta_t=0.1, delta_t_split=6,
        n_solve=80, n_coarse=80,
        save_num=10, tolerance=0.08, forget_scale=0.7,
        noise_level=noise, lowrank='BFG', data_num=1,
        kappa=1e10, max_inner=80,
    )


# ============================================================
# 15. Inclusion geometry & ground truth (for IoU)
# ============================================================

def ground_truth_p0_example_5_1(
    coarse_mesh: EllipticMesh, t: float, cfg: ParabolicConfig,
) -> np.ndarray:
    """Matches the corresponding parabolic FreeFEM reference block."""
    centers = (coarse_mesh.points[coarse_mesh.triangles[:, 0]]
               + coarse_mesh.points[coarse_mesh.triangles[:, 1]]
               + coarse_mesh.points[coarse_mesh.triangles[:, 2]]) / 3.0
    cx = centers[:, 0]; cy = centers[:, 1]
    return c_func_example_5_1(t, cx, cy, cfg)


# ============================================================
# ============================================================

def trajectory_example_5_2(t: float, traj_index: int) -> np.ndarray:
    """Matches the corresponding parabolic FreeFEM reference block.

    """
    result = np.zeros(2)
    if traj_index == 0:
        result[0] = 0.65 * np.cos(1 * t * np.pi / 8 - 7 * np.pi / 6)
        result[1] = 0.65 * np.sin(1 * t * np.pi / 8 - 7 * np.pi / 6)
    elif traj_index == 1:
        result[0] = 0.6 * np.cos(1 * t * np.pi / 8 - 2 * np.pi / 6)
        result[1] = 0.7 * np.sin(1 * t * np.pi / 8 - 2 * np.pi / 6)
    elif traj_index == 2:
        result[0] = 0.65 * np.cos(1 * t * np.pi / 8 - 2 * np.pi / 6)
        result[1] = 0.65 * np.sin(1 * t * np.pi / 8 - 2 * np.pi / 6)
    elif traj_index == 3:
        result[0] = 2.0
        result[1] = 2.0
    return result


def c_func_example_5_2(t: float, x: np.ndarray, y: np.ndarray, cfg: ParabolicConfig) -> np.ndarray:
    """Matches the corresponding parabolic FreeFEM reference block."""
    x = np.asarray(x); y = np.asarray(y)
    cp1 = trajectory_example_5_2(t, 0); cp2 = trajectory_example_5_2(t, 1)
    dis1 = np.sqrt((x - cp1[0]) ** 2 + (y - cp1[1]) ** 2)
    dis2 = np.sqrt((x - cp2[0]) ** 2 + (y - cp2[1]) ** 2)
    dis = np.minimum(dis1, dis2)
    return np.where(dis < 0.2, cfg.cB, cfg.cA)


def v_func_example_5_2(t: float, x: np.ndarray, y: np.ndarray, cfg: ParabolicConfig) -> np.ndarray:
    """Matches the corresponding parabolic FreeFEM reference block."""
    x = np.asarray(x); y = np.asarray(y)
    cp1 = trajectory_example_5_2(t, 2); cp2 = trajectory_example_5_2(t, 3)
    dis1 = np.sqrt((x - cp1[0]) ** 2 + (y - cp1[1]) ** 2)
    dis2 = np.sqrt((x - cp2[0]) ** 2 + (y - cp2[1]) ** 2)
    dis = np.minimum(dis1, dis2)
    return np.where(dis < 0.2, cfg.vB, cfg.vA)


def edp_cfg_example_5_2(noise: float = 0.05) -> ParabolicConfig:
    """MixedMoving .edp defaults (parabolic_MixedMoving.edp L7-31).

    forwardDeltat=0.015; vB=15.0 (vs Ex 5.1 vB=2e-10).
    """
    return ParabolicConfig(
        cA=1.0, cB=0.1, vA=1e-10, vB=15.0,
        model='double',
        total_time=10.31, forward_dt=0.015, delta_t=0.1, delta_t_split=8,
        n_solve=80, n_coarse=80,
        save_num=10, tolerance=0.08, forget_scale=0.7,
        noise_level=noise, lowrank='DFP', data_num=1,
        kappa=1e10, max_inner=80,
    )


def ground_truth_p0_example_5_2(coarse_mesh: EllipticMesh, t: float, cfg: ParabolicConfig) -> np.ndarray:
    centers = (coarse_mesh.points[coarse_mesh.triangles[:, 0]]
               + coarse_mesh.points[coarse_mesh.triangles[:, 1]]
               + coarse_mesh.points[coarse_mesh.triangles[:, 2]]) / 3.0
    return c_func_example_5_2(t, centers[:, 0], centers[:, 1], cfg)


# ============================================================
# 17. Example 5.3 (Nonlinear): N(y)u = u·y·|y|, p=3 in Eq.2.5
# ============================================================

def trajectory_example_5_3(t: float, traj_index: int = 0) -> np.ndarray:
    """Matches the corresponding parabolic FreeFEM reference block."""
    result = np.zeros(2)
    if traj_index == 0:
        result[0] = 0.5 * np.cos(4 * t * np.pi / 24 + np.pi / 4)
        result[1] = 0.7 * np.sin(4 * t * np.pi / 24 + np.pi / 4)
    return result


def u_func_example_5_3(t: float, x: np.ndarray, y: np.ndarray, cfg: ParabolicConfig) -> np.ndarray:
    """Matches the corresponding parabolic FreeFEM reference block."""
    x = np.asarray(x); y = np.asarray(y)
    cp = trajectory_example_5_3(t, 0)
    dis = np.sqrt((x - cp[0]) ** 2 + (y - cp[1]) ** 2)
  
    return np.where(dis < 0.2, cfg.vB, cfg.vA)


def c_func_example_5_3(t: float, x: np.ndarray, y: np.ndarray, cfg: ParabolicConfig) -> np.ndarray:
    """Matches the corresponding parabolic FreeFEM reference block."""
    x = np.asarray(x)
    return np.full_like(x, cfg.cA, dtype=float)


def v_func_example_5_3(t: float, x: np.ndarray, y: np.ndarray, cfg: ParabolicConfig) -> np.ndarray:
    """Matches the corresponding parabolic FreeFEM reference block."""
    return u_func_example_5_3(t, x, y, cfg)


def edp_cfg_example_5_3(noise: float = 0.05) -> ParabolicConfig:
    """Nonlinear .edp defaults (parabolic_Nonlinear.edp L8-31)."""
    return ParabolicConfig(
        cA=1.0, cB=1.0, vA=1e-10, vB=20.0,
        model='nonlinear',
        total_time=0.51, forward_dt=0.02, delta_t=0.1, delta_t_split=6,
        n_solve=200, n_coarse=80,
        save_num=10, tolerance=0.08, forget_scale=0.7,
        noise_level=noise, lowrank='BFG', data_num=1,
        kappa=1e10, max_inner=80,
    )


def ground_truth_p0_example_5_3(coarse_mesh: EllipticMesh, t: float, cfg: ParabolicConfig) -> np.ndarray:
    centers = (coarse_mesh.points[coarse_mesh.triangles[:, 0]]
               + coarse_mesh.points[coarse_mesh.triangles[:, 1]]
               + coarse_mesh.points[coarse_mesh.triangles[:, 2]]) / 3.0
    return u_func_example_5_3(t, centers[:, 0], centers[:, 1], cfg)


# ============================================================
# ============================================================

def trajectory_example_5_4(t: float, traj_index: int) -> np.ndarray:
    """Matches the corresponding parabolic FreeFEM reference block.

    """
    result = np.zeros(2)
    if traj_index in (0, 1):
        result[0] = 2.0
        result[1] = 2.0
    elif traj_index == 2:
        result[0] = 0.7 * np.cos(3 * t * np.pi / 24)
        result[1] = 0.6 * np.sin(3 * t * np.pi / 24)
    elif traj_index == 3:
        result[0] = 0.5 * np.cos(3 * t * np.pi / 24 + 4 * np.pi / 5)
        result[1] = 0.6 * np.sin(3 * t * np.pi / 24 + 4 * np.pi / 5)
    return result


def radius_example_5_4(traj_index: int) -> np.ndarray:
    """Matches the corresponding parabolic FreeFEM reference block."""
    result = np.zeros(2)
    if traj_index in (0, 1):
        result[0] = 1e-10
        result[1] = 1e-10
    elif traj_index in (2, 3):
        result[0] = 0.2
        result[1] = 0.2
    return result


def c_func_example_5_4(t: float, x: np.ndarray, y: np.ndarray, cfg: ParabolicConfig) -> np.ndarray:
    """Matches the corresponding parabolic FreeFEM reference block."""
    x = np.asarray(x)
    return np.full_like(x, cfg.cA, dtype=float)


def v_func_example_5_4(t: float, x: np.ndarray, y: np.ndarray, cfg: ParabolicConfig) -> np.ndarray:
    """Matches the corresponding parabolic FreeFEM reference block.

    """
    x = np.asarray(x); y = np.asarray(y)
    cp1 = trajectory_example_5_4(t, 2); cp2 = trajectory_example_5_4(t, 3)
    r1 = radius_example_5_4(2); r2 = radius_example_5_4(3)
    dis1 = np.sqrt(((x - cp1[0]) / r1[0]) ** 2 + ((y - cp1[1]) / r1[1]) ** 2)
    dis2 = np.sqrt(((x - cp2[0]) / r2[0]) ** 2 + ((y - cp2[1]) / r2[1]) ** 2)
    v_decay = max(cfg.vB + t * (cfg.vA - cfg.vB) / 6.0, cfg.vA)
    v_grow = min(cfg.vA + t * (cfg.vB - cfg.vA) / 6.0, cfg.vB)
    out = np.full_like(x, cfg.vA, dtype=float)
    out = np.where(dis1 < 1.0, v_decay, out)
  
    only_dis2 = (dis1 >= 1.0) & (dis2 < 1.0)
    out = np.where(only_dis2, v_grow, out)
    return out


def edp_cfg_example_5_4(noise: float = 0.05) -> ParabolicConfig:
    """PotentialFading .edp defaults (parabolic_PotentialFading.edp L7-31).

    """
    return ParabolicConfig(
        cA=1.0, cB=1.0 + 1e-10, vA=1e-10, vB=15.0,
        model='potential',
        total_time=10.31, forward_dt=0.02, delta_t=0.1, delta_t_split=6,
        n_solve=200, n_coarse=80,
        save_num=10, tolerance=0.08, forget_scale=0.7,
        noise_level=noise, lowrank='BFG', data_num=1,
        kappa=1e10, max_inner=80,
    )


def ground_truth_v_p0_example_5_4(coarse_mesh: EllipticMesh, t: float, cfg: ParabolicConfig) -> np.ndarray:
    """Matches the corresponding parabolic FreeFEM reference block."""
    centers = (coarse_mesh.points[coarse_mesh.triangles[:, 0]]
               + coarse_mesh.points[coarse_mesh.triangles[:, 1]]
               + coarse_mesh.points[coarse_mesh.triangles[:, 2]]) / 3.0
    return v_func_example_5_4(t, centers[:, 0], centers[:, 1], cfg)


# ============================================================
# ============================================================

def trajectory_example_5_5(t: float, traj_index: int) -> np.ndarray:
    """Matches the corresponding parabolic FreeFEM reference block.

    """
    result = np.zeros(2)
    if traj_index == 0:
        result[0] = 0.7 * np.cos(4 * t * np.pi / 24 + 1 * np.pi / 3)
        result[1] = 0.6 * np.sin(4 * t * np.pi / 24 + 1 * np.pi / 3)
    elif traj_index == 1:
        result[0] = 0.6 * np.cos(4 * t * np.pi / 24 - 2 * np.pi / 3)
        result[1] = 0.5 * np.sin(4 * t * np.pi / 24 - 2 * np.pi / 3)
    elif traj_index in (2, 3):
        result[0] = 2.0
        result[1] = 2.0
    return result


def radius_example_5_5(t: float, traj_index: int) -> np.ndarray:
    """Matches the corresponding parabolic FreeFEM reference block."""
    result = np.zeros(2)
    if traj_index == 0:
        result[0] = 0.2
        result[1] = 0.2
    elif traj_index == 1:
        r = max(0.3 - 0.03 * t, 1e-10)
        result[0] = r
        result[1] = r
    elif traj_index in (2, 3):
        result[0] = 1e-10
        result[1] = 1e-10
    return result


def c_func_example_5_5(t: float, x: np.ndarray, y: np.ndarray, cfg: ParabolicConfig) -> np.ndarray:
    """Matches the corresponding parabolic FreeFEM reference block."""
    x = np.asarray(x); y = np.asarray(y)
    cp1 = trajectory_example_5_5(t, 0); cp2 = trajectory_example_5_5(t, 1)
    r1 = radius_example_5_5(t, 0); r2 = radius_example_5_5(t, 1)
    dis1 = np.sqrt(((x - cp1[0]) / r1[0]) ** 2 + ((y - cp1[1]) / r1[1]) ** 2)
    dis2 = np.sqrt(((x - cp2[0]) / r2[0]) ** 2 + ((y - cp2[1]) / r2[1]) ** 2)
    dis = np.minimum(dis1, dis2)
    return np.where(dis < 1.0, cfg.cB, cfg.cA)


def v_func_example_5_5(t: float, x: np.ndarray, y: np.ndarray, cfg: ParabolicConfig) -> np.ndarray:
    """Matches the corresponding parabolic FreeFEM reference block."""
    x = np.asarray(x)
    return np.full_like(x, cfg.vA, dtype=float)


def edp_cfg_example_5_5(noise: float = 0.05) -> ParabolicConfig:
    """Conductivity-Diminishing .edp defaults (parabolic_ConductivityDiminishing.edp L7-31).

    type='conductivity'; lowrank='DFP'; nSolve=100; total_time=5.31.
    """
    return ParabolicConfig(
        cA=1.0, cB=0.1, vA=1e-10, vB=2e-10,
        model='conductivity',
        total_time=5.31, forward_dt=0.02, delta_t=0.1, delta_t_split=6,
        n_solve=100, n_coarse=80,
        save_num=10, tolerance=0.08, forget_scale=0.7,
        noise_level=noise, lowrank='DFP', data_num=1,
        kappa=1e10, max_inner=80,
    )


def ground_truth_p0_example_5_5(coarse_mesh: EllipticMesh, t: float, cfg: ParabolicConfig) -> np.ndarray:
    centers = (coarse_mesh.points[coarse_mesh.triangles[:, 0]]
               + coarse_mesh.points[coarse_mesh.triangles[:, 1]]
               + coarse_mesh.points[coarse_mesh.triangles[:, 2]]) / 3.0
    return c_func_example_5_5(t, centers[:, 0], centers[:, 1], cfg)


# ============================================================
# ============================================================

def paper_cfg_example_5_1(noise: float = 0.05) -> ParabolicConfig:
    """Paper-style Example 5.1 noise with FreeFEM numerical defaults."""
    return edp_cfg_example_5_1(noise=noise)


def paper_cfg_example_5_2(noise: float = 0.05) -> ParabolicConfig:
    """Paper-style Example 5.2 noise with FreeFEM numerical defaults."""
    return edp_cfg_example_5_2(noise=noise)


def paper_cfg_example_5_3(noise: float = 0.05) -> ParabolicConfig:
    """Paper-style Example 5.3 noise with FreeFEM numerical defaults."""
    return edp_cfg_example_5_3(noise=noise)


def paper_cfg_example_5_4(noise: float = 0.05) -> ParabolicConfig:
    """Paper-style Example 5.4 noise with FreeFEM numerical defaults."""
    return edp_cfg_example_5_4(noise=noise)


def paper_cfg_example_5_5(noise: float = 0.05) -> ParabolicConfig:
    """Paper-style Example 5.5 noise with FreeFEM numerical defaults."""
    return edp_cfg_example_5_5(noise=noise)
