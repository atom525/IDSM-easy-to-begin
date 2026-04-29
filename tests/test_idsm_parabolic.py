"""
test_idsm_parabolic.py
======================

Unit tests for src/idsm_parabolic.py — Phase 5 (Paper 2 / arXiv:2511.08197).

Tests cover the forward Crank-Nicolson solver, the backward adjoint solver,
local-dual P0 evaluation, R₀ initialization, projection, and the inter-segment
forgetScale damping. Each test is short (≤ a few seconds) so the full suite
stays within the project's `pytest` budget.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.mesh import generate_disk_mesh
from src.fem_skfem import (
    assemble_stiffness_matrix,
    assemble_mass_matrix,
    assemble_boundary_mass_matrix,
)
from src.idsm import LowRankPreconditioner
from src.idsm_parabolic import (
    ParabolicConfig,
    solve_forward_parabolic_segment,
    solve_backward_adjoint_segment,
    compute_local_dual,
    initialize_r0_parabolic,
    apply_inclusion_projection,
    damp_lowrank_state,
    _assemble_cn_operator,
)


@pytest.fixture(scope='module')
def disk_mesh():
    return generate_disk_mesh(n_boundary=60)


@pytest.fixture(scope='module')
def fem_matrices(disk_mesh):
    M = assemble_mass_matrix(disk_mesh)
    K_bg = assemble_stiffness_matrix(disk_mesh, np.ones(disk_mesh.n_triangles))
    M_v_bg = assemble_mass_matrix(
        disk_mesh, np.full(disk_mesh.n_triangles, 1e-10)
    )
    return M, K_bg, M_v_bg


def test_config_defaults():
    cfg = ParabolicConfig()
    assert cfg.total_time == pytest.approx(10.21)
    assert cfg.delta_t == 0.1
    assert cfg.delta_t_split == 6
    assert cfg.inverse_dt == pytest.approx(0.1 / 6.0)
    assert cfg.n_segments == 102
    assert cfg.lowrank == 'BFG'
    assert cfg.diag_exponent == 0.7


def test_cn_operator_symmetric_positive(disk_mesh, fem_matrices):
    M, K_bg, M_v_bg = fem_matrices
    A = _assemble_cn_operator(M, K_bg, M_v_bg, dt=0.05).toarray()
    assert np.allclose(A, A.T, atol=1e-10)
    eigs = np.linalg.eigvalsh(A)
    assert eigs.min() > 0.0


def test_forward_constant_initial_constant_solution(disk_mesh, fem_matrices):
    """y_0 ≡ const, f=0, g=0, V≈0  ⇒  y(t) ≡ const (modulo V·const decay)."""
    M, K_bg, M_v_bg = fem_matrices
    n = disk_mesh.n_points
    y_init = np.full(n, 2.5)

    def f_zero(t):
        return lambda x, y: np.zeros_like(x)

    def g_zero(t):
        return lambda x, y: np.zeros_like(x)

    sigma_p0 = np.ones(disk_mesh.n_triangles)
    v_p0 = np.full(disk_mesh.n_triangles, 1e-10)

    y_hist = solve_forward_parabolic_segment(
        disk_mesh, M, K_bg, M_v_bg, sigma_p0, v_p0,
        y_init, t_begin=0.0, t_end=0.5, n_substeps=6,
        f_func=f_zero, g_func=g_zero,
    )
    # decay rate is ~vA ≈ 1e-10 over T=0.5  ⇒  drift ~5e-11
    assert y_hist.shape == (7, n)
    assert np.allclose(y_hist[-1], 2.5, atol=1e-6)


def test_forward_eigenmode_decay(disk_mesh, fem_matrices):
    """Heat equation eigenfunction y_0 = J_0(α₀ r) decays as exp(-α₀² t).

    For the unit disk with Neumann BC and no potential, the leading non-trivial
    eigenmode has eigenvalue ≈ 14.682 (J_1'(α) = 0 root, α₀ ≈ 3.83171).
    We test with the simpler smooth initial condition cos(π·r²/2) which is not
    a single eigenmode but should decay monotonically.
    """
    M, K_bg, M_v_bg = fem_matrices
    n = disk_mesh.n_points
    p = disk_mesh.points
    r2 = p[:, 0]**2 + p[:, 1]**2

    y_init = np.cos(np.pi * r2 / 2.0)  # smooth, dies off near boundary

    def f_zero(t):
        return lambda x, y: np.zeros_like(x)

    def g_zero(t):
        return lambda x, y: np.zeros_like(x)

    sigma_p0 = np.ones(disk_mesh.n_triangles)
    v_p0 = np.full(disk_mesh.n_triangles, 1e-10)

    y_hist = solve_forward_parabolic_segment(
        disk_mesh, M, K_bg, M_v_bg, sigma_p0, v_p0,
        y_init, t_begin=0.0, t_end=1.0, n_substeps=20,
        f_func=f_zero, g_func=g_zero,
    )
    # L2 energy must decrease in time (parabolic dissipation)
    energies = np.array([yi @ (M @ yi) for yi in y_hist])
    assert np.all(np.diff(energies) <= 1e-10), \
        f"energy not monotonically decreasing: {energies}"


def test_backward_zero_residual_returns_zero(disk_mesh, fem_matrices):
    """If y_s^d ≡ 0 then the adjoint z ≡ 0 (Eq. 4.1 with zero data + zero terminal)."""
    M, K_bg, M_v_bg = fem_matrices
    n = disk_mesh.n_points
    n_sub = 6
    meas_zero = np.zeros((n_sub, n))

    z_hist, ns = solve_backward_adjoint_segment(
        disk_mesh, M, K_bg, M_v_bg, meas_zero,
        n_substeps=n_sub, dt=0.0167,
    )
    assert np.allclose(z_hist, 0.0, atol=1e-12)
    assert ns == 0.0


def test_backward_terminal_condition_is_zero(disk_mesh, fem_matrices):
    """z(·, t_{n+1}) = 0 — the final entry of the adjoint history is zero."""
    M, K_bg, M_v_bg = fem_matrices
    n = disk_mesh.n_points
    n_sub = 6
    rng = np.random.default_rng(0)
    meas = rng.standard_normal((n_sub, n)) * 0.1

    z_hist, _ = solve_backward_adjoint_segment(
        disk_mesh, M, K_bg, M_v_bg, meas,
        n_substeps=n_sub, dt=0.0167,
    )
    assert np.allclose(z_hist[-1], 0.0, atol=1e-12)
    # but earlier entries should be nonzero in general
    assert np.linalg.norm(z_hist[0]) > 0.0


def test_backward_normal_scale_inverse_l2(disk_mesh, fem_matrices):
    """normal_scale = 1 / Σ_j ‖meas_j‖²_{L²(Γ)}."""
    M, K_bg, M_v_bg = fem_matrices
    n = disk_mesh.n_points
    M_bdry = assemble_boundary_mass_matrix(disk_mesh)

    n_sub = 4
    meas = np.ones((n_sub, n))
    _, ns = solve_backward_adjoint_segment(
        disk_mesh, M, K_bg, M_v_bg, meas,
        n_substeps=n_sub, dt=0.0167,
    )
    expected = 1.0 / (n_sub * float(np.ones(n) @ (M_bdry @ np.ones(n))))
    assert ns == pytest.approx(expected, rel=1e-9)


def test_local_dual_shapes_and_zero_cases(disk_mesh):
    n = disk_mesh.n_points
    n_tri = disk_mesh.n_triangles

    # zero y, zero z → ζ = 0
    y_hist = np.zeros((7, n))
    z_hist = np.zeros((7, n))
    zc, zv = compute_local_dual(disk_mesh, y_hist, z_hist, normal_scale=1.0)
    assert zc.shape == (n_tri,)
    assert zv.shape == (n_tri,)
    assert np.all(zc == 0.0)
    assert np.all(zv == 0.0)


def test_local_dual_constant_y_zero_grad(disk_mesh):
    """Constant y ⇒ ∇y = 0 ⇒ ζ_c = 0 regardless of z."""
    n = disk_mesh.n_points
    y_hist = np.full((7, n), 3.0)
    rng = np.random.default_rng(1)
    z_hist = rng.standard_normal((7, n))
    zc, zv = compute_local_dual(disk_mesh, y_hist, z_hist, normal_scale=1.0)
    # roundoff: gradient of an exactly-constant P1 function is the cancellation of
    # three nonzero gradients (sum to zero up to ~1e-13 per triangle).
    assert np.max(np.abs(zc)) < 1e-9
    # ζ_v should generally be nonzero
    assert np.linalg.norm(zv) > 0.0


def test_initialize_r0_parabolic(disk_mesh):
    diag = initialize_r0_parabolic(disk_mesh, exponent=0.7, cutoff=0.01)
    n_tri = disk_mesh.n_triangles
    assert diag.shape == (2 * n_tri,)
    # symmetry between blocks
    assert np.allclose(diag[:n_tri], diag[n_tri:])
    # values are non-negative
    assert np.all(diag >= 0)
    # near-boundary triangles are zeroed
    centroids = disk_mesh.centroids
    r = np.sqrt(np.sum(centroids ** 2, axis=1))
    near_bdry = (1.0 - r) ** 2 < 0.01
    assert np.all(diag[:n_tri][near_bdry] == 0.0)


def test_apply_inclusion_projection_conductivity(disk_mesh):
    n_tri = disk_mesh.n_triangles
    eta = np.zeros(2 * n_tri)
    # set a strong negative gradient on conductivity block
    eta[:n_tri] = -1.0
    eta[n_tri:] = 5.0
    sigma, v_pot = apply_inclusion_projection(
        eta, n_tri, model='conductivity',
        cA=1.0, cB=0.1, vA=1e-10, vB=2e-10,
    )
    delta_c = 0.9
    expected = max(-1.0, -0.99 / delta_c) * delta_c + 1.0  # = -0.99 + 1.0 = 0.01
    assert np.allclose(sigma, expected, atol=1e-10)
    assert np.all(v_pot == 1e-10)  # potential frozen for conductivity model


def test_apply_inclusion_projection_potential(disk_mesh):
    n_tri = disk_mesh.n_triangles
    eta = np.zeros(2 * n_tri)
    eta[:n_tri] = -1.0
    eta[n_tri:] = 1.5
    sigma, v_pot = apply_inclusion_projection(
        eta, n_tri, model='potential',
        cA=1.0, cB=1.0 + 1e-10, vA=1e-10, vB=15.0,
    )
    assert np.all(sigma == 1.0)  # conductivity frozen
    delta_v = abs(1e-10 - 15.0)
    expected = 1.5 * delta_v + 1e-10
    assert np.allclose(v_pot, expected, atol=1e-6)


def test_damp_lowrank_state_scales_s_and_ry(disk_mesh):
    n_tri = disk_mesh.n_triangles
    R = LowRankPreconditioner(np.ones(2 * n_tri), method='BFG', max_store=5)
    s = np.ones(2 * n_tri)
    y = np.full(2 * n_tri, 2.0)
    ry = np.full(2 * n_tri, 0.5)
    R.update(s, y, ry)
    R.update(s * 0.7, y * 1.3, ry * 0.9)

    s_before = [arr.copy() for arr in R.s_store]
    y_before = [arr.copy() for arr in R.y_store]
    ry_before = [arr.copy() for arr in R.ry_store]

    damp_lowrank_state(R, forget_scale=0.7)
    for j in range(R.count):
        assert np.allclose(R.s_store[j], 0.7 * s_before[j])
        assert np.allclose(R.y_store[j], y_before[j])  # y unchanged
        assert np.allclose(R.ry_store[j], 0.7 * ry_before[j])


def test_lowrank_psd_preserved_after_damping(disk_mesh):
    """R after damping must remain symmetric PSD when applied to vectors."""
    n_tri = disk_mesh.n_triangles
    rng = np.random.default_rng(2)
    diag = np.abs(rng.standard_normal(2 * n_tri)) + 0.1
    R = LowRankPreconditioner(diag.copy(), method='BFG', max_store=3)
    for _ in range(3):
        s = rng.standard_normal(2 * n_tri)
        y = rng.standard_normal(2 * n_tri)
        # ensure update keeps PSD: pick ry = R y (current operator)
        ry = R.apply(y)
        if np.dot(s, y) > 0:
            R.update(s, y, ry)

    damp_lowrank_state(R, forget_scale=0.7)
    # quadratic form xᵀ R x ≥ 0 on a few random probes
    for _ in range(8):
        x = rng.standard_normal(2 * n_tri)
        q = float(x @ R.apply(x))
        assert q >= -1e-9, f"R lost PSD after damping: x'Rx = {q}"


# ============================================================
# Trajectories & high-level smoke test
# ============================================================

def test_inclusion_trajectory_example1():
    from src.idsm_parabolic import inclusion_trajectory_example1
    # at t=0 the two active inclusions are at (0, ±0.6)
    c0, r0, a0 = inclusion_trajectory_example1(0.0)
    assert a0[0] and a0[1]
    assert not a0[2] and not a0[3]
    assert np.allclose(c0[0, 1], 0.6)
    assert np.allclose(c0[1, 1], -0.6)
    # at t=3 they collide
    c3, r3, a3 = inclusion_trajectory_example1(3.0)
    assert np.allclose(c3[0], c3[1], atol=1e-9)


def test_inclusion_trajectory_example4():
    from src.idsm_parabolic import inclusion_trajectory_example4
    c0, r0, a0, p0 = inclusion_trajectory_example4(0.0)
    assert not a0[0] and not a0[1]
    assert a0[2] and a0[3]
    # at t=0 traj[2] starts at full vB, traj[3] at vA
    assert np.isclose(p0[2], 15.0, atol=1e-6)
    assert np.isclose(p0[3], 1e-10, atol=1e-6)
    c6, r6, a6, p6 = inclusion_trajectory_example4(6.0)
    # at t=6 traj[2] faded to vA, traj[3] grew to vB
    assert np.isclose(p6[2], 1e-10, atol=1e-6)
    assert np.isclose(p6[3], 15.0, atol=1e-6)


def test_circle_indicator_p0(disk_mesh):
    from src.idsm_parabolic import circle_indicator_p0
    centers = np.array([[0.3, 0.0], [-0.3, 0.0], [2.0, 2.0], [2.0, 2.0]])
    radii = np.array([[0.15, 0.15], [0.15, 0.15], [1e-10, 1e-10], [1e-10, 1e-10]])
    active = np.array([True, True, False, False])
    ind = circle_indicator_p0(disk_mesh, centers, radii, active)
    assert ind.dtype == bool
    assert ind.sum() > 0
    # ghost circles do not contribute
    centers2 = centers.copy(); active2 = np.array([False, False, False, False])
    ind2 = circle_indicator_p0(disk_mesh, centers2, radii, active2)
    assert ind2.sum() == 0


def test_interpolate_boundary_data():
    from src.idsm_parabolic import interpolate_boundary_data
    # 5 fine snapshots at t = 0, 0.1, 0.2, 0.3, 0.4
    y_fine = np.array([np.full(3, k * 1.0) for k in range(5)])
    forward_dt = 0.1
    # at exact grid time -> exact value
    assert np.allclose(interpolate_boundary_data(y_fine, 0.2, forward_dt), 2.0)
    # at midpoint -> linear interp
    assert np.allclose(interpolate_boundary_data(y_fine, 0.15, forward_dt), 1.5)
    # outside grid is clamped
    assert np.allclose(interpolate_boundary_data(y_fine, 0.6, forward_dt), 4.0)


def test_run_idsm_parabolic_end_to_end_smoke(disk_mesh, fem_matrices):
    """Full driver runs on a 4-segment problem with a moving conductive inclusion."""
    from src.idsm_parabolic import (
        run_idsm_parabolic, inclusion_trajectory_example1,
        circle_indicator_p0, solve_forward_parabolic_segment,
    )
    from src.fem_skfem import assemble_mass_matrix, assemble_stiffness_matrix

    M = assemble_mass_matrix(disk_mesh)
    cfg = ParabolicConfig(
        total_time=0.8, delta_t=0.2, delta_t_split=3,
        forget_scale=0.7, tolerance=0.04, save_num=5, max_inner=6,
        cA=1.0, cB=0.05, vA=1e-10, vB=2e-10, model='conductivity',
    )

    def f_at(t):
        return lambda x, y: np.sin(t * np.pi / 4) * 25 * np.sin(3 * x) * np.cos(4 * y)

    def g_at(t):
        def gn(x, y):
            r = np.sqrt(x * x + y * y + 1e-30)
            nx = x / r; ny = y / r
            return np.cos(t * np.pi / 6) * (
                3 * np.cos(3 * x) * np.cos(4 * y) * nx
                - 4 * np.sin(3 * x) * np.sin(4 * y) * ny
            )
        return gn

    init = 3.0 + np.sin(3 * disk_mesh.points[:, 0]) * np.cos(4 * disk_mesh.points[:, 1])

    # Build synthetic data with the true inclusion present
    forward_dt = 0.1
    n_fine = int(np.ceil(cfg.total_time / forward_dt)) + 1
    y_data = np.zeros((n_fine, disk_mesh.n_points))
    y_data[0] = init.copy()
    y_curr = init.copy()
    for k in range(n_fine - 1):
        t = k * forward_dt
        c, r, a = inclusion_trajectory_example1(t + forward_dt / 2)
        inc = circle_indicator_p0(disk_mesh, c, r, a)
        sigma_t = np.where(inc, cfg.cB, cfg.cA)
        K = assemble_stiffness_matrix(disk_mesh, sigma_t)
        Mv = assemble_mass_matrix(disk_mesh, np.full(disk_mesh.n_triangles, cfg.vA))
        yh = solve_forward_parabolic_segment(
            disk_mesh, M, K, Mv, sigma_t, np.full(disk_mesh.n_triangles, cfg.vA),
            y_curr, t, t + forward_dt, n_substeps=2, f_func=f_at, g_func=g_at,
        )
        y_curr = yh[-1]
        y_data[k + 1] = y_curr.copy()
    rng = np.random.default_rng(0)
    y_data += 0.02 * np.abs(y_data) * (2 * rng.random(y_data.shape) - 1)

    out = run_idsm_parabolic(
        disk_mesh, y_data, forward_dt, f_at, g_at, init, cfg,
        truth_traj_func=inclusion_trajectory_example1, verbose=False,
    )
    assert out['sigma_history'].shape == (cfg.n_segments, disk_mesh.n_triangles)
    assert out['v_pot_history'].shape == (cfg.n_segments, disk_mesh.n_triangles)
    assert out['iou_history'] is not None
    # at least one segment must localize an inclusion above chance
    assert np.max(out['iou_history']) > 0.10, (
        f"max IoU too low: {out['iou_history']}"
    )
    # σ within projection bounds: cA - 0.99 ≤ σ ≤ cA (parabolic_*.edp L416)
    sigma_lower = cfg.cA - 0.99
    assert np.all(out['sigma_history'] <= cfg.cA + 1e-9)
    assert np.all(out['sigma_history'] >= sigma_lower - 1e-9)


def test_inclusion_trajectory_example2():
    from src.idsm_parabolic import inclusion_trajectory_example2
    c, r, a, k = inclusion_trajectory_example2(0.0)
    # all radii fixed at 0.2 except ghost
    assert np.allclose(r[:3], 0.2)
    assert np.allclose(r[3], 1e-10)
    # 3 active inclusions; traj 3 is ghost
    assert a.tolist() == [True, True, True, False]
    # kind tags: traj 0,1 conductivity, traj 2 potential, traj 3 ghost
    assert k.tolist() == ['c', 'c', 'v', '-']
    # cross-check first inclusion against .edp formula 0.65*cos(-7π/6)
    assert np.isclose(c[0, 0], 0.65 * np.cos(-7 * np.pi / 6), atol=1e-9)
    assert np.isclose(c[0, 1], 0.65 * np.sin(-7 * np.pi / 6), atol=1e-9)


def test_inclusion_trajectory_example3():
    from src.idsm_parabolic import inclusion_trajectory_example3
    c, r, a, u = inclusion_trajectory_example3(0.0)
    assert c.shape == (1, 2)
    assert np.allclose(r, 0.2)
    assert a.tolist() == [True]
    assert u == 20.0
    # cross-check (0.5*cos(π/4), 0.7*sin(π/4))
    assert np.isclose(c[0, 0], 0.5 * np.cos(np.pi / 4), atol=1e-9)
    assert np.isclose(c[0, 1], 0.7 * np.sin(np.pi / 4), atol=1e-9)


def test_inclusion_trajectory_example5():
    from src.idsm_parabolic import inclusion_trajectory_example5
    c0, r0, a0 = inclusion_trajectory_example5(0.0)
    assert a0.tolist() == [True, True, False, False]
    assert np.isclose(r0[1, 0], 0.3, atol=1e-9)  # max(0.3 - 0, 1e-10) = 0.3
    # by t=10 the second inclusion has effectively vanished (radius < 1e-3)
    c10, r10, a10 = inclusion_trajectory_example5(10.0)
    assert a10[1] is np.False_ or not bool(a10[1])
    assert r10[1, 0] < 1e-3


def test_nonlinear_forward_smoke(disk_mesh, fem_matrices):
    """Newton inner loop converges on a tiny segment with N(y)u = u·y·|y|."""
    from src.idsm_parabolic import solve_forward_parabolic_nonlinear_segment
    M, K_bg, _ = fem_matrices
    n = disk_mesh.n_points
    n_tri = disk_mesh.n_triangles

    init = 3.0 + np.sin(disk_mesh.points[:, 0])
    sigma = np.ones(n_tri)        # cA = 1
    u_p0 = np.full(n_tri, 1e-10)  # background uA → near-linear regime

    f_zero = lambda t: (lambda x, y: np.zeros_like(x))
    g_zero = lambda t: (lambda x, y: np.zeros_like(x))

    yh = solve_forward_parabolic_nonlinear_segment(
        disk_mesh, M, K_bg, sigma, u_p0, init,
        t_begin=0.0, t_end=0.1, n_substeps=2,
        f_func=f_zero, g_func=g_zero,
        newton_tol=1e-8, newton_max_iter=20,
    )
    assert yh.shape == (3, n)
    # smooth bounded field: max value should not blow up
    assert np.all(np.isfinite(yh))
    assert np.abs(yh).max() < 1e3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
