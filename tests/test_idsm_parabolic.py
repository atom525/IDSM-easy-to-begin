"""
Regression tests for the parabolic IDSM implementation.

These tests target the current ``src.idsm_parabolic`` API and the FreeFEM
conventions used by ``reference/parabolic_*.edp``.  They deliberately use tiny
meshes and short horizons; paper-scale runs belong in the notebooks/scripts.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mesh import generate_disk_mesh
from src.idsm import LowRankPreconditioner
from src.idsm_parabolic import (
    ParabolicConfig,
    assemble_const_operators,
    boundary_data_at,
    c_func_example_5_1,
    compute_zeta_p0,
    edp_cfg_example_5_1,
    init_diag_func,
    project_p1_fine_to_coarse,
    radius_example_5_1,
    run_idsm_parabolic,
    solve_adjoint_segment,
    solve_empty_segment,
    solve_forward_segment,
    trajectory_example_5_1,
    v_func_example_5_1,
)


@pytest.fixture(scope="module")
def small_mesh():
    return generate_disk_mesh(n_boundary=24)


def test_config_segment_count_matches_freefem_loop():
    cfg = ParabolicConfig(total_time=10.21, delta_t=0.1)
    assert int(np.floor(cfg.total_time / cfg.delta_t)) == 102
    # FreeFEM loops tIndex < inverseTimeNum - 1.
    assert cfg.n_segments == 101


def test_example_5_1_geometry_matches_reference():
    c0 = trajectory_example_5_1(0.0, 0)
    c1 = trajectory_example_5_1(0.0, 1)
    assert np.allclose(c0, [0.0, 0.6])
    assert np.allclose(c1, [0.0, -0.6])
    assert np.allclose(trajectory_example_5_1(3.0, 0), trajectory_example_5_1(3.0, 1))
    assert np.allclose(radius_example_5_1(0), [0.2, 0.2])
    assert np.allclose(radius_example_5_1(2), [1e-10, 1e-10])


def test_boundary_data_at_interpolates_linearly():
    y_data = np.zeros((5, 1, 3))
    for i in range(5):
        y_data[i, 0] = i
    assert np.allclose(boundary_data_at(0.2, 0, y_data, 0.1), 2.0)
    assert np.allclose(boundary_data_at(0.15, 0, y_data, 0.1), 1.5)
    assert np.allclose(boundary_data_at(0.6, 0, y_data, 0.1), 4.0)


def test_adjoint_normal_scale_uses_residual_energy(small_mesh):
    cfg = ParabolicConfig(delta_t=0.1, delta_t_split=3)
    ops = assemble_const_operators(small_mesh, cfg)
    n_pts = small_mesh.n_points
    residual = np.ones((cfg.delta_t_split, n_pts))
    measurement = np.full_like(residual, 10.0)

    _, normal_scale = solve_adjoint_segment(small_mesh, ops, residual, measurement, cfg)
    expected = 1.0 / sum(float(r @ (ops.M_bdry @ r)) for r in residual)
    assert normal_scale == pytest.approx(expected)


def test_empty_forward_and_local_dual_shapes(small_mesh):
    cfg = ParabolicConfig(delta_t=0.1, delta_t_split=2)
    ops = assemble_const_operators(small_mesh, cfg)
    y0 = np.ones(small_mesh.n_points)
    y_empty = solve_empty_segment(small_mesh, ops, y0, t_begin=0.0, cfg=cfg, data_index=0)
    assert y_empty.shape == (cfg.delta_t_split + 1, small_mesh.n_points)

    zeta_c, zeta_v = compute_zeta_p0(
        small_mesh,
        y_curr=y_empty[-1],
        y_last=y_empty[0],
        y_dual=np.ones(small_mesh.n_points),
        normal_scale=1.0,
    )
    assert zeta_c.shape == (small_mesh.n_triangles,)
    assert zeta_v.shape == (small_mesh.n_triangles,)


def test_lowrank_damping_matches_freefem_pair_scaling(small_mesh):
    n_tri = small_mesh.n_triangles
    R = LowRankPreconditioner(np.ones(2 * n_tri), method='BFG', max_store=3)
    s = np.ones(2 * n_tri)
    y = np.full(2 * n_tri, 2.0)
    ry = R.apply(y)
    R.update(s, y, ry)

    s_before = R.s_store[0].copy()
    y_before = R.y_store[0].copy()
    ry_before = R.ry_store[0].copy()
    forget_scale = 0.7
    R.s_store[0] *= forget_scale
    R.ry_store[0] *= forget_scale

    assert np.allclose(R.s_store[0], forget_scale * s_before)
    assert np.allclose(R.y_store[0], y_before)
    assert np.allclose(R.ry_store[0], forget_scale * ry_before)


def test_projection_fine_to_coarse_preserves_constant(small_mesh):
    fine = generate_disk_mesh(n_boundary=32)
    coarse = small_mesh
    const = np.full(fine.n_points, 3.25)
    projected = project_p1_fine_to_coarse(fine, coarse, const)
    assert np.allclose(projected, 3.25)


def test_run_idsm_parabolic_current_api_smoke(small_mesh):
    fine = generate_disk_mesh(n_boundary=32)
    coarse = small_mesh
    cfg = edp_cfg_example_5_1(noise=0.0)
    cfg.total_time = 0.31
    cfg.delta_t = 0.1
    cfg.delta_t_split = 2
    cfg.forward_dt = 0.05
    cfg.max_inner = 2
    cfg.tolerance = 10.0

    out = run_idsm_parabolic(
        coarse_mesh=coarse,
        fine_mesh=fine,
        cfg=cfg,
        c_func=c_func_example_5_1,
        v_func=v_func_example_5_1,
        seed=0,
        verbose=False,
    )
    assert len(out['sigma_history']) == cfg.n_segments
    assert len(out['v_history']) == cfg.n_segments
    assert len(out['iou_history']) == cfg.n_segments


def test_run_idsm_parabolic_three_mesh_smoke():
    data = generate_disk_mesh(n_boundary=32)
    solve = generate_disk_mesh(n_boundary=28)
    coeff = generate_disk_mesh(n_boundary=20)
    cfg = edp_cfg_example_5_1(noise=0.0)
    cfg.total_time = 0.31
    cfg.delta_t = 0.1
    cfg.delta_t_split = 2
    cfg.forward_dt = 0.05
    cfg.max_inner = 2
    cfg.tolerance = 10.0

    out = run_idsm_parabolic(
        coarse_mesh=coeff,
        fine_mesh=data,
        solve_mesh=solve,
        cfg=cfg,
        c_func=c_func_example_5_1,
        v_func=v_func_example_5_1,
        seed=0,
        verbose=False,
    )
    assert out['sigma_history'][0].shape == (coeff.n_triangles,)
    assert out['v_history'][0].shape == (coeff.n_triangles,)
    assert out['y_data'].shape[2] == solve.n_points


def test_forward_segment_projection_bounds(small_mesh):
    cfg = ParabolicConfig(delta_t=0.1, delta_t_split=2)
    ops = assemble_const_operators(small_mesh, cfg)
    n_tri = small_mesh.n_triangles
    y0 = np.ones(small_mesh.n_points)
    sigma = np.full(n_tri, cfg.cA)
    v_pot = np.full(n_tri, cfg.vA)
    hist = solve_forward_segment(
        small_mesh,
        ops,
        sigma_prev=sigma,
        sigma_curr=sigma,
        v_prev=v_pot,
        v_curr=v_pot,
        y_last=y0,
        t_begin=0.0,
        cfg=cfg,
        data_index=0,
        is_first_segment=True,
    )
    assert hist.shape == (cfg.delta_t_split + 1, small_mesh.n_points)
    assert np.all(np.isfinite(hist))
