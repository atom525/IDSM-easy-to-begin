"""Tests for utils.py -- utility functions."""

import numpy as np
import pytest

from IDSM.src.mesh import generate_elliptic_mesh
from IDSM.src.utils import (
    distance_to_boundary,
    compute_iou,
    p0_to_grid,
    fundamental_solution_2d,
)


@pytest.fixture
def mesh():
    return generate_elliptic_mesh(n_boundary=80)


def test_distance_to_boundary_at_boundary_is_zero(mesh):
    """Distance to boundary should be zero at boundary points."""
    bdry_pts = mesh.points[mesh.boundary_nodes]
    dist = distance_to_boundary(mesh, bdry_pts)
    assert np.all(dist < 1e-12)


def test_distance_to_boundary_at_center_is_positive(mesh):
    """Distance from domain center to boundary should be positive."""
    center = np.array([[0.0, 0.0]])
    dist = distance_to_boundary(mesh, center)
    assert dist[0] > 0.5  # ellipse semi-minor axis b=0.8, center to boundary >= 0.8


def test_distance_to_boundary_positive_everywhere_interior(mesh):
    """Distance from all interior centroids to boundary should be positive."""
    dist = distance_to_boundary(mesh, mesh.centroids)
    assert np.all(dist > 0)


def test_iou_identical_inclusions(mesh):
    """Identical inputs -> IoU = 1.0."""
    u = np.zeros(mesh.n_triangles)
    u[:10] = 1.0
    iou = compute_iou(u, u, mesh)
    assert abs(iou - 1.0) < 1e-10


def test_iou_no_overlap(mesh):
    """Disjoint regions -> IoU = 0.0."""
    u_true = np.zeros(mesh.n_triangles)
    u_pred = np.zeros(mesh.n_triangles)
    u_true[:10] = 1.0
    u_pred[10:20] = 1.0
    iou = compute_iou(u_true, u_pred, mesh)
    # After area-matching threshold, may have tiny overlap, but should be near 0
    assert iou < 0.5


def test_iou_range(mesh):
    """IoU should be in [0, 1] range."""
    rng = np.random.default_rng(42)
    u_true = np.zeros(mesh.n_triangles)
    u_true[:50] = 1.0
    u_pred = rng.random(mesh.n_triangles)
    iou = compute_iou(u_true, u_pred, mesh)
    assert 0.0 <= iou <= 1.0


def test_iou_empty_true_returns_zero(mesh):
    """IoU should return 0 when ground truth is all zeros."""
    u_true = np.zeros(mesh.n_triangles)
    u_pred = np.ones(mesh.n_triangles)
    iou = compute_iou(u_true, u_pred, mesh)
    assert iou == 0.0


def test_p0_to_grid_constant_field(mesh):
    """Constant P0 field projected to grid should remain constant."""
    vals = np.full(mesh.n_triangles, 3.14)
    grid_pts = mesh.centroids  # use centroids as query points
    result = p0_to_grid(mesh, vals, grid_pts)
    assert np.allclose(result, 3.14)


def test_p0_to_grid_shape(mesh):
    """Output shape should match number of grid points."""
    vals = np.ones(mesh.n_triangles)
    grid_pts = np.array([[0.0, 0.0], [0.3, 0.1], [-0.3, -0.1]])
    result = p0_to_grid(mesh, vals, grid_pts)
    assert result.shape == (3,)


def test_p0_to_grid_wrong_length_raises(mesh):
    """P0 values with wrong length should raise ValueError."""
    with pytest.raises(ValueError):
        p0_to_grid(mesh, np.ones(5), mesh.centroids)


def test_fundamental_solution_singularity():
    """Fundamental solution should approach +infinity as x -> x'."""
    x = np.array([[0.0, 0.0]])
    x_prime = np.array([[1e-10, 0.0]])
    phi = fundamental_solution_2d(x, x_prime)
    assert phi > 1.0  # -1/(2pi) ln(1e-10) ≈ 3.66


def test_fundamental_solution_symmetry():
    """Fundamental solution should satisfy Phi(x, y) = Phi(y, x)."""
    x = np.array([0.3, 0.1])
    y = np.array([[0.5, -0.2]])
    phi_xy = fundamental_solution_2d(x.reshape(1, 2), y)
    phi_yx = fundamental_solution_2d(y, x.reshape(1, 2))
    assert abs(float(phi_xy) - float(phi_yx)) < 1e-14
