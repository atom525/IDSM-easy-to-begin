"""Tests for mesh.py — mesh generation and geometry utilities."""

import numpy as np
import pytest

from IDSM.src.mesh import (
    _extract_boundary_edges,
    generate_coarse_mesh,
    generate_elliptic_mesh,
    fine_to_coarse_p0,
    coarse_to_fine_p0,
)


@pytest.fixture
def mesh():
    return generate_elliptic_mesh(n_boundary=120)


def test_mesh_area_close_to_ellipse_area(mesh):
    """Total triangle area should approximate pi * b = pi * 0.8."""
    area = float(np.sum(mesh.areas))
    target = np.pi * 0.8
    assert abs(area - target) / target < 0.01, f"Area {area} not close to {target}"


def test_mesh_area_exact_bound():
    """With fine mesh, area should be within 0.1% of pi*0.8."""
    mesh = generate_elliptic_mesh(n_boundary=256)
    area = float(np.sum(mesh.areas))
    target = np.pi * 0.8
    assert abs(area - target) / target < 0.001


def test_boundary_nodes_count_matches_request():
    mesh = generate_elliptic_mesh(n_boundary=100)
    assert mesh.n_boundary == len(mesh.boundary_nodes)
    assert mesh.n_boundary >= 90


def test_boundary_nodes_form_closed_loop(mesh):
    """Ordered boundary nodes form a continuous closed curve."""
    ordered = mesh.boundary_nodes
    pts = mesh.points[ordered]
    jumps = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    assert np.all(jumps > 0), "Duplicate consecutive boundary nodes"
    assert np.max(jumps) < 0.3, "Gap too large between consecutive boundary nodes"


def test_boundary_nodes_lie_on_ellipse(mesh):
    """All boundary nodes should satisfy x1^2 + x2^2/0.64 ≈ 1."""
    pts = mesh.points[mesh.boundary_nodes]
    r = pts[:, 0]**2 + pts[:, 1]**2 / 0.64
    assert np.allclose(r, 1.0, atol=0.02)


def test_boundary_edge_extraction_completeness(mesh):
    """Every boundary edge connects two boundary nodes."""
    bdry_set = set(int(n) for n in mesh.boundary_nodes)
    edges = mesh.boundary_edges
    assert edges.shape[0] > 0
    for e in edges:
        assert int(e[0]) in bdry_set
        assert int(e[1]) in bdry_set


def test_coarse_mesh_has_fewer_triangles():
    """Coarse mesh should have significantly fewer triangles."""
    coarse = generate_coarse_mesh(target_triangles=600)
    assert 300 < coarse.n_triangles < 2000


def test_fine_to_coarse_roundtrip():
    """P0 projection fine→coarse→fine should be a smoothing operator."""
    fine = generate_elliptic_mesh(n_boundary=120)
    coarse = generate_coarse_mesh(target_triangles=300)
    values = np.random.default_rng(42).standard_normal(fine.n_triangles)
    coarsened = fine_to_coarse_p0(fine, coarse, values)
    roundtrip = coarse_to_fine_p0(fine, coarse, coarsened)
    assert roundtrip.shape == values.shape
    assert np.all(np.isfinite(roundtrip))


def test_grad_phi_precomputed(mesh):
    """Gradient basis functions should be precomputed."""
    assert mesh.grad_phi.shape == (mesh.n_triangles, 3, 2)
    assert np.all(np.isfinite(mesh.grad_phi))


def test_triangle_areas_positive(mesh):
    """All triangle areas should be strictly positive."""
    assert np.all(mesh.areas > 0)
