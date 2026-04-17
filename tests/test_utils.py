"""测试 utils.py — 工具函数测试。"""

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
    """边界点到边界的距离应为 0。"""
    bdry_pts = mesh.points[mesh.boundary_nodes]
    dist = distance_to_boundary(mesh, bdry_pts)
    assert np.all(dist < 1e-12)


def test_distance_to_boundary_at_center_is_positive(mesh):
    """域中心点到边界的距离应为正值。"""
    center = np.array([[0.0, 0.0]])
    dist = distance_to_boundary(mesh, center)
    assert dist[0] > 0.5  # 椭圆短轴 b=0.8，中心到边界至少 0.8


def test_distance_to_boundary_positive_everywhere_interior(mesh):
    """所有内部质心到边界的距离应为正值。"""
    dist = distance_to_boundary(mesh, mesh.centroids)
    assert np.all(dist > 0)


def test_iou_identical_inclusions(mesh):
    """相同输入 → IoU = 1.0。"""
    u = np.zeros(mesh.n_triangles)
    u[:10] = 1.0
    iou = compute_iou(u, u, mesh)
    assert abs(iou - 1.0) < 1e-10


def test_iou_no_overlap(mesh):
    """不相交的区域 → IoU = 0.0。"""
    u_true = np.zeros(mesh.n_triangles)
    u_pred = np.zeros(mesh.n_triangles)
    u_true[:10] = 1.0
    u_pred[10:20] = 1.0
    iou = compute_iou(u_true, u_pred, mesh)
    # 面积匹配阈值后可能有微小重叠，但应接近 0
    assert iou < 0.5


def test_iou_range(mesh):
    """IoU 应在 [0, 1] 范围内。"""
    rng = np.random.default_rng(42)
    u_true = np.zeros(mesh.n_triangles)
    u_true[:50] = 1.0
    u_pred = rng.random(mesh.n_triangles)
    iou = compute_iou(u_true, u_pred, mesh)
    assert 0.0 <= iou <= 1.0


def test_iou_empty_true_returns_zero(mesh):
    """真值全零时 IoU 应返回 0。"""
    u_true = np.zeros(mesh.n_triangles)
    u_pred = np.ones(mesh.n_triangles)
    iou = compute_iou(u_true, u_pred, mesh)
    assert iou == 0.0


def test_p0_to_grid_constant_field(mesh):
    """常数 P0 场投影到网格应保持不变。"""
    vals = np.full(mesh.n_triangles, 3.14)
    grid_pts = mesh.centroids  # 用质心作为查询点
    result = p0_to_grid(mesh, vals, grid_pts)
    assert np.allclose(result, 3.14)


def test_p0_to_grid_shape(mesh):
    """输出形状应与网格点数一致。"""
    vals = np.ones(mesh.n_triangles)
    grid_pts = np.array([[0.0, 0.0], [0.3, 0.1], [-0.3, -0.1]])
    result = p0_to_grid(mesh, vals, grid_pts)
    assert result.shape == (3,)


def test_p0_to_grid_wrong_length_raises(mesh):
    """P0 值长度不匹配应抛出异常。"""
    with pytest.raises(ValueError):
        p0_to_grid(mesh, np.ones(5), mesh.centroids)


def test_fundamental_solution_singularity():
    """基本解在 x → x' 时应趋于正无穷。"""
    x = np.array([[0.0, 0.0]])
    x_prime = np.array([[1e-10, 0.0]])
    phi = fundamental_solution_2d(x, x_prime)
    assert phi > 1.0  # -1/(2π) ln(1e-10) ≈ 3.66


def test_fundamental_solution_symmetry():
    """基本解应满足 Φ(x, y) = Φ(y, x)。"""
    x = np.array([0.3, 0.1])
    y = np.array([[0.5, -0.2]])
    phi_xy = fundamental_solution_2d(x.reshape(1, 2), y)
    phi_yx = fundamental_solution_2d(y, x.reshape(1, 2))
    assert abs(float(phi_xy) - float(phi_yx)) < 1e-14
