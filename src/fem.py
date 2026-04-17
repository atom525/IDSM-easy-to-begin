"""
fem.py — P1 有限元公共接口（委托给 scikit-fem 后端）

所有下游模块（forward_solver, dsm, idsm, idsm_partial）通过此文件导入 FEM 函数。
默认使用 scikit-fem 实现（fem_skfem.py），可通过环境变量切换回手写版本：

    IDSM_FEM_LEGACY=1 python -m pytest tests/

公共 API（签名不变）：
  - assemble_stiffness_matrix(mesh, sigma)
  - assemble_mass_matrix(mesh, coeff=None)
  - assemble_boundary_mass_matrix(mesh)
  - assemble_boundary_load(mesh, f_func)
  - assemble_boundary_mean_constraint(mesh)
  - solve_neumann_system(K, b, B)
  - solve_robin_system(mesh, A_op, alpha, v)
  - compute_boundary_normal_flux(mesh, sigma, y)
  - assemble_partial_boundary_mass_matrix(mesh, gamma_d_node_mask)
  - compute_boundary_normal_derivative(mesh, z, sigma_bg=1.0)  [新增]
"""

import os

_USE_LEGACY = os.getenv("IDSM_FEM_LEGACY", "0").strip().lower() in ("1", "true")

if _USE_LEGACY:
    from .fem_legacy import *  # noqa: F401, F403
else:
    from .fem_skfem import *  # noqa: F401, F403
