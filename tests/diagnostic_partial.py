"""诊断脚本：验证 idsm_partial.py Bug #2-#4 修复效果。

运行 partial IDSM 少量迭代，打印所有关键中间量的统计信息，
用于快速判断修复是否成功。

用法：
    python -m tests.diagnostic_partial
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mesh import generate_elliptic_mesh
from src.forward_solver import (
    make_conductivity_example1,
    generate_cauchy_data,
)
from src.idsm_partial import (
    define_accessible_boundary,
    run_idsm_partial,
    compute_heterogeneous_D,
)
from src.idsm import run_idsm


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_stats(name, arr):
    arr = np.asarray(arr)
    print(f"  {name:30s}: min={arr.min():.6e}  max={arr.max():.6e}  "
          f"mean={arr.mean():.6e}  std={arr.std():.6e}")


def main():
    np.random.seed(42)

    # 使用较粗网格加速诊断
    print_section("生成网格和数据")
    mesh = generate_elliptic_mesh(n_boundary=200)
    print(f"  网格: {mesh.n_points} 节点, {mesh.n_triangles} 三角形, "
          f"{len(mesh.boundary_nodes)} 边界节点")

    sigma_true, u_true = make_conductivity_example1(mesh)
    sources = [lambda x, y: x, lambda x, y: y]
    cauchy = generate_cauchy_data(mesh, sigma_true, sources, noise_level=0.1)

    # --------------------------------------------------------
    # 1. Full data IDSM 基线（应该正常）
    # --------------------------------------------------------
    print_section("Full data IDSM 基线 (5 iter)")
    hist_full = run_idsm(
        mesh, cauchy,
        sigma_bg=1.0, sigma_range=0.01,
        alpha=1.0, n_iter=5, lowrank_method='BFG',
        problem_type='conductivity', coeff_known=False,
        verbose=True,
    )
    r_full = hist_full['residuals']
    print(f"\n  残差: {r_full[0]:.6e} → {r_full[-1]:.6e}")
    print(f"  比率: {r_full[-1]/r_full[0]:.4f}")
    if r_full[-1] < r_full[0]:
        print("  ✓ Full data 残差下降正常")
    else:
        print("  ✗ Full data 残差未下降！基线有问题")

    # --------------------------------------------------------
    # 2. Heterogeneous D 诊断
    # --------------------------------------------------------
    print_section("Heterogeneous D 诊断")
    theta_range = (-np.pi/2, np.pi/2)  # 右半边界
    gamma_d = define_accessible_boundary(mesh, theta_range)
    D = compute_heterogeneous_D(
        mesh,
        gamma_d_node_mask=gamma_d['node_mask'],
        alpha_d=0.05, alpha_n=2.0,
        gamma=4.0, epsilon=0.02,
    )
    print_stats("D (全域)", D)
    nonzero = D > 0
    if nonzero.any():
        print_stats("D (非零)", D[nonzero])
    print(f"  D=0 元素数: {(~nonzero).sum()}/{len(D)} (靠近边界的截断)")

    # 检查 D 是否在 Γ_D 附近更强（alpha_d 小 → 惩罚更强 → D 更大）
    centroids = mesh.centroids
    # 按距右边界的距离分组
    right_mask = centroids[:, 0] > 0.3
    left_mask = centroids[:, 0] < -0.3
    if nonzero.any():
        d_right = D[right_mask & nonzero]
        d_left = D[left_mask & nonzero]
        if len(d_right) > 0 and len(d_left) > 0:
            print(f"\n  D 空间异质性检查:")
            print(f"    右侧 (靠近 Γ_D, α_d=0.05): mean={d_right.mean():.6e}")
            print(f"    左侧 (靠近 Γ_N, α_n=2.0):  mean={d_left.mean():.6e}")
            if d_right.mean() > d_left.mean():
                print("    ✓ D 在可测边界附近更大（正确的空间异质性）")
            else:
                print("    ⚠ D 空间异质性方向可能有误")

    # --------------------------------------------------------
    # 3. Partial IDSM 主诊断
    # --------------------------------------------------------
    n_diag_iter = 8
    print_section(f"Partial IDSM 诊断 ({n_diag_iter} iter, right half)")

    hist_partial = run_idsm_partial(
        mesh, cauchy, gamma_d,
        sigma_bg=1.0, sigma_range=0.01,
        alpha_d=0.05, alpha_n=2.0,
        n_iter=n_diag_iter,
        lowrank_method='BFG',
        problem_type='conductivity',
        coeff_known=False,
        gamma_D=4.0, epsilon_cutoff=0.02, p_norm=2.0,
        verbose=True,
    )

    r_partial = hist_partial['residuals']
    print(f"\n  残差序列: {[f'{r:.4e}' for r in r_partial]}")
    print(f"  初始: {r_partial[0]:.6e}, 最终: {r_partial[-1]:.6e}")
    print(f"  比率: {r_partial[-1]/r_partial[0]:.4f}")

    if r_partial[-1] < r_partial[0]:
        print("  ✓ Partial data 残差下降")
    else:
        print("  ✗ Partial data 残差未下降！需要继续调查")

    # σ 统计
    print(f"\n  σ 统计:")
    for i, sg in enumerate(hist_partial['sigma_guess']):
        deviated = np.sum(np.abs(sg - 1.0) > 0.01)
        print(f"    Iter {i}: σ ∈ [{sg.min():.4f}, {sg.max():.4f}], "
              f"|σ-1|>0.01 元素: {deviated}/{len(sg)}")

    # λ 阻尼因子
    lam = hist_partial.get('lambda_history', [])
    if lam:
        print(f"\n  λ 阻尼因子: {[f'{l:.4e}' for l in lam]}")
        print(f"    范围: [{min(lam):.4e}, {max(lam):.4e}]")
        if len(lam) >= 3:
            # 检查是否有 U 形趋势
            mid = len(lam) // 2
            early = np.mean(lam[:max(1, mid//2)])
            middle = np.mean(lam[mid//2:mid+mid//2])
            late = np.mean(lam[-max(1, mid//2):])
            print(f"    早期均值: {early:.4e}, 中期: {middle:.4e}, 晚期: {late:.4e}")

    # --------------------------------------------------------
    # 4. 多配置对比
    # --------------------------------------------------------
    print_section("多边界配置对比 (5 iter)")
    configs = [
        ('Right half', (-np.pi/2, np.pi/2)),
        ('Upper half', (0, np.pi)),
        ('3/4 boundary', (-np.pi/4, 5*np.pi/4)),
    ]
    for name, theta_range in configs:
        gd = define_accessible_boundary(mesh, theta_range)
        h = run_idsm_partial(
            mesh, cauchy, gd,
            sigma_bg=1.0, sigma_range=0.01,
            alpha_d=0.05, alpha_n=2.0,
            n_iter=5, lowrank_method='BFG',
            problem_type='conductivity', coeff_known=False,
            verbose=False,
        )
        r = h['residuals']
        sf = h['sigma_guess'][-1]
        decreased = "✓" if r[-1] < r[0] else "✗"
        print(f"  {name:20s}: resid {r[0]:.4e} → {r[-1]:.4e} "
              f"(ratio={r[-1]/r[0]:.4f}) {decreased}  "
              f"σ∈[{sf.min():.4f}, {sf.max():.4f}]")

    print_section("诊断完成")


if __name__ == '__main__':
    main()
