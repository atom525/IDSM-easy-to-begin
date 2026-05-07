"""论文 §5 Paper-Scale Parabolic 重构图与 Table 1 复现.

读取 results/parabolic/ex_5_*_{paper,edp}.npz, 输出:
- figures/parabolic/fig_5_X_{paper,edp}.png  : 10 帧 heatmap (Fig 1-5 论文风格)
- figures/parabolic/table1.txt               : PDE solve / IoU 汇总表

API 说明：
- ``trajectory_example_5_X(t, traj_index)`` 返回单条 inclusion 圆心 (cx, cy)
- ``radius_example_5_X(traj_index)`` 或 ``radius_example_5_5(t, traj_index)`` 返回半径 (lx, ly)
- traj_index 0/1 通常为 σ inclusion，2/3 为 V inclusion；半径 1e-10 视为屏蔽 ghost

Usage:
    python scripts/plot_parabolic_paper_figures.py
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.patches import Ellipse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.idsm_parabolic import (
    trajectory_example_5_1, radius_example_5_1,
    trajectory_example_5_2,
    trajectory_example_5_3,
    trajectory_example_5_4, radius_example_5_4,
    trajectory_example_5_5, radius_example_5_5,
)

# 椭圆描绘的最小半径阈值 — 小于此值视为 ghost (.edp 用 1e-10 屏蔽)
GHOST_R = 1e-6


def _radii_5_2(traj_index: int) -> np.ndarray:
    """Ex 5.2 简单欧氏距离 + 半径 0.2，traj_index=3 屏蔽."""
    if traj_index in (0, 1, 2):
        return np.array([0.2, 0.2])
    return np.array([1e-10, 1e-10])


def _radii_5_3(traj_index: int) -> np.ndarray:
    """Ex 5.3 唯一 inclusion 半径 0.2."""
    if traj_index == 0:
        return np.array([0.2, 0.2])
    return np.array([1e-10, 1e-10])


# 每个 example 的元信息
EX_INFO = {
    '5_1': dict(
        title='Example 5.1: Conductivity Merging',
        traj=trajectory_example_5_1, radius=lambda t, i: radius_example_5_1(i),
        sigma_traj_idx=(0, 1), v_traj_idx=(2, 3), model='cond',
    ),
    '5_1_n10': dict(
        title='Example 5.1 (noise=10%): Conductivity Merging',
        traj=trajectory_example_5_1, radius=lambda t, i: radius_example_5_1(i),
        sigma_traj_idx=(0, 1), v_traj_idx=(2, 3), model='cond',
    ),
    '5_2': dict(
        title='Example 5.2: Mixed Moving (σ + V)',
        traj=trajectory_example_5_2, radius=lambda t, i: _radii_5_2(i),
        sigma_traj_idx=(0, 1), v_traj_idx=(2, 3), model='double',
    ),
    '5_3': dict(
        title='Example 5.3: Nonlinear p=3 (linear surrogate)',
        traj=trajectory_example_5_3, radius=lambda t, i: _radii_5_3(i),
        sigma_traj_idx=(0,), v_traj_idx=(0,), model='nonlinear',
    ),
    '5_4': dict(
        title='Example 5.4: Potential Fading',
        traj=trajectory_example_5_4, radius=lambda t, i: radius_example_5_4(i),
        sigma_traj_idx=(0, 1), v_traj_idx=(2, 3), model='pot',
    ),
    '5_5': dict(
        title='Example 5.5: Conductivity Diminishing',
        traj=trajectory_example_5_5, radius=lambda t, i: radius_example_5_5(t, i),
        sigma_traj_idx=(0, 1), v_traj_idx=(2, 3), model='cond',
    ),
}


def load_npz(key: str, mode: str):
    path = ROOT / 'results' / 'parabolic' / f'ex_{key}_{mode}.npz'
    if not path.exists():
        return None
    return np.load(path, allow_pickle=False)


def _draw_inclusions(ax, info, t_now, idx_tuple, edge='red'):
    """在 ax 上画 inclusion 椭圆轮廓（半径 < GHOST_R 视为屏蔽）."""
    for ti in idx_tuple:
        cp = info['traj'](t_now, ti)
        r = info['radius'](t_now, ti)
        if max(float(r[0]), float(r[1])) < GHOST_R:
            continue
        e = Ellipse(
            (float(cp[0]), float(cp[1])),
            width=2.0 * float(r[0]), height=2.0 * float(r[1]),
            fill=False, edgecolor=edge, linewidth=1.5,
        )
        ax.add_patch(e)


def plot_example(key: str, mode: str = 'paper'):
    data = load_npz(key, mode)
    if data is None:
        print(f'  skip ex_{key}_{mode} (no npz)')
        return
    info = EX_INFO[key]
    pts = data['coarse_points']
    tris = data['coarse_triangles']
    sigma_h = data['sigma_history']
    v_pot_h = data['v_history']
    iou = data['iou_history']
    total_time = float(data['cfg_total_time'])
    cA = float(data['cfg_cA']); cB = float(data['cfg_cB'])
    vA = float(data['cfg_vA']); vB = float(data['cfg_vB'])
    n_seg = sigma_h.shape[0]

    n_frames = 10 if n_seg >= 10 else n_seg
    idx_frames = np.linspace(0, n_seg - 1, n_frames).astype(int)

    triang = Triangulation(pts[:, 0], pts[:, 1], tris)

    rows = 2 if info['model'] == 'double' else 1
    fig, axes = plt.subplots(
        rows, n_frames,
        figsize=(2.0 * n_frames, 2.2 * rows + 0.3),
        squeeze=False,
    )

    for col, seg_idx in enumerate(idx_frames):
        t_now = (seg_idx + 1) * total_time / n_seg
        sigma = sigma_h[seg_idx]
        v_pot = v_pot_h[seg_idx]

        if info['model'] == 'double':
            ax = axes[0, col]
            ax.tripcolor(triang, facecolors=sigma,
                         shading='flat', cmap='viridis',
                         vmin=cB, vmax=cA)
            _draw_inclusions(ax, info, t_now, info['sigma_traj_idx'], edge='red')
            ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f't={t_now:.2f}', fontsize=9)
            if col == 0:
                ax.set_ylabel('σ', fontsize=10)
            ax = axes[1, col]
            vmax_v = max(vB, vA + 1e-6)
            ax.tripcolor(triang, facecolors=v_pot,
                         shading='flat', cmap='magma',
                         vmin=vA, vmax=vmax_v)
            _draw_inclusions(ax, info, t_now, info['v_traj_idx'], edge='cyan')
            ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel('V', fontsize=10)
        else:
            ax = axes[0, col]
            if info['model'] in ('cond', 'nonlinear'):
                ax.tripcolor(triang, facecolors=sigma,
                             shading='flat', cmap='viridis',
                             vmin=cB, vmax=cA)
                if col == 0:
                    ax.set_ylabel('σ', fontsize=10)
                _draw_inclusions(ax, info, t_now, info['sigma_traj_idx'], edge='red')
            else:  # pot
                vmax_v = max(vB, vA + 1e-6)
                ax.tripcolor(triang, facecolors=v_pot,
                             shading='flat', cmap='magma',
                             vmin=vA, vmax=vmax_v)
                if col == 0:
                    ax.set_ylabel('V', fontsize=10)
                _draw_inclusions(ax, info, t_now, info['v_traj_idx'], edge='cyan')
            ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f't={t_now:.2f}', fontsize=9)

    iou_str = ''
    if iou.size > 0:
        iou_str = f"  IoU mean={iou.mean():.3f} max={iou.max():.3f}"
    fig.suptitle(f"{info['title']}{iou_str}  [cfg={mode}]", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir = ROOT / 'figures' / 'parabolic'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'fig_{key}_{mode}.png'
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'  saved {out_path}')


def make_table1():
    """Table 1: 各 example 平均每段 PDE solve 计数 / IoU 汇总.

    PDE solves per segment ≈ inner_iter * (forward + adjoint per iter)
                           = mean_inner * 2 + 3 (背景 forward + backward + transition).
    """
    lines = [
        '# Table 1 — PDE Solve Counts per Segment / IoU 汇总',
        '# 单元格：mean_inner = 平均内迭代步数；PDE/seg = 每段 PDE solve 估算',
        '',
        f"{'Example':<10s} {'mode':<6s} {'inner_mean':>10s} "
        f"{'PDE/seg':>8s} {'n_seg':>6s} {'IoU_mean':>9s} "
        f"{'IoU_max':>8s} {'runtime':>8s}",
    ]
    for key in ['5_1', '5_1_n10', '5_2', '5_3', '5_4', '5_5']:
        for mode in ['paper', 'edp']:
            data = load_npz(key, mode)
            if data is None:
                continue
            n_inner = data['n_inner_per_segment']
            iou = data['iou_history']
            mean_inner = float(n_inner.mean())
            pde_per_seg = mean_inner * 2 + 3
            n_seg = int(n_inner.size)
            iou_mean = float(iou.mean()) if iou.size > 0 else float('nan')
            iou_max = float(iou.max()) if iou.size > 0 else float('nan')
            rt = float(data['runtime_seconds'])
            lines.append(
                f"{key:<10s} {mode:<6s} {mean_inner:>10.2f} "
                f"{pde_per_seg:>8.1f} {n_seg:>6d} {iou_mean:>9.3f} "
                f"{iou_max:>8.3f} {rt:>7.1f}s"
            )
    out = '\n'.join(lines)
    print(out)
    out_dir = ROOT / 'figures' / 'parabolic'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'table1.txt').write_text(out)
    print(f'\nsaved → {out_dir / "table1.txt"}')


if __name__ == '__main__':
    for key in EX_INFO.keys():
        for mode in ['paper', 'edp']:
            plot_example(key, mode)
    make_table1()
