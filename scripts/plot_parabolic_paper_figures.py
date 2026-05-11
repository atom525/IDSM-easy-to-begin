"""Generate the main Section 5 parabolic reproduction figures.

This script reads the Python-generated caches in ``results/parabolic`` and
writes one main figure per paper example to ``figures/parabolic``:

- ``fig_5_1.png``: Example 5.1, 5% noise
- ``fig_5_1_noise10.png``: Example 5.1, 10% noise
- ``fig_5_2.png``: Example 5.2
- ``fig_5_3.png``: Example 5.3
- ``fig_5_4.png``: Example 5.4
- ``fig_5_5.png``: Example 5.5
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.tri import Triangulation
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.idsm_parabolic import (
    radius_example_5_1,
    radius_example_5_4,
    radius_example_5_5,
    trajectory_example_5_1,
    trajectory_example_5_2,
    trajectory_example_5_3,
    trajectory_example_5_4,
    trajectory_example_5_5,
)


GHOST_R = 1e-6


def _radii_5_2(_t, traj_index):
    return np.array([0.2, 0.2]) if traj_index in (0, 1, 2) else np.array([1e-10, 1e-10])


def _radii_5_3(_t, traj_index):
    return np.array([0.2, 0.2]) if traj_index == 0 else np.array([1e-10, 1e-10])


MAIN_FIGURES = [
    ('5_1', 'paper', 'fig_5_1.png', 'Example 5.1: Conductivity Merging (5% noise)',
     trajectory_example_5_1, lambda t, i: radius_example_5_1(i), (0, 1), 'cond'),
    ('5_1_n10', 'paper', 'fig_5_1_noise10.png', 'Example 5.1: Conductivity Merging (10% noise)',
     trajectory_example_5_1, lambda t, i: radius_example_5_1(i), (0, 1), 'cond'),
    ('5_2', 'paper', 'fig_5_2.png', 'Example 5.2: Mixed Moving',
     trajectory_example_5_2, _radii_5_2, (0, 1, 2), 'double'),
    ('5_3', 'edp', 'fig_5_3.png', 'Example 5.3: Nonlinear U Recovery',
     trajectory_example_5_3, _radii_5_3, (0,), 'nonlinear'),
    ('5_4', 'edp', 'fig_5_4.png', 'Example 5.4: Potential Fading',
     trajectory_example_5_4, lambda t, i: radius_example_5_4(i), (2, 3), 'pot'),
    ('5_5', 'paper', 'fig_5_5.png', 'Example 5.5: Conductivity Diminishing',
     trajectory_example_5_5, radius_example_5_5, (0, 1), 'cond'),
]


def load_npz(key, mode):
    path = ROOT / 'results' / 'parabolic' / f'ex_{key}_{mode}.npz'
    if not path.exists():
        raise FileNotFoundError(f'Missing result cache: {path}')
    return np.load(path, allow_pickle=False)


def draw_inclusions(ax, traj_func, radius_func, t_now, idx_tuple, *, edge='red'):
    for traj_index in idx_tuple:
        center = traj_func(t_now, traj_index)
        radii = radius_func(t_now, traj_index)
        if max(float(radii[0]), float(radii[1])) < GHOST_R:
            continue
        ax.add_patch(Ellipse(
            (float(center[0]), float(center[1])),
            width=2.0 * float(radii[0]),
            height=2.0 * float(radii[1]),
            fill=False,
            edgecolor=edge,
            linewidth=1.4,
        ))


def field_for_model(data, model, frame_index):
    if model in ('pot',):
        return data['v_history'][frame_index]
    if model in ('nonlinear',):
        return data['v_history'][frame_index]
    return data['sigma_history'][frame_index]


def plot_main_figure(key, mode, filename, title, traj_func, radius_func, idx_tuple, model):
    data = load_npz(key, mode)
    points = data['coarse_points']
    triangles = data['coarse_triangles']
    triang = Triangulation(points[:, 0], points[:, 1], triangles)

    n_seg = int(data['iou_history'].shape[0])
    frame_count = min(10, n_seg)
    frame_idx = np.linspace(0, n_seg - 1, frame_count).astype(int)
    total_time = float(data['cfg_total_time'])

    if model == 'double':
        rows = 2
    else:
        rows = 1
    fig, axes = plt.subplots(rows, frame_count, figsize=(2.0 * frame_count, 2.4 * rows), squeeze=False)

    for col, seg_idx in enumerate(frame_idx):
        t_now = (seg_idx + 1) * total_time / n_seg
        if model == 'double':
            sigma = data['sigma_history'][seg_idx]
            potential = data['v_history'][seg_idx]
            c_a = float(data['cfg_cA'])
            c_b = float(data['cfg_cB'])
            v_a = float(data['cfg_vA'])
            v_b = float(data['cfg_vB'])

            ax = axes[0, col]
            ax.tripcolor(triang, facecolors=sigma, shading='flat', cmap='viridis', vmin=c_b, vmax=c_a)
            draw_inclusions(ax, traj_func, radius_func, t_now, (0, 1), edge='red')
            ax.set_ylabel('sigma' if col == 0 else '')

            ax = axes[1, col]
            ax.tripcolor(triang, facecolors=potential, shading='flat', cmap='magma', vmin=v_a, vmax=max(v_b, v_a + 1e-6))
            draw_inclusions(ax, traj_func, radius_func, t_now, (2,), edge='cyan')
            ax.set_ylabel('V' if col == 0 else '')
        else:
            ax = axes[0, col]
            field = field_for_model(data, model, seg_idx)
            if model == 'pot':
                v_a = float(data['cfg_vA'])
                v_b = float(data['cfg_vB'])
                ax.tripcolor(triang, facecolors=field, shading='flat', cmap='magma', vmin=v_a, vmax=max(v_b, v_a + 1e-6))
                draw_inclusions(ax, traj_func, radius_func, t_now, idx_tuple, edge='cyan')
                ax.set_ylabel('V' if col == 0 else '')
            elif model == 'nonlinear':
                u_a = float(data['cfg_vA'])
                u_b = float(data['cfg_vB'])
                ax.tripcolor(triang, facecolors=field, shading='flat', cmap='viridis',
                             vmin=u_a, vmax=2.0 * u_b)
                draw_inclusions(ax, traj_func, radius_func, t_now, idx_tuple, edge='red')
                ax.set_ylabel('U' if col == 0 else '')
            else:
                c_a = float(data['cfg_cA'])
                c_b = float(data['cfg_cB'])
                ax.tripcolor(triang, facecolors=field, shading='flat', cmap='viridis', vmin=c_b, vmax=c_a)
                draw_inclusions(ax, traj_func, radius_func, t_now, idx_tuple, edge='red')
                ax.set_ylabel('sigma' if col == 0 else '')

        for ax in axes[:, col]:
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f't={t_now:.2f}', fontsize=9)

    iou = data['iou_history']
    fig.suptitle(f'{title}  |  IoU mean={iou.mean():.3f}, max={iou.max():.3f}', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_dir = ROOT / 'figures' / 'parabolic'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_path}')


def make_table():
    lines = [
        '# Main Parabolic Reproduction Summary',
        '',
        f"{'Example':<10s} {'cache':<18s} {'n_seg':>6s} {'inner':>7s} {'IoU_mean':>9s} {'IoU_max':>8s} {'runtime':>8s}",
    ]
    for key, mode, _filename, _title, *_rest in MAIN_FIGURES:
        data = load_npz(key, mode)
        iou = data['iou_history']
        inner = data['n_inner_per_segment']
        lines.append(
            f"{key:<10s} {f'ex_{key}_{mode}.npz':<18s} {iou.shape[0]:>6d} "
            f"{float(inner.mean()):>7.2f} {float(iou.mean()):>9.3f} "
            f"{float(iou.max()):>8.3f} {float(data['runtime_seconds']):>7.1f}s"
        )
    out = '\n'.join(lines)
    out_path = ROOT / 'figures' / 'parabolic' / 'table1.txt'
    out_path.write_text(out)
    print(out)
    print(f'saved {out_path}')


if __name__ == '__main__':
    for spec in MAIN_FIGURES:
        plot_main_figure(*spec)
    make_table()
