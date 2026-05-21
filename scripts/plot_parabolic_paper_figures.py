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
from src.plot_style import (
    CONDUCTIVITY_CMAP,
    NONLINEAR_CMAP,
    POTENTIAL_CMAP,
    POTENTIAL_EDGE,
    SIGMA_EDGE,
    add_domain_boundary,
    add_outline_effect,
    apply_idsm_plot_style,
    contrast_norm,
    format_domain_axis,
    save_figure,
)


GHOST_R = 1e-6
apply_idsm_plot_style()


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


def draw_inclusions(ax, traj_func, radius_func, t_now, idx_tuple, *, edge=SIGMA_EDGE):
    for traj_index in idx_tuple:
        center = traj_func(t_now, traj_index)
        radii = radius_func(t_now, traj_index)
        if max(float(radii[0]), float(radii[1])) < GHOST_R:
            continue
        patch = Ellipse(
            (float(center[0]), float(center[1])),
            width=2.0 * float(radii[0]),
            height=2.0 * float(radii[1]),
            fill=False,
            edgecolor=edge,
            linewidth=2.0,
        )
        add_outline_effect(patch, stroke="#111111" if edge == POTENTIAL_EDGE else "white")
        ax.add_patch(patch)


def field_for_model(data, model, frame_index):
    if model in ('pot',):
        return data['v_history'][frame_index]
    if model in ('nonlinear',):
        return data['v_history'][frame_index]
    return data['sigma_history'][frame_index]


def normalize_inhomogeneity(field, background, reference):
    """Normalize one reconstructed inhomogeneity frame for paper-style heatmaps."""
    denom = max(abs(float(reference) - float(background)), 1e-300)
    contrast = np.abs(np.asarray(field, dtype=float) - float(background)) / denom
    max_contrast = float(np.nanmax(contrast)) if contrast.size else 0.0
    if max_contrast <= 1e-14:
        return np.zeros_like(contrast)
    return np.clip(contrast / max_contrast, 0.0, 1.0)


def plot_field(ax, triang, field, *, cmap, vmin, vmax, gamma):
    return ax.tripcolor(
        triang,
        facecolors=field,
        shading='flat',
        cmap=cmap,
        norm=contrast_norm(vmin, vmax, gamma=gamma),
        rasterized=True,
    )


def plot_main_figure(key, mode, filename, title, traj_func, radius_func, idx_tuple, model):
    data = load_npz(key, mode)
    points = data['coarse_points']
    triangles = data['coarse_triangles']
    triang = Triangulation(points[:, 0], points[:, 1], triangles)

    total_time = float(data['cfg_total_time'])
    delta_t = float(data['cfg_delta_t'])
    n_seg = int(data['iou_history'].shape[0])
    target_times = np.arange(1.0, np.floor(total_time) + 1.0, 1.0)
    frame_idx = []
    frame_times = []
    for target_t in target_times:
        idx = int(round(target_t / delta_t)) - 1
        if 0 <= idx < n_seg:
            frame_idx.append(idx)
            frame_times.append(target_t)
    if not frame_idx:
        frame_idx = np.linspace(0, n_seg - 1, min(10, n_seg)).astype(int).tolist()
        frame_times = [(idx + 1) * delta_t for idx in frame_idx]
    frame_count = len(frame_idx)

    if model == 'double':
        rows = 2
    else:
        rows = 1
    # Reserve right-hand margin for shared colorbars.
    fig, axes = plt.subplots(rows, frame_count, figsize=(2.15 * frame_count + 0.6, 2.55 * rows), squeeze=False)

    last_im_row0 = None  # for shared colorbar at row 0
    last_im_row1 = None  # for shared colorbar at row 1 (double mode)
    cmap0 = CONDUCTIVITY_CMAP
    cmap1 = POTENTIAL_CMAP
    label0 = 'sigma norm.'
    label1 = 'V norm.'

    for col, seg_idx in enumerate(frame_idx):
        t_now = float(frame_times[col])
        if model == 'double':
            sigma = data['sigma_history'][seg_idx]
            potential = data['v_history'][seg_idx]
            c_a = float(data['cfg_cA'])
            c_b = float(data['cfg_cB'])
            v_a = float(data['cfg_vA'])
            v_b = float(data['cfg_vB'])

            ax = axes[0, col]
            sigma_show = normalize_inhomogeneity(sigma, c_a, c_b)
            last_im_row0 = plot_field(ax, triang, sigma_show, cmap=CONDUCTIVITY_CMAP, vmin=0.0, vmax=1.0, gamma=0.72)
            draw_inclusions(ax, traj_func, radius_func, t_now, (0, 1), edge=SIGMA_EDGE)
            ax.set_ylabel('sigma norm.' if col == 0 else '', fontweight='bold')

            ax = axes[1, col]
            potential_show = normalize_inhomogeneity(potential, v_a, v_b)
            last_im_row1 = plot_field(ax, triang, potential_show, cmap=POTENTIAL_CMAP, vmin=0.0, vmax=1.0, gamma=0.45)
            draw_inclusions(ax, traj_func, radius_func, t_now, (2,), edge=POTENTIAL_EDGE)
            ax.set_ylabel('V norm.' if col == 0 else '', fontweight='bold')
            cmap0 = CONDUCTIVITY_CMAP
            cmap1 = POTENTIAL_CMAP
            label0, label1 = 'sigma norm.', 'V norm.'
        else:
            ax = axes[0, col]
            field = field_for_model(data, model, seg_idx)
            if model == 'pot':
                v_a = float(data['cfg_vA'])
                v_b = float(data['cfg_vB'])
                field_show = normalize_inhomogeneity(field, v_a, v_b)
                last_im_row0 = plot_field(ax, triang, field_show, cmap=POTENTIAL_CMAP, vmin=0.0, vmax=1.0, gamma=0.42)
                draw_inclusions(ax, traj_func, radius_func, t_now, idx_tuple, edge=POTENTIAL_EDGE)
                ax.set_ylabel('V norm.' if col == 0 else '', fontweight='bold')
                cmap0 = POTENTIAL_CMAP; label0 = 'V norm.'
            elif model == 'nonlinear':
                u_a = float(data['cfg_vA'])
                u_b = float(data['cfg_vB'])
                field_show = normalize_inhomogeneity(field, u_a, u_b)
                last_im_row0 = plot_field(ax, triang, field_show, cmap=NONLINEAR_CMAP, vmin=0.0, vmax=1.0, gamma=0.50)
                draw_inclusions(ax, traj_func, radius_func, t_now, idx_tuple, edge=SIGMA_EDGE)
                ax.set_ylabel('U norm.' if col == 0 else '', fontweight='bold')
                cmap0 = NONLINEAR_CMAP; label0 = 'U norm.'
            else:
                c_a = float(data['cfg_cA'])
                c_b = float(data['cfg_cB'])
                field_show = normalize_inhomogeneity(field, c_a, c_b)
                last_im_row0 = plot_field(ax, triang, field_show, cmap=CONDUCTIVITY_CMAP, vmin=0.0, vmax=1.0, gamma=0.72)
                draw_inclusions(ax, traj_func, radius_func, t_now, idx_tuple, edge=SIGMA_EDGE)
                ax.set_ylabel('sigma norm.' if col == 0 else '', fontweight='bold')
                cmap0 = CONDUCTIVITY_CMAP; label0 = 'sigma norm.'

        for ax in axes[:, col]:
            add_domain_boundary(ax)
            format_domain_axis(ax)
            ax.set_title(f't={t_now:.2f}', fontsize=9, fontweight='bold')

    iou = data['iou_history']
    fig.suptitle(f'{title}  |  IoU mean={iou.mean():.3f}, max={iou.max():.3f}',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 0.93, 0.94])
    # Shared colorbar(s) on the right.
    if rows == 2 and last_im_row0 is not None and last_im_row1 is not None:
        cax0 = fig.add_axes([0.945, 0.53, 0.012, 0.34])
        fig.colorbar(last_im_row0, cax=cax0, label=label0)
        cax1 = fig.add_axes([0.945, 0.10, 0.012, 0.34])
        fig.colorbar(last_im_row1, cax=cax1, label=label1)
    elif last_im_row0 is not None:
        cax0 = fig.add_axes([0.945, 0.18, 0.012, 0.62])
        fig.colorbar(last_im_row0, cax=cax0, label=label0)
    out_dir = ROOT / 'figures' / 'parabolic'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    save_figure(fig, out_path)
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
