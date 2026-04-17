"""重跑 NB04 全部 13 张图（不依赖 jupyter kernel）。

用法：
    cd /mnt/c/Users/maxfo/Desktop/Summer_Research_CUHK/IDSM
    python -m tests.run_nb04_figures
"""

import sys, os, time
import numpy as np

# Agg 后端：无窗口输出
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Notebook04Config, RuntimeConfig
from src.mesh import generate_elliptic_mesh
from src.forward_solver import (
    make_conductivity_example1,
    make_conductivity_conductive,
    make_conductivity_single,
    make_potential_example3,
    generate_cauchy_data,
    generate_cauchy_data_general,
    solve_forward,
)
from src.dsm import compute_dsm_indicator
from src.idsm import run_idsm
from src.idsm_partial import (
    define_accessible_boundary,
    run_idsm_partial,
)
from src.utils import compute_iou, EXAMPLE1_BOXES

# ============================================================
# 初始化
# ============================================================
cfg = Notebook04Config()
runtime_cfg = RuntimeConfig.from_env()
np.random.seed(runtime_cfg.random_seed)

mesh = generate_elliptic_mesh(n_boundary=cfg.mesh.n_boundary)
print(f"网格: {mesh.n_points} 节点, {mesh.n_triangles} 三角形, "
      f"{len(mesh.boundary_nodes)} 边界节点")

fig_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(fig_dir, exist_ok=True)

tri = plt.matplotlib.tri.Triangulation(
    mesh.points[:, 0], mesh.points[:, 1], mesh.triangles
)

T_GLOBAL = time.time()


def save_fig(fig, name):
    path = os.path.join(fig_dir, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → 已保存 {name}")


# ============================================================
# Section 1: DSM vs IDSM — Full-Data Comparison
# ============================================================
print("\n" + "=" * 60)
print("  Section 1: DSM vs IDSM (Full Data)")
print("=" * 60)

sigma_true, u_true = make_conductivity_example1(mesh)
sources = [lambda x, y: x, lambda x, y: y]

noise_levels = cfg.noise_levels
n_iter_idsm = cfg.full.n_iter

dsm_results = {}
idsm_results = {}

for eps in noise_levels:
    np.random.seed(runtime_cfg.random_seed)
    cauchy = generate_cauchy_data(mesh, sigma_true, sources, noise_level=eps)

    t0 = time.time()
    dsm_results[eps] = compute_dsm_indicator(
        mesh, cauchy, gamma=cfg.dsm_gamma, n_grid=cfg.mesh.n_grid
    )
    t_dsm = time.time() - t0

    t0 = time.time()
    idsm_results[eps] = run_idsm(
        mesh, cauchy,
        sigma_bg=cfg.full.sigma_bg, sigma_range=cfg.full.sigma_range,
        alpha=cfg.full.alpha, n_iter=n_iter_idsm,
        lowrank_method=cfg.full.lowrank_method,
        problem_type=cfg.full.problem_type, coeff_known=cfg.full.coeff_known,
        verbose=False, runtime_config=runtime_cfg,
    )
    t_idsm = time.time() - t0

    r = idsm_results[eps]['residuals']
    sf = idsm_results[eps]['sigma_guess'][-1]
    iou = compute_iou(u_true, sf - 1.0, mesh)
    print(f"  ε={eps}: DSM {t_dsm:.1f}s | IDSM {t_idsm:.1f}s, "
          f"resid {r[0]:.4e}→{r[-1]:.4e}, IoU={iou:.4f}")

# --- Figure 1: 04_dsm_vs_idsm.png ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('DSM vs IDSM (Full Data, Example 1)', fontsize=16, y=0.98)

for j, eps in enumerate(noise_levels):
    ax = axes[0, j]
    ind = dsm_results[eps]['indicator']
    gx = dsm_results[eps]['grid_x']
    gy = dsm_results[eps]['grid_y']
    im = ax.pcolormesh(gx, gy, ind.T, cmap='hot_r', shading='auto')
    for box in EXAMPLE1_BOXES:
        cx, cy = box['center']
        hw = box['half_width']
        ax.add_patch(Rectangle((cx - hw, cy - hw), 2 * hw, 2 * hw,
                     fill=False, edgecolor='cyan', linewidth=2, linestyle='--'))
    ax.set_aspect('equal')
    ax.set_title(f'DSM, ε={eps}')
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1, j]
    sf = idsm_results[eps]['sigma_guess'][-1]
    im = ax.tripcolor(tri, sf, cmap='RdBu_r', shading='flat', vmin=0.0, vmax=1.1)
    for box in EXAMPLE1_BOXES:
        cx, cy = box['center']
        hw = box['half_width']
        ax.add_patch(Rectangle((cx - hw, cy - hw), 2 * hw, 2 * hw,
                     fill=False, edgecolor='black', linewidth=2, linestyle='--'))
    ax.set_aspect('equal')
    ax.set_title(f'IDSM, ε={eps}')
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
save_fig(fig, '04_dsm_vs_idsm.png')

# --- Figure 2: 04_idsm_convergence.png ---
fig, ax = plt.subplots(figsize=(8, 5))
for eps in noise_levels:
    r = idsm_results[eps]['residuals']
    ax.semilogy(range(len(r)), r, 'o-', label=f'ε={eps}', markersize=4)
ax.set_xlabel('Iteration')
ax.set_ylabel(r'Residual $\|y^k - y^d\|_{\Gamma}$')
ax.set_title('IDSM Residual Convergence (Example 1, Full Data)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
save_fig(fig, '04_idsm_convergence.png')


# ============================================================
# Section 2: Example 3 Potential
# ============================================================
print("\n" + "=" * 60)
print("  Section 2: Example 3 Potential (pot_exponent=1.5)")
print("=" * 60)

sigma_ex3, potential_ex3, u_pot_ex3 = make_potential_example3(mesh)
sources_ex3 = [lambda x, y: x]

np.random.seed(runtime_cfg.random_seed)
cauchy_ex3 = generate_cauchy_data_general(
    mesh, sigma_ex3, potential_ex3, sources_ex3, noise_level=0.1
)

t0 = time.time()
hist_ex3 = run_idsm(
    mesh, cauchy_ex3,
    sigma_bg=1.0, potential_bg=1e-10,
    sigma_range=1.0, potential_range=10.0,
    alpha=cfg.full.alpha, n_iter=cfg.full.n_iter,
    lowrank_method='DFP', problem_type='potential',
    coeff_known=False, pot_exponent=1.5,
    verbose=False, runtime_config=runtime_cfg,
)
t_ex3 = time.time() - t0
v_final = hist_ex3['potential_guess'][-1]
r = hist_ex3['residuals']
print(f"  时间: {t_ex3:.1f}s, resid {r[0]:.4e}→{r[-1]:.4e}")
print(f"  v ∈ [{v_final.min():.4f}, {v_final.max():.4f}]")

# --- Figure 3: 04_example3_potential.png ---
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

ax = axes[0]
im = ax.tripcolor(tri, potential_ex3, cmap='viridis', shading='flat')
ax.set_aspect('equal')
ax.set_title(r'True potential $v(x)$')
plt.colorbar(im, ax=ax, shrink=0.8)

ax = axes[1]
im = ax.tripcolor(tri, v_final, cmap='viridis', shading='flat')
ax.set_aspect('equal')
ax.set_title(f'IDSM reconstructed $v(x)$\n(22 iters, DFP, pot_exp=1.5)')
plt.colorbar(im, ax=ax, shrink=0.8)

ax = axes[2]
ax.semilogy(range(len(r)), r, 'o-', markersize=4)
ax.set_xlabel('Iteration')
ax.set_ylabel('Residual')
ax.set_title('Residual convergence')
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_fig(fig, '04_example3_potential.png')


# ============================================================
# Section 3: Partial-Data IDSM
# ============================================================
print("\n" + "=" * 60)
print("  Section 3: Partial-Data IDSM")
print("=" * 60)

partial_configs = [
    ('Right half', (-np.pi / 2, np.pi / 2)),
    ('Upper half', (0, np.pi)),
    ('3/4 boundary', (-np.pi / 4, 5 * np.pi / 4)),
]

# --- Figure 4: 04_boundary_configs.png ---
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for i, (label, theta_range) in enumerate(partial_configs):
    gamma_d = define_accessible_boundary(mesh, theta_range)
    ax = axes[i]
    bdry_pts = mesh.points[mesh.boundary_nodes]
    gd_mask = gamma_d['bdry_mask']
    ax.scatter(bdry_pts[gd_mask, 0], bdry_pts[gd_mask, 1],
               c='blue', s=3, label=r'$\Gamma_D$ (data)', zorder=3)
    ax.scatter(bdry_pts[~gd_mask, 0], bdry_pts[~gd_mask, 1],
               c='red', s=3, label=r'$\Gamma_N$ (no data)', zorder=3)
    for box in EXAMPLE1_BOXES:
        cx, cy = box['center']
        hw = box['half_width']
        ax.add_patch(Rectangle((cx - hw, cy - hw), 2 * hw, 2 * hw,
                     fill=True, facecolor='gray', alpha=0.3,
                     edgecolor='black', linewidth=1.5))
    ax.set_aspect('equal')
    ax.set_title(f'{label}\n($\\Gamma_D$: {gd_mask.sum()}/{len(gd_mask)} nodes)')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.95, 0.95)
plt.tight_layout()
save_fig(fig, '04_boundary_configs.png')

# 重建
sigma_true, u_true = make_conductivity_example1(mesh)
sources = [lambda x, y: x, lambda x, y: y]
np.random.seed(runtime_cfg.random_seed)
cauchy = generate_cauchy_data(mesh, sigma_true, sources, noise_level=0.1)

print("  Full-data IDSM (reference)...")
t0 = time.time()
hist_full = run_idsm(
    mesh, cauchy,
    sigma_bg=cfg.full.sigma_bg, sigma_range=cfg.full.sigma_range,
    alpha=cfg.full.alpha, n_iter=cfg.full.n_iter,
    lowrank_method=cfg.full.lowrank_method,
    problem_type=cfg.full.problem_type, coeff_known=cfg.full.coeff_known,
    verbose=False, runtime_config=runtime_cfg,
)
print(f"    resid {hist_full['residuals'][-1]:.4e}, {time.time() - t0:.1f}s")

partial_results = {}
for name, theta_range in partial_configs:
    gamma_d = define_accessible_boundary(mesh, theta_range)
    n_gd = gamma_d['node_mask'][mesh.boundary_nodes].sum()
    print(f"  Partial: {name} (Γ_D: {n_gd}/{len(mesh.boundary_nodes)})...")
    t0 = time.time()
    hist = run_idsm_partial(
        mesh, cauchy, gamma_d,
        sigma_bg=cfg.partial.sigma_bg, sigma_range=cfg.partial.sigma_range,
        alpha_d=cfg.partial.alpha_d, alpha_n=cfg.partial.alpha_n,
        n_iter=cfg.partial.n_iter,
        lowrank_method=cfg.partial.lowrank_method,
        problem_type=cfg.partial.problem_type, coeff_known=cfg.partial.coeff_known,
        gamma_D=cfg.partial.gamma_D, epsilon_cutoff=cfg.partial.epsilon_cutoff,
        p_norm=cfg.partial.p_norm,
        verbose=False, runtime_config=runtime_cfg,
    )
    t_p = time.time() - t0
    r = hist['residuals']
    sp = hist['sigma_guess'][-1]
    iou = compute_iou(u_true, sp - 1.0, mesh)
    decreased = "✓" if r[-1] < r[0] else "✗"
    print(f"    resid {r[0]:.4e}→{r[-1]:.4e} {decreased}, IoU={iou:.4f}, {t_p:.1f}s")
    partial_results[name] = hist

# --- Figure 5: 04_partial_reconstruction.png ---
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

ax = axes[0, 0]
sigma_full_f = hist_full['sigma_guess'][-1]
im = ax.tripcolor(tri, sigma_full_f, cmap='RdBu_r', shading='flat', vmin=0.0, vmax=1.1)
for box in EXAMPLE1_BOXES:
    cx, cy = box['center']
    hw = box['half_width']
    ax.add_patch(Rectangle((cx - hw, cy - hw), 2 * hw, 2 * hw,
                 fill=False, edgecolor='black', linewidth=2, linestyle='--'))
ax.set_aspect('equal')
ax.set_title(f'Full data IDSM\nσ range: [{sigma_full_f.min():.3f}, {sigma_full_f.max():.3f}]')
plt.colorbar(im, ax=ax, shrink=0.8)

for idx, (name, theta_range) in enumerate(partial_configs):
    ax = axes[(idx + 1) // 2, (idx + 1) % 2]
    sp = partial_results[name]['sigma_guess'][-1]
    im = ax.tripcolor(tri, sp, cmap='RdBu_r', shading='flat', vmin=0.0, vmax=1.1)
    for box in EXAMPLE1_BOXES:
        cx, cy = box['center']
        hw = box['half_width']
        ax.add_patch(Rectangle((cx - hw, cy - hw), 2 * hw, 2 * hw,
                     fill=False, edgecolor='black', linewidth=2, linestyle='--'))
    ax.set_aspect('equal')
    ax.set_title(f'Partial: {name}\nσ range: [{sp.min():.3f}, {sp.max():.3f}]')
    plt.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle('Full vs Partial Data IDSM Reconstruction (ε=0.1)', fontsize=14, y=1.01)
plt.tight_layout()
save_fig(fig, '04_partial_reconstruction.png')

# --- Figure 6: 04_partial_convergence.png ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(range(len(hist_full['residuals'])), hist_full['residuals'],
            'k-o', label='Full data', markersize=4, linewidth=2)
colors = ['tab:blue', 'tab:orange', 'tab:green']
for (name, _), color in zip(partial_configs, colors):
    r = partial_results[name]['residuals']
    ax.semilogy(range(len(r)), r, '-o', color=color, label=f'Partial: {name}', markersize=3)
ax.set_xlabel('Iteration')
ax.set_ylabel('Residual')
ax.set_title('Residual Convergence: Full vs Partial Data')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
save_fig(fig, '04_partial_convergence.png')

# --- Figure 7: 04_damping_factor.png ---
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for idx, (name, _) in enumerate(partial_configs):
    lam_hist = partial_results[name].get('lambda_history', [])
    ax = axes[idx]
    if len(lam_hist) > 0:
        ax.plot(range(len(lam_hist)), lam_hist, 'o-', markersize=4, linewidth=1.5)
        ax.set_xlabel('Iteration $k$')
        ax.set_ylabel(r'$\lambda_{k,p}$')
        ax.set_title(f'Partial: {name}')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    else:
        ax.text(0.5, 0.5, 'No lambda history', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title(f'Partial: {name}')
fig.suptitle(r'Damping Factor $\lambda_{k,p}$ History (Paper 3, Fig. 7 style)',
             fontsize=13, y=1.02)
plt.tight_layout()
save_fig(fig, '04_damping_factor.png')

for name, _ in partial_configs:
    lam = partial_results[name].get('lambda_history', [])
    if lam:
        print(f"  λ ({name}): [{min(lam):.4e}, {max(lam):.4e}], "
              f"min at iter {np.argmin(lam)}")

# --- Ablation ---
print("\n  Ablation: Homogeneous vs HR-DtN (right half)...")
gamma_right_abl = define_accessible_boundary(mesh, (-np.pi / 2, np.pi / 2))

# 使用与 partial 相同的 cauchy 数据（上面已生成）
hist_homo = run_idsm_partial(
    mesh, cauchy, gamma_right_abl,
    sigma_bg=1.0, sigma_range=0.01,
    alpha_d=1.0, alpha_n=1.0,
    n_iter=cfg.partial.n_iter, lowrank_method='BFG',
    problem_type='conductivity',
    verbose=False, runtime_config=runtime_cfg,
)
hist_hr = run_idsm_partial(
    mesh, cauchy, gamma_right_abl,
    sigma_bg=1.0, sigma_range=0.01,
    alpha_d=0.05, alpha_n=2.0,
    n_iter=cfg.partial.n_iter, lowrank_method='BFG',
    problem_type='conductivity',
    verbose=False, runtime_config=runtime_cfg,
)
iou_homo = compute_iou(u_true, hist_homo['sigma_guess'][-1] - 1.0, mesh)
iou_hr = compute_iou(u_true, hist_hr['sigma_guess'][-1] - 1.0, mesh)
print(f"    Homo resid={hist_homo['residuals'][-1]:.4e}, IoU={iou_homo:.4f}")
print(f"    HR-DtN resid={hist_hr['residuals'][-1]:.4e}, IoU={iou_hr:.4f}")

# --- Figure 8: 04_ablation_dtn.png ---
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

configs_abl = [
    (r'Homogeneous ($\alpha_d=\alpha_n=1$)', hist_homo),
    (r'HR-DtN ($\alpha_d=0.05, \alpha_n=2$)', hist_hr),
]
for idx, (label, hist) in enumerate(configs_abl):
    ax = axes[idx]
    sf = hist['sigma_guess'][-1]
    im = ax.tripcolor(tri, sf, cmap='RdBu_r', shading='flat', vmin=0.0, vmax=1.1)
    for box in EXAMPLE1_BOXES:
        cx, cy = box['center']
        hw = box['half_width']
        ax.add_patch(Rectangle((cx - hw, cy - hw), 2 * hw, 2 * hw,
                     fill=False, edgecolor='black', linewidth=2, linestyle='--'))
    ax.set_aspect('equal')
    ax.set_title(label)
    plt.colorbar(im, ax=ax, shrink=0.8)

ax = axes[2]
ax.semilogy(range(len(hist_homo['residuals'])), hist_homo['residuals'],
            'b-o', label='Homogeneous', markersize=3)
ax.semilogy(range(len(hist_hr['residuals'])), hist_hr['residuals'],
            'r-s', label='HR-DtN', markersize=3)
ax.set_xlabel('Iteration')
ax.set_ylabel('Residual')
ax.set_title('Residual Convergence')
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle(r'Ablation: Homogeneous vs Heterogeneous DtN (right half, $\varepsilon=10\%$)',
             fontsize=13, y=1.02)
plt.tight_layout()
save_fig(fig, '04_ablation_dtn.png')

# --- Disconnected arcs ---
print("\n  Disconnected arcs...")
node_mask_disc = np.zeros(mesh.n_points, dtype=bool)
for idx in mesh.boundary_nodes:
    x, y_coord = mesh.points[idx]
    theta = np.arctan2(y_coord / 0.8, x)
    in_arc1 = (0 <= theta <= np.pi / 2)
    in_arc2 = (-np.pi <= theta <= -np.pi / 2)
    node_mask_disc[idx] = in_arc1 or in_arc2
gamma_disc = {"node_mask": node_mask_disc, "bdry_mask": node_mask_disc[mesh.boundary_nodes]}
n_disc = gamma_disc['bdry_mask'].sum()
print(f"    Γ_D: {n_disc}/{len(mesh.boundary_nodes)} nodes")

# --- Figure 9: 04_disconnected_arcs.png ---
fig, ax = plt.subplots(figsize=(6, 5))
bdry_pts = mesh.points[mesh.boundary_nodes]
gd_mask_d = gamma_disc['bdry_mask']
ax.scatter(bdry_pts[gd_mask_d, 0], bdry_pts[gd_mask_d, 1],
           c='blue', s=5, label=r'$\Gamma_D$ (data)', zorder=3)
ax.scatter(bdry_pts[~gd_mask_d, 0], bdry_pts[~gd_mask_d, 1],
           c='red', s=5, label=r'$\Gamma_N$ (no data)', zorder=3)
for box in EXAMPLE1_BOXES:
    cx, cy = box['center']
    hw = box['half_width']
    ax.add_patch(Rectangle((cx - hw, cy - hw), 2 * hw, 2 * hw,
                 fill=True, facecolor='gray', alpha=0.3,
                 edgecolor='black', linewidth=1.5))
ax.set_aspect('equal')
ax.set_title(f'Disconnected arcs (top-right + bottom-left)\n'
             f'$\\Gamma_D$: {n_disc}/{len(mesh.boundary_nodes)} nodes')
ax.legend(fontsize=8)
plt.tight_layout()
save_fig(fig, '04_disconnected_arcs.png')

hist_disc = run_idsm_partial(
    mesh, cauchy, gamma_disc,
    sigma_bg=1.0, sigma_range=0.01,
    alpha_d=0.05, alpha_n=2.0,
    n_iter=cfg.partial.n_iter, lowrank_method='BFG',
    problem_type='conductivity',
    verbose=False, runtime_config=runtime_cfg,
)
sf_disc = hist_disc['sigma_guess'][-1]
iou_disc = compute_iou(u_true, sf_disc - 1.0, mesh)
r_disc = hist_disc['residuals']
print(f"    resid {r_disc[0]:.4e}→{r_disc[-1]:.4e}, IoU={iou_disc:.4f}")

# --- Figure 10: 04_disconnected_reconstruction.png ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
im = ax.tripcolor(tri, sf_disc, cmap='RdBu_r', shading='flat', vmin=0.0, vmax=1.1)
for box in EXAMPLE1_BOXES:
    cx, cy = box['center']
    hw = box['half_width']
    ax.add_patch(Rectangle((cx - hw, cy - hw), 2 * hw, 2 * hw,
                 fill=False, edgecolor='black', linewidth=2, linestyle='--'))
ax.set_aspect('equal')
ax.set_title(f'Disconnected arcs reconstruction\n'
             f'$\\sigma$ range: [{sf_disc.min():.3f}, {sf_disc.max():.3f}]')
plt.colorbar(im, ax=ax, shrink=0.8)

ax = axes[1]
ax.semilogy(range(len(r_disc)), r_disc, 'g-o', markersize=3)
ax.set_xlabel('Iteration')
ax.set_ylabel('Residual')
ax.set_title('Residual Convergence (disconnected arcs)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_fig(fig, '04_disconnected_reconstruction.png')


# ============================================================
# Section 4: Noise Robustness Sweep
# ============================================================
print("\n" + "=" * 60)
print("  Section 4: Noise Robustness Sweep")
print("=" * 60)

eps_sweep = cfg.eps_sweep
gamma_right = define_accessible_boundary(mesh, theta_range=(-np.pi / 2, np.pi / 2))

iou_dsm_list = []
iou_idsm_list = []
iou_partial_list = []

for eps in eps_sweep:
    np.random.seed(runtime_cfg.random_seed)
    cauchy_sw = generate_cauchy_data(mesh, sigma_true, sources, noise_level=eps)

    # DSM
    dsm_r = compute_dsm_indicator(mesh, cauchy_sw, gamma=cfg.dsm_gamma, n_grid=cfg.mesh.n_grid)
    zeta = dsm_r['zeta_sum']
    u_dsm = np.zeros(mesh.n_triangles)
    for tri_idx in range(mesh.n_triangles):
        i0, i1, i2 = mesh.triangles[tri_idx]
        u_dsm[tri_idx] = (zeta[i0] + zeta[i1] + zeta[i2]) / 3.0
    iou_dsm_list.append(compute_iou(u_true, u_dsm, mesh))

    # IDSM full
    hist_f = run_idsm(
        mesh, cauchy_sw,
        sigma_bg=cfg.full.sigma_bg, sigma_range=cfg.full.sigma_range,
        alpha=cfg.full.alpha, n_iter=cfg.full.n_iter,
        lowrank_method=cfg.full.lowrank_method,
        problem_type=cfg.full.problem_type, coeff_known=cfg.full.coeff_known,
        verbose=False, runtime_config=runtime_cfg,
    )
    iou_idsm_list.append(compute_iou(u_true, hist_f['sigma_guess'][-1] - 1.0, mesh))

    # IDSM partial (right half)
    hist_p = run_idsm_partial(
        mesh, cauchy_sw, gamma_right,
        sigma_bg=cfg.partial.sigma_bg, sigma_range=cfg.partial.sigma_range,
        alpha_d=cfg.partial.alpha_d, alpha_n=cfg.partial.alpha_n,
        n_iter=cfg.partial.n_iter,
        lowrank_method=cfg.partial.lowrank_method,
        problem_type=cfg.partial.problem_type, coeff_known=cfg.partial.coeff_known,
        gamma_D=cfg.partial.gamma_D, epsilon_cutoff=cfg.partial.epsilon_cutoff,
        p_norm=cfg.partial.p_norm,
        verbose=False, runtime_config=runtime_cfg,
    )
    iou_partial_list.append(compute_iou(u_true, hist_p['sigma_guess'][-1] - 1.0, mesh))

    print(f"  ε={eps:.2f}: DSM={iou_dsm_list[-1]:.4f}, "
          f"IDSM-full={iou_idsm_list[-1]:.4f}, "
          f"IDSM-partial={iou_partial_list[-1]:.4f}")

# --- Figure 11: 04_noise_robustness.png ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(eps_sweep, iou_dsm_list, 's-', label='DSM', markersize=6)
ax.plot(eps_sweep, iou_idsm_list, 'o-', label='IDSM (full data)', markersize=6)
ax.plot(eps_sweep, iou_partial_list, '^-', label='IDSM (right half)', markersize=6)
ax.set_xlabel('Noise level ε')
ax.set_ylabel('IoU')
ax.set_title('Noise Robustness: IoU vs Noise Level')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)
plt.tight_layout()
save_fig(fig, '04_noise_robustness.png')


# ============================================================
# Section 5: Conductive vs Insulating
# ============================================================
print("\n" + "=" * 60)
print("  Section 5: Conductive vs Insulating")
print("=" * 60)

# Insulating (Example 1, σ=0.3)
sigma_ins, u_ins = make_conductivity_example1(mesh)
np.random.seed(runtime_cfg.random_seed)
cauchy_ins = generate_cauchy_data(mesh, sigma_ins, sources, noise_level=0.1)

dsm_ins = compute_dsm_indicator(mesh, cauchy_ins, gamma=0.5, n_grid=201)
hist_ins = run_idsm(mesh, cauchy_ins, sigma_bg=1.0, sigma_range=0.01,
                     alpha=1.0, n_iter=22, lowrank_method='BFG',
                     verbose=False, runtime_config=runtime_cfg)

# Conductive (σ=3.0)
sigma_con, u_con = make_conductivity_conductive(mesh)
np.random.seed(runtime_cfg.random_seed)
cauchy_con = generate_cauchy_data(mesh, sigma_con, sources, noise_level=0.1)

dsm_con = compute_dsm_indicator(mesh, cauchy_con, gamma=0.5, n_grid=201)
hist_con = run_idsm(mesh, cauchy_con, sigma_bg=1.0, sigma_range=3.0,
                     alpha=1.0, n_iter=22, lowrank_method='BFG',
                     verbose=False, runtime_config=runtime_cfg)

sf_ins = hist_ins['sigma_guess'][-1]
sf_con = hist_con['sigma_guess'][-1]
iou_ins = compute_iou(u_ins, sf_ins - 1.0, mesh)
iou_con = compute_iou(u_con, sf_con - 1.0, mesh)
print(f"  Insulating: IoU={iou_ins:.4f}, σ_min={sf_ins.min():.4f}")
print(f"  Conductive: IoU={iou_con:.4f}, σ_max={sf_con.max():.4f}")

# --- Figure 12: 04_classify_comparison.png ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

ax = axes[0, 0]
im = ax.tripcolor(tri, sigma_ins, cmap='RdBu_r', shading='flat', vmin=0.0, vmax=3.5)
for box in EXAMPLE1_BOXES:
    cx, cy = box['center']
    hw = box['half_width']
    ax.add_patch(Rectangle((cx - hw, cy - hw), 2 * hw, 2 * hw, fill=False,
                 edgecolor='k', linewidth=2, linestyle='--'))
ax.set_aspect('equal')
ax.set_title(r'True: Insulating ($\sigma=0.3$)')
plt.colorbar(im, ax=ax, shrink=0.8)

ax = axes[0, 1]
gx = dsm_ins['grid_x']
gy = dsm_ins['grid_y']
im = ax.pcolormesh(gx, gy, dsm_ins['indicator'].T, cmap='hot_r', shading='auto')
for box in EXAMPLE1_BOXES:
    cx, cy = box['center']
    hw = box['half_width']
    ax.add_patch(Rectangle((cx - hw, cy - hw), 2 * hw, 2 * hw, fill=False,
                 edgecolor='cyan', linewidth=2, linestyle='--'))
ax.set_aspect('equal')
ax.set_title(r'DSM (insulating): $\eta \geq 0$ always')
plt.colorbar(im, ax=ax, shrink=0.8)

ax = axes[0, 2]
im = ax.tripcolor(tri, sf_ins, cmap='RdBu_r', shading='flat', vmin=0.0, vmax=3.5)
for box in EXAMPLE1_BOXES:
    cx, cy = box['center']
    hw = box['half_width']
    ax.add_patch(Rectangle((cx - hw, cy - hw), 2 * hw, 2 * hw, fill=False,
                 edgecolor='k', linewidth=2, linestyle='--'))
ax.set_aspect('equal')
ax.set_title(r'IDSM: $\sigma_{\min}=%.3f$ ($\sigma < \sigma_0$ ✓)' % sf_ins.min())
plt.colorbar(im, ax=ax, shrink=0.8)

ax = axes[1, 0]
im = ax.tripcolor(tri, sigma_con, cmap='RdBu_r', shading='flat', vmin=0.0, vmax=3.5)
for box in EXAMPLE1_BOXES:
    cx, cy = box['center']
    hw = box['half_width']
    ax.add_patch(Rectangle((cx - hw, cy - hw), 2 * hw, 2 * hw, fill=False,
                 edgecolor='k', linewidth=2, linestyle='--'))
ax.set_aspect('equal')
ax.set_title(r'True: Conductive ($\sigma=3.0$)')
plt.colorbar(im, ax=ax, shrink=0.8)

ax = axes[1, 1]
im = ax.pcolormesh(gx, gy, dsm_con['indicator'].T, cmap='hot_r', shading='auto')
for box in EXAMPLE1_BOXES:
    cx, cy = box['center']
    hw = box['half_width']
    ax.add_patch(Rectangle((cx - hw, cy - hw), 2 * hw, 2 * hw, fill=False,
                 edgecolor='cyan', linewidth=2, linestyle='--'))
ax.set_aspect('equal')
ax.set_title(r'DSM (conductive): $\eta \geq 0$ always')
plt.colorbar(im, ax=ax, shrink=0.8)

ax = axes[1, 2]
im = ax.tripcolor(tri, sf_con, cmap='RdBu_r', shading='flat', vmin=0.0, vmax=3.5)
for box in EXAMPLE1_BOXES:
    cx, cy = box['center']
    hw = box['half_width']
    ax.add_patch(Rectangle((cx - hw, cy - hw), 2 * hw, 2 * hw, fill=False,
                 edgecolor='k', linewidth=2, linestyle='--'))
ax.set_aspect('equal')
ax.set_title(r'IDSM: $\sigma_{\max}=%.3f$ ($\sigma > \sigma_0$ ✓)' % sf_con.max())
plt.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle(r'Conductive vs Insulating: DSM Cannot Classify, IDSM Can ($\varepsilon=10\%$)',
             fontsize=14, y=1.01)
plt.tight_layout()
save_fig(fig, '04_classify_comparison.png')


# ============================================================
# Section 6: Single vs Multiple Inclusions
# ============================================================
print("\n" + "=" * 60)
print("  Section 6: Single vs Multiple")
print("=" * 60)

sigma_single, u_single = make_conductivity_single(mesh)
np.random.seed(runtime_cfg.random_seed)
cauchy_single = generate_cauchy_data(mesh, sigma_single, sources, noise_level=0.1)

dsm_single = compute_dsm_indicator(mesh, cauchy_single, gamma=0.5, n_grid=201)
hist_single = run_idsm(mesh, cauchy_single, sigma_bg=1.0, sigma_range=0.01,
                        alpha=1.0, n_iter=22, lowrank_method='BFG',
                        verbose=False, runtime_config=runtime_cfg)

sigma_multi, u_multi = make_conductivity_example1(mesh)
np.random.seed(runtime_cfg.random_seed)
cauchy_multi = generate_cauchy_data(mesh, sigma_multi, sources, noise_level=0.1)

dsm_multi = compute_dsm_indicator(mesh, cauchy_multi, gamma=0.5, n_grid=201)
hist_multi = run_idsm(mesh, cauchy_multi, sigma_bg=1.0, sigma_range=0.01,
                       alpha=1.0, n_iter=22, lowrank_method='BFG',
                       verbose=False, runtime_config=runtime_cfg)

sf_s = hist_single['sigma_guess'][-1]
sf_m = hist_multi['sigma_guess'][-1]
iou_s = compute_iou(u_single, sf_s - 1.0, mesh)
iou_m = compute_iou(u_multi, sf_m - 1.0, mesh)
print(f"  Single: IoU={iou_s:.4f}, σ_min={sf_s.min():.4f}")
print(f"  Multiple: IoU={iou_m:.4f}, σ_min={sf_m.min():.4f}")

# --- Figure 13: 04_single_vs_multiple.png ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
gx = dsm_single['grid_x']
gy = dsm_single['grid_y']

ax = axes[0, 0]
im = ax.tripcolor(tri, sigma_single, cmap='RdBu_r', shading='flat', vmin=0.0, vmax=1.1)
circle = plt.Circle((0.3, 0.0), 0.25, fill=False, edgecolor='k', linewidth=2, linestyle='--')
ax.add_patch(circle)
ax.set_aspect('equal')
ax.set_title('True: Single Circular Inclusion')
plt.colorbar(im, ax=ax, shrink=0.8)

ax = axes[0, 1]
im = ax.pcolormesh(gx, gy, dsm_single['indicator'].T, cmap='hot_r', shading='auto')
circle = plt.Circle((0.3, 0.0), 0.25, fill=False, edgecolor='cyan', linewidth=2, linestyle='--')
ax.add_patch(circle)
ax.set_aspect('equal')
ax.set_title('DSM (single)')
plt.colorbar(im, ax=ax, shrink=0.8)

ax = axes[0, 2]
im = ax.tripcolor(tri, sf_s, cmap='RdBu_r', shading='flat', vmin=0.0, vmax=1.1)
circle = plt.Circle((0.3, 0.0), 0.25, fill=False, edgecolor='k', linewidth=2, linestyle='--')
ax.add_patch(circle)
ax.set_aspect('equal')
ax.set_title(f'IDSM (single): $\\sigma_{{\\min}}={sf_s.min():.3f}$')
plt.colorbar(im, ax=ax, shrink=0.8)

ax = axes[1, 0]
im = ax.tripcolor(tri, sigma_multi, cmap='RdBu_r', shading='flat', vmin=0.0, vmax=1.1)
for box in EXAMPLE1_BOXES:
    cx, cy = box['center']
    hw = box['half_width']
    ax.add_patch(Rectangle((cx - hw, cy - hw), 2 * hw, 2 * hw, fill=False,
                 edgecolor='k', linewidth=2, linestyle='--'))
ax.set_aspect('equal')
ax.set_title('True: Two Square Inclusions')
plt.colorbar(im, ax=ax, shrink=0.8)

ax = axes[1, 1]
im = ax.pcolormesh(gx, gy, dsm_multi['indicator'].T, cmap='hot_r', shading='auto')
for box in EXAMPLE1_BOXES:
    cx, cy = box['center']
    hw = box['half_width']
    ax.add_patch(Rectangle((cx - hw, cy - hw), 2 * hw, 2 * hw, fill=False,
                 edgecolor='cyan', linewidth=2, linestyle='--'))
ax.set_aspect('equal')
ax.set_title('DSM (multiple)')
plt.colorbar(im, ax=ax, shrink=0.8)

ax = axes[1, 2]
im = ax.tripcolor(tri, sf_m, cmap='RdBu_r', shading='flat', vmin=0.0, vmax=1.1)
for box in EXAMPLE1_BOXES:
    cx, cy = box['center']
    hw = box['half_width']
    ax.add_patch(Rectangle((cx - hw, cy - hw), 2 * hw, 2 * hw, fill=False,
                 edgecolor='k', linewidth=2, linestyle='--'))
ax.set_aspect('equal')
ax.set_title(f'IDSM (multiple): $\\sigma_{{\\min}}={sf_m.min():.3f}$')
plt.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle(r'Single vs Multiple Inclusions ($\varepsilon=10\%$)', fontsize=14, y=1.01)
plt.tight_layout()
save_fig(fig, '04_single_vs_multiple.png')


# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("  全部完成")
print("=" * 60)
total = time.time() - T_GLOBAL
print(f"  总耗时: {total:.0f}s ({total / 60:.1f}min)")
print(f"  图片目录: {os.path.abspath(fig_dir)}")

# 列出所有生成的图片
figs_generated = sorted(f for f in os.listdir(fig_dir) if f.startswith('04_'))
print(f"  生成图片数: {len(figs_generated)}")
for f in figs_generated:
    sz = os.path.getsize(os.path.join(fig_dir, f))
    print(f"    {f} ({sz / 1024:.0f} KB)")
