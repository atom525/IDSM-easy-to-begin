"""快速验证 NB04 修复效果（不依赖 jupyter kernel）。"""

import sys, os, time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Notebook04Config, RuntimeConfig
from src.mesh import generate_elliptic_mesh
from src.forward_solver import (
    make_conductivity_example1,
    make_potential_example3,
    generate_cauchy_data,
    generate_cauchy_data_general,
)
from src.idsm import run_idsm
from src.idsm_partial import define_accessible_boundary, run_idsm_partial
from src.utils import compute_iou

cfg = Notebook04Config()
runtime_cfg = RuntimeConfig.from_env()
np.random.seed(runtime_cfg.random_seed)

mesh = generate_elliptic_mesh(n_boundary=cfg.mesh.n_boundary)
print(f"网格: {mesh.n_points} 节点, {mesh.n_triangles} 三角形")

# ============================================================
# 1. Full data IDSM (Example 1, baseline)
# ============================================================
print("\n" + "="*60)
print("  Section 1: Full data IDSM (Example 1)")
print("="*60)
sigma_true, u_true = make_conductivity_example1(mesh)
sources = [lambda x, y: x, lambda x, y: y]

np.random.seed(runtime_cfg.random_seed)
cauchy = generate_cauchy_data(mesh, sigma_true, sources, noise_level=0.1)

hist_full = run_idsm(
    mesh, cauchy,
    sigma_bg=cfg.full.sigma_bg, sigma_range=cfg.full.sigma_range,
    alpha=cfg.full.alpha, n_iter=cfg.full.n_iter,
    lowrank_method=cfg.full.lowrank_method,
    problem_type=cfg.full.problem_type,
    coeff_known=cfg.full.coeff_known,
    verbose=False, runtime_config=runtime_cfg,
)
sf = hist_full['sigma_guess'][-1]
iou_full = compute_iou(u_true, sf - 1.0, mesh)
print(f"  残差: {hist_full['residuals'][0]:.4e} → {hist_full['residuals'][-1]:.4e}")
print(f"  σ ∈ [{sf.min():.4f}, {sf.max():.4f}]")
print(f"  IoU: {iou_full:.4f}")

# ============================================================
# 2. Example 3: Potential (with pot_exponent=1.5)
# ============================================================
print("\n" + "="*60)
print("  Section 2: Example 3 Potential (pot_exponent=1.5)")
print("="*60)
sigma_ex3, potential_ex3, u_pot_ex3 = make_potential_example3(mesh)
sources_ex3 = [lambda x, y: x]
np.random.seed(runtime_cfg.random_seed)
cauchy_ex3 = generate_cauchy_data_general(
    mesh, sigma_ex3, potential_ex3, sources_ex3, noise_level=0.1
)

hist_ex3 = run_idsm(
    mesh, cauchy_ex3,
    sigma_bg=1.0, potential_bg=1e-10,
    sigma_range=1.0, potential_range=10.0,
    alpha=cfg.full.alpha, n_iter=cfg.full.n_iter,
    lowrank_method='DFP', problem_type='potential',
    coeff_known=False, pot_exponent=1.5,
    verbose=False, runtime_config=runtime_cfg,
)
v_final = hist_ex3['potential_guess'][-1]
print(f"  残差: {hist_ex3['residuals'][0]:.4e} → {hist_ex3['residuals'][-1]:.4e}")
print(f"  v ∈ [{v_final.min():.4f}, {v_final.max():.4f}]")
# potential 重建和真值的 IoU（v > threshold 作为夹杂体）
v_true_u = potential_ex3 - 1e-10
v_recon_u = v_final - 1e-10
iou_pot = compute_iou(v_true_u, v_recon_u, mesh)
print(f"  IoU (potential): {iou_pot:.4f}")

# ============================================================
# 3. Partial data IDSM (3 configs)
# ============================================================
print("\n" + "="*60)
print("  Section 3: Partial data IDSM")
print("="*60)
partial_configs = [
    ('Right half', (-np.pi/2, np.pi/2)),
    ('Upper half', (0, np.pi)),
    ('3/4 boundary', (-np.pi/4, 5*np.pi/4)),
]

np.random.seed(runtime_cfg.random_seed)
cauchy_p = generate_cauchy_data(mesh, sigma_true, sources, noise_level=0.1)

for name, theta_range in partial_configs:
    gamma_d = define_accessible_boundary(mesh, theta_range)
    n_gd = gamma_d['node_mask'][mesh.boundary_nodes].sum()

    t0 = time.time()
    hist_p = run_idsm_partial(
        mesh, cauchy_p, gamma_d,
        sigma_bg=cfg.partial.sigma_bg, sigma_range=cfg.partial.sigma_range,
        alpha_d=cfg.partial.alpha_d, alpha_n=cfg.partial.alpha_n,
        n_iter=cfg.partial.n_iter,
        lowrank_method=cfg.partial.lowrank_method,
        problem_type=cfg.partial.problem_type,
        coeff_known=cfg.partial.coeff_known,
        gamma_D=cfg.partial.gamma_D,
        epsilon_cutoff=cfg.partial.epsilon_cutoff,
        p_norm=cfg.partial.p_norm,
        verbose=False, runtime_config=runtime_cfg,
    )
    t_p = time.time() - t0
    sp = hist_p['sigma_guess'][-1]
    iou_p = compute_iou(u_true, sp - 1.0, mesh)
    lam = hist_p.get('lambda_history', [])

    decreased = "✓" if hist_p['residuals'][-1] < hist_p['residuals'][0] else "✗"
    print(f"\n  {name} (Γ_D: {n_gd}/{len(mesh.boundary_nodes)} nodes)")
    print(f"    残差: {hist_p['residuals'][0]:.4e} → {hist_p['residuals'][-1]:.4e} {decreased}")
    print(f"    σ ∈ [{sp.min():.4f}, {sp.max():.4f}]")
    print(f"    IoU: {iou_p:.4f}")
    print(f"    时间: {t_p:.1f}s")
    if lam:
        print(f"    λ: [{min(lam):.4e}, {max(lam):.4e}]")

# ============================================================
# 4. Ablation: Homogeneous vs HR-DtN
# ============================================================
print("\n" + "="*60)
print("  Section 4: Ablation (right half)")
print("="*60)
gamma_right = define_accessible_boundary(mesh, (-np.pi/2, np.pi/2))

hist_homo = run_idsm_partial(
    mesh, cauchy_p, gamma_right,
    sigma_bg=1.0, sigma_range=0.01,
    alpha_d=1.0, alpha_n=1.0,
    n_iter=cfg.partial.n_iter,
    lowrank_method='BFG', problem_type='conductivity',
    verbose=False, runtime_config=runtime_cfg,
)
hist_hr = run_idsm_partial(
    mesh, cauchy_p, gamma_right,
    sigma_bg=1.0, sigma_range=0.01,
    alpha_d=0.05, alpha_n=2.0,
    n_iter=cfg.partial.n_iter,
    lowrank_method='BFG', problem_type='conductivity',
    verbose=False, runtime_config=runtime_cfg,
)
iou_homo = compute_iou(u_true, hist_homo['sigma_guess'][-1] - 1.0, mesh)
iou_hr = compute_iou(u_true, hist_hr['sigma_guess'][-1] - 1.0, mesh)
print(f"  Homogeneous (α_d=α_n=1): resid {hist_homo['residuals'][-1]:.4e}, IoU {iou_homo:.4f}")
print(f"  HR-DtN (α_d=0.05, α_n=2): resid {hist_hr['residuals'][-1]:.4e}, IoU {iou_hr:.4f}")

print("\n" + "="*60)
print("  验证完成")
print("="*60)
