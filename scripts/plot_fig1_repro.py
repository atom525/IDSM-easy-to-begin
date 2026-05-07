"""复现 paper Fig 1：Example 5.1 σ heatmap at t=1..10, noise=5%/10%。
两次跑 (noise=0.05, 0.10)，dump 每段 σ，画 4 行 × 10 列 figure：
  row 1: noise=5% 重建 σ at t=1..10   (即 seg index 9,19,29,...,99)
  row 2: noise=5% 真值 σ at t=1..10
  row 3: noise=10% 重建 σ at t=1..10
  row 4: noise=10% 真值 σ at t=1..10
"""
import sys, time, json, pickle
from pathlib import Path
ROOT = Path('/data1/liulingfeng/cooperation/ghy/IDSM'); sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from src.mesh import generate_disk_mesh
from src.idsm_parabolic import (edp_cfg_example_5_1, c_func_example_5_1, v_func_example_5_1,
    run_idsm_parabolic)

OUT = ROOT/'logs'
fine = generate_disk_mesh(n_boundary=int(80*np.sqrt(2)))
coarse = generate_disk_mesh(n_boundary=80)

# coarse mesh 三角形重心 & triangulation
centers = (coarse.points[coarse.triangles[:,0]] +
           coarse.points[coarse.triangles[:,1]] +
           coarse.points[coarse.triangles[:,2]]) / 3.0
tri = mtri.Triangulation(coarse.points[:,0], coarse.points[:,1], coarse.triangles)

results = {}
for noise in [0.05, 0.10]:
    cfg = edp_cfg_example_5_1(noise=noise)
    cfg.total_time = 10.2
    cfg.tolerance = 0.10  # paper §5.1
    print(f"\n========== noise={noise} tol=0.10 102段 ==========")
    t0 = time.time()
    res = run_idsm_parabolic(coarse_mesh=coarse, fine_mesh=fine, cfg=cfg,
        c_func=c_func_example_5_1, v_func=v_func_example_5_1,
        seed=42, verbose=False)
    print(f"  耗时 {time.time()-t0:.0f}s, 段数 {len(res['sigma_history'])}")
    iou = np.array(res['iou_history'])
    print(f"  IoU max={iou.max():.4f}@{int(iou.argmax())} mean_last20={iou[-20:].mean():.4f}")
    # 保存关键段 σ for 复现图
    seg_idx = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]  # t=1..10
    sigma_keep = [res['sigma_history'][k].copy() for k in seg_idx]
    results[noise] = {'sigma': sigma_keep, 'segs': seg_idx, 'iou_max': float(iou.max()),
                       'iou_seq': iou.tolist()}

# pickle 中转
with open(OUT/'fig1_repro_data.pkl','wb') as f:
    pickle.dump({'noise_005': results[0.05], 'noise_010': results[0.10],
                 'centers': centers, 'tri_x': coarse.points[:,0],
                 'tri_y': coarse.points[:,1], 'tri_idx': coarse.triangles}, f)
print(f"\n  pickle -> {OUT/'fig1_repro_data.pkl'}")

# === 画图 ===
fig, axes = plt.subplots(4, 10, figsize=(28, 11))
cmap = 'viridis'  # σ ∈ [0.01, 1.0]
vmin, vmax = 0.0, 1.0
ts = list(range(1, 11))

for col, (k, t_eval) in enumerate(zip(results[0.05]['segs'], ts)):
    # row 0: noise=5% 重建
    ax = axes[0, col]
    sg = results[0.05]['sigma'][col]
    # 用三角形重心 → tripcolor
    ax.tripcolor(tri, facecolors=sg, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors='none')
    ax.set_xlim(-1.05,1.05); ax.set_ylim(-1.05,1.05); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    if col == 0: ax.set_ylabel(r'$\varepsilon=5\%$ recon', fontsize=11)
    ax.set_title(f't={t_eval}', fontsize=10)
    
    # row 1: noise=5% 真值 (=Example 5.1 真值 σ on centers)
    ax = axes[1, col]
    truth = c_func_example_5_1(t_eval, centers[:,0], centers[:,1], cfg)
    ax.tripcolor(tri, facecolors=truth, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors='none')
    ax.set_xlim(-1.05,1.05); ax.set_ylim(-1.05,1.05); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    if col == 0: ax.set_ylabel(r'$\varepsilon=5\%$ truth', fontsize=11)
    
    # row 2: noise=10% 重建
    ax = axes[2, col]
    sg10 = results[0.10]['sigma'][col]
    ax.tripcolor(tri, facecolors=sg10, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors='none')
    ax.set_xlim(-1.05,1.05); ax.set_ylim(-1.05,1.05); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    if col == 0: ax.set_ylabel(r'$\varepsilon=10\%$ recon', fontsize=11)
    
    # row 3: noise=10% 真值
    ax = axes[3, col]
    ax.tripcolor(tri, facecolors=truth, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors='none')
    ax.set_xlim(-1.05,1.05); ax.set_ylim(-1.05,1.05); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    if col == 0: ax.set_ylabel(r'$\varepsilon=10\%$ truth', fontsize=11)

plt.suptitle('Fig 1 reproduction: Example 5.1 (merging) — IDSM-BFG recon vs truth (cA=1.0,cB=0.1)', fontsize=13)
plt.tight_layout()
out_png = OUT/'fig1_repro.png'
plt.savefig(out_png, dpi=120, bbox_inches='tight')
plt.savefig(OUT/'fig1_repro.pdf', bbox_inches='tight')
print(f"\n-> {out_png}")
print(f"-> {OUT/'fig1_repro.pdf'}")
