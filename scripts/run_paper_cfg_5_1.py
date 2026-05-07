"""paper_cfg_example_5_1 (noise=0.05, max_inner=5, forget=0.6, tol=0.10) 全 102 段。"""
import sys, json, time
from pathlib import Path
ROOT = Path('/data1/liulingfeng/cooperation/ghy/IDSM')
sys.path.insert(0, str(ROOT))

import numpy as np
from src.mesh import generate_disk_mesh
from src.idsm_parabolic import (
    paper_cfg_example_5_1, c_func_example_5_1, v_func_example_5_1,
    run_idsm_parabolic,
)

cfg = paper_cfg_example_5_1(noise=0.05)
cfg.total_time = 10.2
print(f"=== paper_cfg 5_1 noise=0.05 max_inner={cfg.max_inner} forget={cfg.forget_scale} tol={cfg.tolerance} ===")

fine = generate_disk_mesh(n_boundary=int(80*np.sqrt(2)))
coarse = generate_disk_mesh(n_boundary=80)
print(f"mesh: fine {fine.n_triangles} tri / coarse {coarse.n_triangles} tri")

t0 = time.time()
res = run_idsm_parabolic(
    coarse_mesh=coarse, fine_mesh=fine, cfg=cfg,
    c_func=c_func_example_5_1, v_func=v_func_example_5_1,
    seed=42, verbose=False,
)
elapsed = time.time()-t0
print(f"\n=== run done in {elapsed:.1f}s ===")

iou_h = res['iou_history']
sigma_h = res['sigma_history']
n_inner_h = [len(r) for r in res['residuals_per_segment']]
n_done = len(iou_h)
iou_arr = np.asarray(iou_h)

print(f"\n=== summary paper_cfg 5_1 noise=0.05 ===")
print(f"segments done    : {n_done}/102")
print(f"IoU mean (all)   : {iou_arr.mean():.4f}")
print(f"IoU max          : {iou_arr.max():.4f} @ seg {int(iou_arr.argmax())}")
print(f"IoU final        : {iou_arr[-1]:.4f}")
print(f"IoU mean last 20 : {iou_arr[-20:].mean():.4f}")
print(f"IoU mean last 50 : {iou_arr[-50:].mean():.4f}")
print(f"segs IoU>=0.3    : {int((iou_arr>=0.3).sum())}/{n_done}")
print(f"segs IoU>=0.5    : {int((iou_arr>=0.5).sum())}/{n_done}")
print(f"n_inner mean     : {np.mean(n_inner_h):.2f}  (table1 paper says 1.20)")

out = ROOT/'logs'/'iou_paper_cfg_5_1_n0.05.json'
json.dump({
    'noise': 0.05, 'cfg': 'paper_cfg_5_1', 'segs': n_done,
    'iou': [float(v) for v in iou_h],
    'n_inner': [int(v) for v in n_inner_h],
    'iou_mean': float(iou_arr.mean()),
    'iou_max': float(iou_arr.max()),
    'iou_argmax': int(iou_arr.argmax()),
    'iou_final': float(iou_arr[-1]),
    'iou_mean_last20': float(iou_arr[-20:].mean()),
    'iou_mean_last50': float(iou_arr[-50:].mean()),
    'segs_ge_03': int((iou_arr>=0.3).sum()),
    'segs_ge_05': int((iou_arr>=0.5).sum()),
    'n_inner_mean': float(np.mean(n_inner_h)),
    'elapsed_sec': elapsed,
}, open(out,'w'), indent=2)
print(f"-> {out}")

txt = ROOT/'logs'/'full_102_paper_cfg_5_1.txt'
with open(txt,'w') as f:
    f.write("seg t sigma_min sigma_max IoU n_inner\n")
    for k in range(n_done):
        sg = sigma_h[k]
        f.write(f"{k:3d} {(k+1)*cfg.delta_t:5.2f} {sg.min():.4f} {sg.max():.4f} "
                f"{iou_h[k]:.4f} {n_inner_h[k]:3d}\n")
print(f"-> {txt}")
