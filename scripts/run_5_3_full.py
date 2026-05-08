"""严格复刻 Paper 2 §5.3：total_time=10.0, 100段, paper_cfg."""
import sys, json, time
from pathlib import Path
ROOT = Path('/data1/liulingfeng/cooperation/ghy/IDSM')
sys.path.insert(0, str(ROOT))

import numpy as np
from src.mesh import generate_disk_mesh
from src.idsm_parabolic import (
    edp_cfg_example_5_3,
    paper_cfg_example_5_3, c_func_example_5_3, v_func_example_5_3,
    run_idsm_parabolic, u_func_example_5_3,
)

import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--total-time', type=float, default=10.0)
ap.add_argument('--noise', type=float, default=0.05)
ap.add_argument('--mode', default='paper', choices=['paper','edp'])
ap.add_argument('--out', default='results/parabolic_fixed/ex_5_3_paper.npz')
args = ap.parse_args()

cfg = (paper_cfg_example_5_3 if args.mode=="paper" else edp_cfg_example_5_3)(noise=args.noise)
cfg.total_time = args.total_time
print(f'=== Ex 5.3 Nonlinear |y|y·U paper_cfg total_time={cfg.total_time} ===')
print(f'  cA={cfg.cA} vA={cfg.vA} vB={cfg.vB} (uA=vA uB=vB) noise={cfg.noise_level}')
print(f'  max_inner={cfg.max_inner} forget={cfg.forget_scale} tol={cfg.tolerance} lowrank={cfg.lowrank}')

n_b_fine = int(80*np.sqrt(2)); n_b_coarse = 80
fine = generate_disk_mesh(n_boundary=n_b_fine)
coarse = generate_disk_mesh(n_boundary=n_b_coarse)
print(f'  mesh: fine {fine.n_triangles} tri / coarse {coarse.n_triangles} tri')

t0 = time.time()
res = run_idsm_parabolic(
    coarse_mesh=coarse, fine_mesh=fine, cfg=cfg,
    c_func=c_func_example_5_3, v_func=v_func_example_5_3,
    seed=42, verbose=True,
)
elapsed = time.time() - t0
print(f'\n=== done in {elapsed/60:.1f} min ===')

iou = np.array(res['iou_history'])
ni = np.array(res['n_inner_per_segment'])
print(f'segments      : {len(iou)}')
print(f'IoU mean      : {iou.mean():.4f}')
print(f'IoU max       : {iou.max():.4f} @ seg {int(iou.argmax())}')
print(f'IoU final     : {iou[-1]:.4f}')
print(f'IoU mean last20: {iou[-20:].mean():.4f}')
print(f'segs IoU>=0.3 : {int((iou>=0.3).sum())}/{len(iou)}')
print(f'segs IoU>=0.5 : {int((iou>=0.5).sum())}/{len(iou)}')
print(f'n_inner mean  : {ni.mean():.2f} (paper Table 1 says 7.46)')

centers = (coarse.points[coarse.triangles[:,0]]+coarse.points[coarse.triangles[:,1]]+coarse.points[coarse.triangles[:,2]])/3.0

out = ROOT / args.out
out.parent.mkdir(parents=True, exist_ok=True)
cfg_dict = {f'cfg_{k}': getattr(cfg, k) for k in
    ('cA','cB','vA','vB','model','total_time','forward_dt','delta_t','delta_t_split',
     'n_solve','n_coarse','save_num','tolerance','forget_scale','noise_level',
     'lowrank','data_num','kappa','max_inner')}
np.savez_compressed(out,
    sigma_history=np.stack(res['sigma_history']),
    v_history=np.stack(res['v_history']),
    y_quote_history=np.stack(res['y_quote_history']),
    residuals_per_segment=np.array([np.array(r) for r in res['residuals_per_segment']], dtype=object),
    n_inner_per_segment=ni,
    iou_history=iou,
    coarse_points=coarse.points,
    coarse_triangles=coarse.triangles,
    coarse_areas=coarse.areas,
    coarse_centroids=centers,
    runtime_seconds=elapsed,
    noise_level=cfg.noise_level,
    example_id='5.3',
    mode=args.mode,
    **cfg_dict,
)
print(f'-> {out}')
