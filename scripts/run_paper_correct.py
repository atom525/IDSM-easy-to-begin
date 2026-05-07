"""按 paper §5.1 的真正 setting 跑 102 段：noise=5%, tol=10%."""
import sys, json, time
from pathlib import Path
ROOT = Path('/data1/liulingfeng/cooperation/ghy/IDSM')
sys.path.insert(0, str(ROOT))
import numpy as np
from src.mesh import generate_disk_mesh
from src.idsm_parabolic import (edp_cfg_example_5_1, c_func_example_5_1, v_func_example_5_1, run_idsm_parabolic)

fine = generate_disk_mesh(n_boundary=int(80*np.sqrt(2)))
coarse = generate_disk_mesh(n_boundary=80)
print(f"mesh: fine {fine.n_triangles} / coarse {coarse.n_triangles}")
results = {}
for noise in [0.05, 0.10]:
    cfg = edp_cfg_example_5_1(noise=noise)
    cfg.total_time = 10.2
    cfg.tolerance = 0.10
    print(f"\n========== noise={noise} tol=0.10 (paper §5.1) ==========")
    t0 = time.time()
    res = run_idsm_parabolic(coarse_mesh=coarse, fine_mesh=fine, cfg=cfg,
        c_func=c_func_example_5_1, v_func=v_func_example_5_1,
        seed=42, verbose=False)
    elapsed = time.time()-t0
    iou = np.array(res['iou_history'])
    ni = np.array([len(r) for r in res['residuals_per_segment']])
    fr = np.array([float(r[-1]) for r in res['residuals_per_segment']])
    sh = res['sigma_history']
    print(f"用时 {elapsed:.0f}s")
    print(f"  IoU max={iou.max():.4f} @{int(iou.argmax())}  final={iou[-1]:.4f}")
    print(f"      mean_last20={iou[-20:].mean():.4f}  mean_last50={iou[-50:].mean():.4f}")
    print(f"      >=0.5: {int((iou>=0.5).sum())}/102  >=0.7: {int((iou>=0.7).sum())}/102")
    print(f"  n_inner mean={ni.mean():.2f} max={int(ni.max())} min={int(ni.min())}")
    print(f"  resid mean={fr.mean():.4f}  resid<tol={int((fr<cfg.tolerance).sum())}/102")
    print(f"  σ_min饱和(0.01)={int(sum(s.min()<=0.0101 for s in sh))}/102")
    results[f"noise_{noise}"] = {
        'noise': noise, 'tol': cfg.tolerance,
        'iou_max': float(iou.max()), 'iou_argmax': int(iou.argmax()),
        'iou_final': float(iou[-1]),
        'iou_mean_last20': float(iou[-20:].mean()),
        'iou_mean_last50': float(iou[-50:].mean()),
        'segs_ge_05': int((iou>=0.5).sum()),
        'segs_ge_07': int((iou>=0.7).sum()),
        'n_inner_mean': float(ni.mean()),
        'n_inner_max': int(ni.max()),
        'resid_mean': float(fr.mean()),
        'resid_below_tol': int((fr<cfg.tolerance).sum()),
        'sigma_min_saturated': int(sum(s.min()<=0.0101 for s in sh)),
        'iou_seq': iou.tolist(), 'n_inner_seq': ni.tolist(), 'resid_seq': fr.tolist(),
        'elapsed_sec': elapsed,
    }

out = ROOT/'logs'/'iou_paper_correct.json'
json.dump(results, open(out,'w'), indent=2)
print(f"\n-> {out}")
