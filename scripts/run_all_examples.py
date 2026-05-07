"""统一驱动 Example 5.1-5.5 的 paper §5 复现 runner。

调用约定（与 src.idsm_parabolic.run_idsm_parabolic 对齐）：
    run_idsm_parabolic(coarse_mesh, fine_mesh, cfg, c_func, v_func, seed, verbose)

输出 results/parabolic/ex_5_{X}_{paper|edp}.npz，字段：
    sigma_history, v_history, y_quote_history,
    residuals_per_segment, n_inner_per_segment, iou_history,
    coarse_points, coarse_triangles, coarse_areas, coarse_centroids,
    cfg_*（持久化标量），noise_level, elapsed_sec

Ex 5.3 (Nonlinear) 正问题已用 Newton+CN 复刻 .edp 的 |y|y·U 三次项；反演端
仍是 (σ,V) 线性 IDSM, 因此 IoU 与 σ trivially constant 比对约为 0（U-recovery
inversion 见 task #71）。
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path('/data1/liulingfeng/cooperation/ghy/IDSM')
sys.path.insert(0, str(ROOT))
import numpy as np

from src.mesh import generate_disk_mesh
from src.idsm_parabolic import (
    run_idsm_parabolic,
    edp_cfg_example_5_1, edp_cfg_example_5_2, edp_cfg_example_5_3,
    edp_cfg_example_5_4, edp_cfg_example_5_5,
    paper_cfg_example_5_1, paper_cfg_example_5_2, paper_cfg_example_5_3,
    paper_cfg_example_5_4, paper_cfg_example_5_5,
    c_func_example_5_1, c_func_example_5_2, c_func_example_5_3,
    c_func_example_5_4, c_func_example_5_5,
    v_func_example_5_1, v_func_example_5_2, v_func_example_5_3,
    v_func_example_5_4, v_func_example_5_5,
)


EXAMPLES = {
    '5.1': dict(
        name='ConductivityMerging',
        edp_cfg=edp_cfg_example_5_1, paper_cfg=paper_cfg_example_5_1,
        c_func=c_func_example_5_1, v_func=v_func_example_5_1,
    ),
    '5.2': dict(
        name='MixedMoving',
        edp_cfg=edp_cfg_example_5_2, paper_cfg=paper_cfg_example_5_2,
        c_func=c_func_example_5_2, v_func=v_func_example_5_2,
    ),
    '5.3': dict(
        name='Nonlinear',
        edp_cfg=edp_cfg_example_5_3, paper_cfg=paper_cfg_example_5_3,
        c_func=c_func_example_5_3, v_func=v_func_example_5_3,
    ),
    '5.4': dict(
        name='PotentialFading',
        edp_cfg=edp_cfg_example_5_4, paper_cfg=paper_cfg_example_5_4,
        c_func=c_func_example_5_4, v_func=v_func_example_5_4,
    ),
    '5.5': dict(
        name='ConductivityDiminishing',
        edp_cfg=edp_cfg_example_5_5, paper_cfg=paper_cfg_example_5_5,
        c_func=c_func_example_5_5, v_func=v_func_example_5_5,
    ),
}


def cfg_to_dict(cfg):
    """ParabolicConfig → 标量 dict，可放入 npz."""
    out = {}
    for k in ('cA', 'cB', 'vA', 'vB', 'model', 'total_time', 'forward_dt',
             'delta_t', 'delta_t_split', 'n_solve', 'n_coarse', 'save_num',
             'tolerance', 'forget_scale', 'noise_level', 'lowrank',
             'data_num', 'kappa', 'max_inner'):
        out[f'cfg_{k}'] = getattr(cfg, k)
    return out


def run_single(example_id: str, mode: str, noise: float, out_dir: Path,
               total_time_override: float = None, seed: int = 42,
               verbose: bool = True):
    spec = EXAMPLES[example_id]
    if mode == 'paper':
        cfg = spec['paper_cfg'](noise=noise)
    else:
        cfg = spec['edp_cfg'](noise=noise)
    if total_time_override is not None:
        cfg.total_time = total_time_override

    n_b_fine = int(80 * np.sqrt(2))
    n_b_coarse = 80
    print(f"\n[{example_id} {spec['name']} | {mode} | noise={noise}] "
          f"total_time={cfg.total_time:.2f} tol={cfg.tolerance} "
          f"max_inner={cfg.max_inner} forget={cfg.forget_scale} lowrank={cfg.lowrank}")
    fine = generate_disk_mesh(n_boundary=n_b_fine)
    coarse = generate_disk_mesh(n_boundary=n_b_coarse)
    print(f"  mesh: fine {fine.n_triangles} tri / coarse {coarse.n_triangles} tri")

    t0 = time.time()
    res = run_idsm_parabolic(
        coarse_mesh=coarse, fine_mesh=fine, cfg=cfg,
        c_func=spec['c_func'], v_func=spec['v_func'],
        seed=seed, verbose=verbose,
    )
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s, segments={len(res['sigma_history'])}")

    iou = np.array(res['iou_history'])
    ni = np.array(res['n_inner_per_segment'])
    print(f"  IoU max={iou.max():.4f}@{int(iou.argmax())} final={iou[-1]:.4f} "
          f"mean_last20={iou[-20:].mean():.4f}  n_inner mean={ni.mean():.2f}")

    centers = (coarse.points[coarse.triangles[:, 0]]
               + coarse.points[coarse.triangles[:, 1]]
               + coarse.points[coarse.triangles[:, 2]]) / 3.0
    npz_path = out_dir / f'ex_{example_id.replace(".","_")}_{mode}.npz'
    np.savez_compressed(
        npz_path,
        sigma_history=np.stack(res['sigma_history']),
        v_history=np.stack(res['v_history']),
        y_quote_history=np.stack(res['y_quote_history']),
        residuals_per_segment=np.array([np.array(r) for r in res['residuals_per_segment']],
                                       dtype=object),
        n_inner_per_segment=np.array(res['n_inner_per_segment']),
        iou_history=iou,
        coarse_points=coarse.points,
        coarse_triangles=coarse.triangles,
        coarse_areas=coarse.areas,
        coarse_centroids=centers,
        runtime_seconds=elapsed,
        noise_level=cfg.noise_level,
        example_id=example_id,
        mode=mode,
        **cfg_to_dict(cfg),
    )
    print(f"  → {npz_path}")
    return res, cfg, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--example', default='all',
                        choices=['all', '5.1', '5.2', '5.3', '5.4', '5.5'])
    parser.add_argument('--mode', default='both', choices=['both', 'paper', 'edp'])
    parser.add_argument('--noise', type=float, default=None,
                        help='覆盖默认 noise（默认按 paper=0.05/edp=0.05 或 5.1 .edp=0.20）')
    parser.add_argument('--total-time', type=float, default=None,
                        help='覆盖 total_time（默认按 .edp 各自设置）')
    parser.add_argument('--out-dir', default='results/parabolic')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--include-ex51-n10', action='store_true',
                        help='额外跑 Ex 5.1 noise=10% paper-cfg')
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = ['5.1', '5.2', '5.3', '5.4', '5.5'] if args.example == 'all' else [args.example]
    modes = ['paper', 'edp'] if args.mode == 'both' else [args.mode]

    summary = {}
    t_all = time.time()
    for eid in examples:
        for m in modes:
            # 默认 noise：paper=0.05, edp 5.1=0.20 其余=0.05
            if args.noise is not None:
                noise = args.noise
            elif m == 'paper':
                noise = 0.05
            else:
                noise = 0.20 if eid == '5.1' else 0.05
            try:
                res, cfg, elapsed = run_single(
                    eid, m, noise=noise, out_dir=out_dir,
                    total_time_override=args.total_time,
                    verbose=not args.quiet,
                )
                iou = np.array(res['iou_history'])
                summary[f'ex_{eid}_{m}'] = dict(
                    noise=noise, total_time=cfg.total_time,
                    iou_max=float(iou.max()), iou_argmax=int(iou.argmax()),
                    iou_final=float(iou[-1]),
                    iou_mean_last20=float(iou[-20:].mean()),
                    segments=int(len(res['sigma_history'])),
                    elapsed_sec=float(elapsed),
                )
            except Exception as e:
                print(f"  [ERROR] {eid} {m}: {e}")
                summary[f'ex_{eid}_{m}'] = dict(error=str(e), noise=noise)

    if args.include_ex51_n10:
        try:
            from src.idsm_parabolic import paper_cfg_example_5_1 as pcfg51
            spec = EXAMPLES['5.1']
            cfg = pcfg51(noise=0.10)
            if args.total_time is not None:
                cfg.total_time = args.total_time
            n_b_fine = int(80 * np.sqrt(2)); n_b_coarse = 80
            print(f"\n[5.1 {spec['name']} | paper | noise=0.10] total_time={cfg.total_time}")
            fine = generate_disk_mesh(n_boundary=n_b_fine)
            coarse = generate_disk_mesh(n_boundary=n_b_coarse)
            t0 = time.time()
            res = run_idsm_parabolic(
                coarse_mesh=coarse, fine_mesh=fine, cfg=cfg,
                c_func=spec['c_func'], v_func=spec['v_func'],
                seed=42, verbose=not args.quiet,
            )
            elapsed = time.time() - t0
            iou = np.array(res['iou_history'])
            centers = (coarse.points[coarse.triangles[:, 0]]
                       + coarse.points[coarse.triangles[:, 1]]
                       + coarse.points[coarse.triangles[:, 2]]) / 3.0
            np.savez_compressed(
                out_dir / 'ex_5_1_n10_paper.npz',
                sigma_history=np.stack(res['sigma_history']),
                v_history=np.stack(res['v_history']),
                y_quote_history=np.stack(res['y_quote_history']),
                residuals_per_segment=np.array(
                    [np.array(r) for r in res['residuals_per_segment']], dtype=object),
                n_inner_per_segment=np.array(res['n_inner_per_segment']),
                iou_history=iou,
                coarse_points=coarse.points,
                coarse_triangles=coarse.triangles,
                coarse_areas=coarse.areas,
                coarse_centroids=centers,
                runtime_seconds=elapsed,
                noise_level=0.10,
                example_id='5.1', mode='paper_n10',
                **cfg_to_dict(cfg),
            )
            summary['ex_5_1_n10_paper'] = dict(
                noise=0.10, total_time=cfg.total_time,
                iou_max=float(iou.max()), iou_final=float(iou[-1]),
                segments=int(len(res['sigma_history'])),
                elapsed_sec=float(elapsed),
            )
            print(f"  → ex_5_1_n10_paper.npz   IoU max={iou.max():.4f}")
        except Exception as e:
            print(f"  [ERROR] ex_5_1_n10: {e}")
            summary['ex_5_1_n10_paper'] = dict(error=str(e))

    print(f"\n=========== ALL DONE in {time.time()-t_all:.1f}s ===========")
    json.dump(summary, open(ROOT / 'logs' / 'run_all_examples_summary.json', 'w'),
              indent=2, default=str)
    print(f"summary -> logs/run_all_examples_summary.json")


if __name__ == '__main__':
    main()
