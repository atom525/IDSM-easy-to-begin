"""Run and cache the Python parabolic IDSM examples.

The caches store reconstruction histories, residual traces, IoU values, mesh
geometry, and configuration scalars so figures can be regenerated without
rerunning the PDE solves.

Ex 5.3 (Nonlinear) uses the dedicated U-recovery branch in
``src.idsm_parabolic``: Newton-Crank-Nicolson forward solves and a single-field
P0 reconstruction for the coefficient U.
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from src.mesh import generate_disk_mesh, generate_disk_mesh_paper
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


def make_meshes(cfg, mesh_mode: str):
    """Build the FreeFEM-style ThFine / Th / ThCoarse mesh triplet."""
    if mesh_mode == 'edp':
        data = generate_disk_mesh(n_boundary=max(8, int(cfg.n_solve * np.sqrt(2))))
        solve = generate_disk_mesh(n_boundary=cfg.n_solve)
        coeff = generate_disk_mesh(n_boundary=cfg.n_coarse)
    elif mesh_mode == 'paper':
        data = generate_disk_mesh_paper(target_triangles=13870)
        solve = generate_disk_mesh_paper(target_triangles=7002)
        coeff = generate_disk_mesh_paper(target_triangles=1120)
    elif mesh_mode == 'legacy':
        data = generate_disk_mesh_paper(target_triangles=7002)
        solve = generate_disk_mesh_paper(target_triangles=1120)
        coeff = solve
    else:
        raise ValueError(f'unknown mesh_mode={mesh_mode!r}')
    return data, solve, coeff


def cfg_to_dict(cfg):
    """Serialize scalar configuration fields into an npz-friendly dict."""
    out = {}
    for k in ('cA', 'cB', 'vA', 'vB', 'model', 'total_time', 'forward_dt',
             'delta_t', 'delta_t_split', 'n_solve', 'n_coarse', 'save_num',
             'tolerance', 'forget_scale', 'noise_level', 'lowrank',
             'data_num', 'kappa', 'max_inner'):
        out[f'cfg_{k}'] = getattr(cfg, k)
    return out


def run_single(example_id: str, mode: str, noise: float, out_dir: Path,
               total_time_override: float = None, seed: int = 42,
               verbose: bool = True, mesh_mode: str = 'edp'):
    spec = EXAMPLES[example_id]
    if mode == 'paper':
        cfg = spec['paper_cfg'](noise=noise)
    else:
        cfg = spec['edp_cfg'](noise=noise)
    if total_time_override is not None:
        cfg.total_time = total_time_override
    elif example_id == '5.3':
        # The FreeFEM script keeps a short default for quick debugging, but the
        # paper discussion and figures track the nonlinear inclusion over a
        # long horizon.  Use the paper-scale horizon for the canonical runner.
        cfg.total_time = 10.0

    print(f"\n[{example_id} {spec['name']} | {mode} | noise={noise}] "
          f"total_time={cfg.total_time:.2f} tol={cfg.tolerance} "
          f"max_inner={cfg.max_inner} forget={cfg.forget_scale} lowrank={cfg.lowrank} "
          f"mesh_mode={mesh_mode}")
    data_mesh, solve_mesh, coeff_mesh = make_meshes(cfg, mesh_mode)
    print(
        f"  mesh: data {data_mesh.n_triangles} tri / "
        f"solve {solve_mesh.n_triangles} tri / coeff {coeff_mesh.n_triangles} tri"
    )

    t0 = time.time()
    res = run_idsm_parabolic(
        coarse_mesh=coeff_mesh, fine_mesh=data_mesh, solve_mesh=solve_mesh, cfg=cfg,
        c_func=spec['c_func'], v_func=spec['v_func'],
        seed=seed, verbose=verbose,
    )
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s, segments={len(res['sigma_history'])}")

    iou = np.array(res['iou_history'])
    ni = np.array(res['n_inner_per_segment'])
    print(f"  IoU max={iou.max():.4f}@{int(iou.argmax())} final={iou[-1]:.4f} "
          f"mean_last20={iou[-20:].mean():.4f}  n_inner mean={ni.mean():.2f}")

    centers = (coeff_mesh.points[coeff_mesh.triangles[:, 0]]
               + coeff_mesh.points[coeff_mesh.triangles[:, 1]]
               + coeff_mesh.points[coeff_mesh.triangles[:, 2]]) / 3.0
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
        coarse_points=coeff_mesh.points,
        coarse_triangles=coeff_mesh.triangles,
        coarse_areas=coeff_mesh.areas,
        coarse_centroids=centers,
        solve_points=solve_mesh.points,
        solve_triangles=solve_mesh.triangles,
        mesh_mode=mesh_mode,
        data_triangles=data_mesh.n_triangles,
        solve_triangles_count=solve_mesh.n_triangles,
        coeff_triangles_count=coeff_mesh.n_triangles,
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
                        help='Override the default noise level.')
    parser.add_argument('--total-time', type=float, default=None,
                        help='Override the example total_time.')
    parser.add_argument('--out-dir', default='results/parabolic')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--mesh-mode', default='edp', choices=['edp', 'paper', 'legacy'],
                        help='edp: ThFine/Th/ThCoarse from nSolve; paper: 13870/7002/1120; legacy: old two-mesh fast path.')
    parser.add_argument('--include-ex51-n10', action='store_true',
                        help='Also run Example 5.1 with 10% noise in paper mode.')
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = ['5.1', '5.2', '5.3', '5.4', '5.5'] if args.example == 'all' else [args.example]
    modes = ['paper', 'edp'] if args.mode == 'both' else [args.mode]

    summary = {}
    t_all = time.time()
    for eid in examples:
        for m in modes:
            if args.noise is not None:
                noise = args.noise
            elif m == 'paper':
                noise = 0.05
            else:
                noise = 0.20 if eid == '5.1' else 0.05
            try:
                res, cfg, elapsed = run_single