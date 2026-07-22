"""Reproduce DSMDL_phaseless training in pure Python.

This script ports ``reference/DSMDL_phaseless/main.py`` into a modular run:

1) load the reference MAT dataset from zip or extracted path,
2) compute DSM indicator inputs (with the same scaling and noise model),
3) train U_Net3Ab,
4) export checkpoint + summary + sample reconstruction figure.
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import io as sio
import torch

ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(ROOT))

from src.phaseless_reference import (  # noqa: E402
    TrainingConfigLegacy,
    compute_inputs_from_mat,
    load_mat_dataset,
    relative_l2_legacy,
    save_checkpoint,
    save_summary,
    train_unet3ab,
)


def _load_mat_from_zip(zip_path: Path) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(zip_path) as zf:
        mat_name = "DSMDL_phaseless/data/Phaseless_MnistRotaCir_12000PS2.mat"
        with zf.open(mat_name) as fh:
            data = sio.loadmat(BytesIO(fh.read()))
    return {
        "Contrast": np.asarray(data["Contrast"], dtype=np.float32),
        "E_i": np.asarray(data["E_i"], dtype=np.complex64),
        "E_s": np.asarray(data["E_s"], dtype=np.complex64),
        "R_mat": np.asarray(data["R_mat"], dtype=np.complex64),
    }


def _render_samples(
    *,
    model: torch.nn.Module,
    x_np: np.ndarray,
    y_np: np.ndarray,
    out_path: Path,
    device: str,
    test_slice: slice,
) -> None:
    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()
    xs = torch.from_numpy(x_np[test_slice]).to(dev)
    ys = y_np[test_slice]
    with torch.no_grad():
        pred = model(xs).detach().cpu().numpy()[:, 0]
    n_show = min(5, pred.shape[0])
    fig, axes = plt.subplots(n_show, 2, figsize=(7, 2.6 * n_show), squeeze=False)
    for i in range(n_show):
        axes[i, 0].imshow(pred[i], origin="lower", cmap="viridis", vmin=1.0, vmax=1.7)
        axes[i, 0].set_title(f"recon #{i+1}")
        axes[i, 1].imshow(ys[i], origin="lower", cmap="viridis", vmin=1.0, vmax=1.7)
        axes[i, 1].set_title(f"truth #{i+1}")
        axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])
        axes[i, 1].set_xticks([]); axes[i, 1].set_yticks([])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DSMDL reference-style reproduction.")
    parser.add_argument(
        "--mat-path",
        type=str,
        default=str(ROOT / "reference" / "DSMDL_phaseless.zip"),
        help="Path to MAT file or DSMDL_phaseless.zip",
    )
    parser.add_argument("--n-incident", type=int, default=16)
    parser.add_argument("--noise-level", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-number", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    mat_path = Path(args.mat_path)
    if mat_path.suffix.lower() == ".zip":
        mat_data = _load_mat_from_zip(mat_path)
    else:
        mat_data = load_mat_dataset(mat_path)

    indicators = compute_inputs_from_mat(
        mat_data,
        n_incident=args.n_incident,
        noise_level=args.noise_level,
        seed=args.seed,
    )
    x_noisy = np.asarray(indicators["inputs_noisy"], dtype=np.float32)
    y = np.asarray(mat_data["Contrast"], dtype=np.float32)

    train_cfg = TrainingConfigLegacy(
        n_incident=args.n_incident,
        noise_level=args.noise_level,
        epochs=args.epochs,
        batch_number=args.batch_number,
        seed=args.seed,
        device=args.device,
    )
    result = train_unet3ab(inputs_noisy=x_noisy, contrast=y, cfg=train_cfg)
    model = result["model"]

    test_slice = slice(train_cfg.test_start, train_cfg.test_stop)
    test_slice = slice(min(test_slice.start, y.shape[0]), min(test_slice.stop, y.shape[0]))
    dev = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        pred_test = model(torch.from_numpy(x_noisy[test_slice]).to(dev)).detach().cpu()
    y_test = torch.from_numpy(y[test_slice, None, :, :])
    test_rel_l2 = relative_l2_legacy(pred_test, y_test)

    ckpt_dir = ROOT / "results" / "phaseless" / "checkpoints"
    ckpt_path = ckpt_dir / f"reference_UNETCircle_Ni{args.n_incident}.pt"
    save_checkpoint(
        ckpt_path,
        model=model,
        train_cfg=train_cfg,
        extra_meta={
            "best_test_rel_l2": float(result["best_test_rel_l2"]),
            "indices": indicators["indices"],
            "norm_gs": indicators["norm_gs"],
        },
    )

    out_fig = ROOT / "figures" / "06_phaseless" / "06_reference_mnist.png"
    _render_samples(
        model=model,
        x_np=x_noisy,
        y_np=y,
        out_path=out_fig,
        device=args.device,
        test_slice=test_slice,
    )

    summary = {
        "mode": "reference_dsmdl_python_port",
        "seed": args.seed,
        "device": str(dev),
        "n_incident": args.n_incident,
        "noise_level": args.noise_level,
        "epochs": args.epochs,
        "batch_number": args.batch_number,
        "mm_scale": indicators["mm_scale"],
        "best_test_rel_l2": float(result["best_test_rel_l2"]),
        "final_test_rel_l2": float(test_rel_l2),
        "checkpoint": str(ckpt_path.relative_to(ROOT)),
        "figure": str(out_fig.relative_to(ROOT)),
        "history": result["history"],
    }
    out_summary = ROOT / "results" / "phaseless" / "reference_summary.json"
    save_summary(out_summary, summary)
    md_lines = [
        "# Reference DSMDL Python Port",
        "",
        f"- device: `{summary['device']}`",
        f"- n_incident: `{summary['n_incident']}`",
        f"- noise_level: `{summary['noise_level']}`",
        f"- best_test_rel_l2: `{summary['best_test_rel_l2']:.6f}`",
        f"- final_test_rel_l2: `{summary['final_test_rel_l2']:.6f}`",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- figure: `{summary['figure']}`",
    ]
    (ROOT / "results" / "phaseless" / "reference_comparison.md").write_text(
        "\n".join(md_lines),
        encoding="utf-8",
    )
    print(f"[ok] wrote {out_summary}")
    print(f"[ok] wrote {ckpt_path}")
    print(f"[ok] wrote {out_fig}")
    return summary


if __name__ == "__main__":
    run(parse_args())
