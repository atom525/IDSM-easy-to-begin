"""Generate DSMDL reference-style forward MAT data using pure Python."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
import zipfile

import numpy as np
from scipy import io as sio

ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(ROOT))

from src.phaseless_reference import ForwardConfig, generate_forward_dataset  # noqa: E402


def _load_contrast_from_mat(path: Path) -> np.ndarray:
    data = sio.loadmat(path)
    for key in ("contrast", "Contrast"):
        if key in data:
            arr = np.asarray(data[key])
            if arr.ndim == 3:
                return arr.astype(np.float32)
    # fallback: search for first (N,64,64) tensor
    for key, value in data.items():
        if key.startswith("__"):
            continue
        arr = np.asarray(value)
        if arr.ndim == 3 and arr.shape[1:] == (64, 64):
            return arr.astype(np.float32)
    raise KeyError("Could not find 3D contrast tensor in MAT file")


def _load_contrast(path: Path) -> np.ndarray:
    if path.suffix.lower() != ".zip":
        return _load_contrast_from_mat(path)
    with zipfile.ZipFile(path) as zf:
        mat_name = "ISP_forward/Mnist64.mat"
        with zf.open(mat_name) as fh:
            data = sio.loadmat(BytesIO(fh.read()))
    for key in ("contrast", "Contrast"):
        if key in data:
            arr = np.asarray(data[key])
            if arr.ndim == 3:
                return arr.astype(np.float32)
    raise KeyError("Mnist64.mat does not contain `contrast`/`Contrast` tensor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pure Python port of DataMnist.m")
    parser.add_argument(
        "--contrast-source",
        type=str,
        default=str(ROOT / "reference" / "ISP_forward.zip"),
        help="Path to ISP_forward.zip or Mnist64.mat",
    )
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "data" / "phaseless" / "Phaseless_MnistRotaCir_py.mat"),
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    source = Path(args.contrast_source)
    contrast = _load_contrast(source)
    n = min(int(args.n_samples), int(contrast.shape[0]))
    if n <= 0:
        raise ValueError("n-samples must be positive")
    data = generate_forward_dataset(
        contrast[:n],
        cfg=ForwardConfig(),
        seed=args.seed,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(
        out_path,
        {
            "Contrast": data["Contrast"],
            "E_i": data["E_i"],
            "E_s": data["E_s"],
            "R_mat": data["R_mat"],
        },
    )
    print(f"[ok] wrote {out_path} with {n} samples")
    return out_path


if __name__ == "__main__":
    run(parse_args())
