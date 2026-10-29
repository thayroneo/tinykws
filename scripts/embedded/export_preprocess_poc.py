"""Export a fixed MFCC and normalization vector for the MicroPython check."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def export(features: Path, output_dir: Path, index: int) -> dict:
    archive = np.load(features, allow_pickle=False)
    normalized = archive["x"][index].astype(np.float32)
    mean = archive["mean"].astype(np.float32)
    std = archive["std"].astype(np.float32)
    raw = normalized * std[None, :] + mean[None, :]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mfcc_raw.f32").write_bytes(raw.reshape(-1).astype("<f4").tobytes())
    (output_dir / "mfcc_mean.f32").write_bytes(mean.astype("<f4").tobytes())
    (output_dir / "mfcc_std.f32").write_bytes(std.astype("<f4").tobytes())
    reference = {"index": index, "shape": [40, 13], "normalized": normalized.reshape(-1).tolist(), "tolerance_abs": 1e-5}
    (output_dir / "preprocess_reference.json").write_text(json.dumps(reference) + "\n", encoding="utf-8")
    return {"output_dir": str(output_dir), "raw_bytes": (output_dir / "mfcc_raw.f32").stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("data/processed/features.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/embedded/mlp_poc"))
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(export(args.features, args.output_dir, args.index)))


if __name__ == "__main__":
    main()
