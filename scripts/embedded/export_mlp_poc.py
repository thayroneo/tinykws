"""Export an MLP checkpoint and a deterministic vector for MicroPython POC."""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch

try:
    from scripts.training.train_mlp import LABELS, MLP
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.training.train_mlp import LABELS, MLP


MAGIC = b"VTKMLP1\0"


def export(checkpoint: Path, features: Path, output_dir: Path, index: int) -> dict:
    archive = np.load(features, allow_pickle=False)
    vector = archive["x"][index].astype(np.float32).reshape(-1)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = MLP()
    model.load_state_dict(saved["model_state"])
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(vector[None, :]))[0].numpy().astype(np.float32)
    layers = [
        (model.net[1].weight, model.net[1].bias),
        (model.net[3].weight, model.net[3].bias),
        (model.net[5].weight, model.net[5].bias),
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "mlp_fp32.bin"
    with weights_path.open("wb") as stream:
        stream.write(MAGIC)
        stream.write(struct.pack("<4I", 520, 64, 32, 5))
        for weight, bias in layers:
            stream.write(weight.detach().numpy().astype("<f4").tobytes())
            stream.write(bias.detach().numpy().astype("<f4").tobytes())
    (output_dir / "input.f32").write_bytes(vector.astype("<f4").tobytes())
    reference = {
        "checkpoint": str(checkpoint),
        "checkpoint_seed": saved.get("seed"),
        "feature_index": index,
        "input_shape": [40, 13],
        "labels": LABELS,
        "logits": [float(value) for value in logits],
        "predicted_label": LABELS[int(np.argmax(logits))],
        "weights_bytes": weights_path.stat().st_size,
        "tolerance_abs": 1e-5,
    }
    (output_dir / "reference.json").write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
    return {"weights": str(weights_path), "input": str(output_dir / "input.f32"), "reference": str(output_dir / "reference.json")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("models/mlp/mlp_seed_20261022.pt"))
    parser.add_argument("--features", type=Path, default=Path("data/processed/features.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/embedded/mlp_poc"))
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(export(args.checkpoint, args.features, args.output_dir, args.index)))


if __name__ == "__main__":
    main()
