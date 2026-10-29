"""Export the TinyTransformer tensors and a deterministic host reference."""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch

from scripts.training.train_transformer import LABELS, TinyTransformer


MAGIC = b"VTKTR1\0\0"
TENSORS = (
    ("input_projection.weight", "input_projection__weight"),
    ("input_projection.bias", "input_projection__bias"),
    ("position", "position"),
    ("attention.in_proj_weight", "attention__in_proj_weight"),
    ("attention.in_proj_bias", "attention__in_proj_bias"),
    ("attention.out_proj.weight", "attention__out_proj_weight"),
    ("attention.out_proj.bias", "attention__out_proj_bias"),
    ("norm1.weight", "norm1__weight"),
    ("norm1.bias", "norm1__bias"),
    ("ffn.0.weight", "ffn0__weight"),
    ("ffn.0.bias", "ffn0__bias"),
    ("ffn.2.weight", "ffn2__weight"),
    ("ffn.2.bias", "ffn2__bias"),
    ("norm2.weight", "norm2__weight"),
    ("norm2.bias", "norm2__bias"),
    ("classifier.weight", "classifier__weight"),
    ("classifier.bias", "classifier__bias"),
)


def numpy_reference(vector: np.ndarray, tensors: dict[str, np.ndarray]) -> np.ndarray:
    x = vector.reshape(40, 13).astype(np.float32)
    h = x @ tensors["input_projection.weight"].T + tensors["input_projection.bias"]
    h = h + tensors["position"][0]
    qkv = h @ tensors["attention.in_proj_weight"].T + tensors["attention.in_proj_bias"]
    q, k, v = np.split(qkv, 3, axis=-1)
    q = q.reshape(40, 2, 16).transpose(1, 0, 2)
    k = k.reshape(40, 2, 16).transpose(1, 0, 2)
    v = v.reshape(40, 2, 16).transpose(1, 0, 2)
    scores = np.matmul(q, np.transpose(k, (0, 2, 1))) / np.sqrt(16.0)
    scores -= scores.max(axis=-1, keepdims=True)
    attention = np.exp(scores)
    attention /= attention.sum(axis=-1, keepdims=True)
    attended = np.matmul(attention, v).transpose(1, 0, 2).reshape(40, 32)
    attended = attended @ tensors["attention.out_proj.weight"].T + tensors["attention.out_proj.bias"]
    h = h + attended
    h = (h - h.mean(axis=-1, keepdims=True)) / np.sqrt(h.var(axis=-1, keepdims=True) + 1e-5)
    h = h * tensors["norm1.weight"] + tensors["norm1.bias"]
    residual = h @ tensors["ffn.0.weight"].T + tensors["ffn.0.bias"]
    residual = np.maximum(residual, 0.0)
    residual = residual @ tensors["ffn.2.weight"].T + tensors["ffn.2.bias"]
    h = h + residual
    h = (h - h.mean(axis=-1, keepdims=True)) / np.sqrt(h.var(axis=-1, keepdims=True) + 1e-5)
    h = h * tensors["norm2.weight"] + tensors["norm2.bias"]
    pooled = h.mean(axis=0)
    return (pooled @ tensors["classifier.weight"].T + tensors["classifier.bias"]).astype(np.float32)


def export(checkpoint: Path, features: Path, output_dir: Path, index: int) -> dict:
    archive = np.load(features, allow_pickle=False)
    vector = archive["x"][index].astype(np.float32).reshape(-1)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = TinyTransformer()
    model.load_state_dict(saved["model_state"])
    model.eval()
    state = model.state_dict()
    tensors = {key: state[key].detach().numpy().astype("<f4") for key, _ in TENSORS}
    with torch.inference_mode():
        torch_logits = model(torch.from_numpy(vector.reshape(1, 40, 13)))[0].numpy().astype(np.float32)
    numpy_logits = numpy_reference(vector, tensors)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "transformer_fp32.bin"
    offsets = {}
    with weights_path.open("wb") as stream:
        stream.write(MAGIC)
        stream.write(struct.pack("<6I", 13, 40, 32, 2, 64, 5))
        for key, _ in TENSORS:
            array = tensors[key]
            offsets[key] = {"offset": stream.tell(), "shape": list(array.shape), "bytes": int(array.nbytes)}
            stream.write(array.tobytes(order="C"))
    (output_dir / "input.f32").write_bytes(vector.astype("<f4").tobytes())
    architecture = {
        "format": "VTKTR1",
        "operators": ["linear", "reshape_qkv", "scaled_dot_product_attention", "softmax", "residual_add", "layer_norm", "relu", "mean_pool", "linear"],
        "input_shape": [40, 13],
        "attention": {"heads": 2, "head_dim": 16, "scores_shape": [2, 40, 40]},
        "buffers": {"tokens_fp32": 40 * 32 * 4, "qkv_fp32": 40 * 96 * 4, "scores_fp32": 2 * 40 * 40 * 4, "attention_fp32": 40 * 32 * 4},
        "weights_bytes": weights_path.stat().st_size - 32,
        "binary_bytes": weights_path.stat().st_size,
        "offsets": offsets,
    }
    (output_dir / "architecture.json").write_text(json.dumps(architecture, indent=2) + "\n", encoding="utf-8")
    reference = {
        "checkpoint": str(checkpoint),
        "feature_index": index,
        "labels": LABELS,
        "torch_logits": [float(value) for value in torch_logits],
        "numpy_logits": [float(value) for value in numpy_logits],
        "numpy_max_abs_error": float(np.max(np.abs(torch_logits - numpy_logits))),
        "predicted_index": int(np.argmax(torch_logits)),
        "tolerance_abs": 1e-4,
    }
    (output_dir / "reference.json").write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
    return {"weights": str(weights_path), "architecture": str(output_dir / "architecture.json"), "reference": str(output_dir / "reference.json")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("models/transformer/transformer_seed_20261022.pt"))
    parser.add_argument("--features", type=Path, default=Path("data/processed/features.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/embedded/transformer_poc"))
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(export(args.checkpoint, args.features, args.output_dir, args.index), indent=2))


if __name__ == "__main__":
    main()
