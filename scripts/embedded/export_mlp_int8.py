"""Export a compressed MLP NPZ with preserved per-layer scales for MicroPython."""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


MAGIC = b"VTKMLP8\0"
LAYERS = (("net__1__weight", "net__1__bias"), ("net__3__weight", "net__3__bias"), ("net__5__weight", "net__5__bias"))


def infer(vector: np.ndarray, arrays: dict[str, np.ndarray], scales: dict[str, float]) -> np.ndarray:
    value = vector.astype(np.float32, copy=False)
    for weight_key, bias_key in LAYERS:
        weight = arrays[weight_key].astype(np.float32) * np.float32(scales[weight_key])
        value = weight @ value + arrays[bias_key].astype(np.float32)
        if weight_key != "net__5__weight":
            value = np.maximum(value, 0.0)
    return value.astype(np.float32)


def export(artifact: Path, scales_json: Path, input_path: Path, output_dir: Path) -> dict:
    archive = np.load(artifact, allow_pickle=False)
    arrays = {key: archive[key] for key in archive.files}
    metadata = json.loads(scales_json.read_text(encoding="utf-8"))
    scales = metadata["scales"] if "scales" in metadata else metadata
    vector = np.fromfile(input_path, dtype="<f4")
    output_dir.mkdir(parents=True, exist_ok=True)
    binary = output_dir / "mlp_int8.bin"
    with binary.open("wb") as stream:
        stream.write(MAGIC)
        stream.write(struct.pack("<4I", 520, 64, 32, 5))
        for weight_key, bias_key in LAYERS:
            weights = np.asarray(arrays[weight_key], dtype=np.int8)
            bias = np.asarray(arrays[bias_key], dtype="<f4")
            stream.write(weights.tobytes(order="C"))
            stream.write(struct.pack("<f", float(scales[weight_key])))
            stream.write(bias.tobytes(order="C"))
    logits = infer(vector, arrays, scales)
    (output_dir / "input.f32").write_bytes(vector.astype("<f4").tobytes())
    reference = {
        "artifact": str(artifact),
        "scales": scales,
        "logits": [float(value) for value in logits],
        "predicted_index": int(np.argmax(logits)),
        "weights_bytes": binary.stat().st_size - 24,
        "binary_bytes": binary.stat().st_size,
        "tolerance_abs": 1e-4,
    }
    (output_dir / "reference.json").write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
    return {"binary": str(binary), "reference": str(output_dir / "reference.json")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--scales-json", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=Path("experiments/embedded/mlp_poc/input.f32"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.artifact, args.scales_json, args.input, args.output_dir)))


if __name__ == "__main__":
    main()
