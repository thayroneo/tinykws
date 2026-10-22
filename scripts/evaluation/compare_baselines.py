"""Compara MLP e TinyTransformer no mesmo teste e mede inferência no host."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.evaluation.evaluate import compute_metrics
from scripts.training.train_mlp import MLP
from scripts.training.train_transformer import TinyTransformer

LABELS = ["yes", "no", "stop", "unknown", "silence"]
SEEDS = (20261022, 20261023, 20261024)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(name: str, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = MLP() if name == "mlp" else TinyTransformer()
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state"])
    return model.to(device).eval()


def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    true, predicted = [], []
    with torch.inference_mode():
        for features, labels in loader:
            predicted.append(model(features.to(device)).argmax(dim=1).cpu().numpy())
            true.append(labels.numpy())
    return np.concatenate(true), np.concatenate(predicted)


def benchmark(model: torch.nn.Module, loader: DataLoader, device: torch.device, repeats: int) -> dict:
    for _ in range(2):
        predict(model, loader, device)
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        predict(model, loader, device)
        durations.append((time.perf_counter() - start) * 1000)
    return {
        "median_ms": float(np.median(durations)),
        "p95_ms": float(np.percentile(durations, 95)),
        "all_ms": [float(value) for value in durations],
    }


def memory_budget() -> dict:
    sram = 264 * 1024
    mlp_parameters = sum(parameter.numel() for parameter in MLP().parameters())
    transformer_parameters = sum(parameter.numel() for parameter in TinyTransformer().parameters())
    mlp_activation = 40 * 13 * 4 + 64 * 4 + 32 * 4 + 5 * 4
    transformer_input = 40 * 13 * 4
    transformer_tokens = 40 * 32 * 4
    transformer_qkv = 3 * transformer_tokens
    transformer_scores = 2 * 40 * 40 * 4
    transformer_ffn = 40 * 64 * 4
    transformer_activation = transformer_input + transformer_tokens + transformer_qkv + transformer_scores + transformer_ffn + 5 * 4
    return {
        "rp2040_sram_bytes": sram,
        "assumption": "weights counted separately as flash; stack and runtime allocator are not measured",
        "mlp": {
            "parameters": mlp_parameters,
            "weights_fp32_bytes": mlp_parameters * 4,
            "weights_int8_bytes_estimate": mlp_parameters,
            "activation_bytes_conservative": mlp_activation,
            "known_sram_margin_after_activations": sram - mlp_activation,
        },
        "tiny_transformer": {
            "parameters": transformer_parameters,
            "weights_fp32_bytes": transformer_parameters * 4,
            "weights_int8_bytes_estimate": transformer_parameters,
            "activation_bytes_conservative": transformer_activation,
            "known_sram_margin_after_activations": sram - transformer_activation,
        },
        "stack_bytes": None,
        "runtime_buffers_bytes": None,
        "status": "preliminary; validate on device",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("data/processed/features.npz"))
    parser.add_argument("--mlp-dir", type=Path, default=Path("models/mlp"))
    parser.add_argument("--transformer-dir", type=Path, default=Path("models/transformer"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/baselines"))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    device = torch.device("cpu")
    archive = np.load(args.features, allow_pickle=False)
    test_mask = archive["splits"] == "test"
    dataset = TensorDataset(torch.from_numpy(archive["x"][test_mask]), torch.from_numpy(archive["y"][test_mask]))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, model_dir, prefix in (("mlp", args.mlp_dir, "mlp"), ("tiny_transformer", args.transformer_dir, "transformer")):
        for seed in SEEDS:
            checkpoint = model_dir / f"{prefix}_seed_{seed}.pt"
            model = load_model(name, checkpoint, device)
            truth, predicted = predict(model, loader, device)
            metrics = compute_metrics(truth, predicted, LABELS)
            timing = benchmark(model, loader, device, args.repeats)
            rows.append({
                "model": name,
                "seed": seed,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "checkpoint_bytes": checkpoint.stat().st_size,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "test_examples": len(dataset),
                **{key: metrics[key] for key in ("accuracy", "balanced_accuracy", "macro_f1", "confusion_matrix")},
                "host_inference_median_ms": timing["median_ms"],
                "host_inference_p95_ms": timing["p95_ms"],
                "host_inference_median_us_per_example": timing["median_ms"] * 1000 / len(dataset),
                "torch_threads": args.torch_threads,
            })
    summary = {"device": str(device), "batch_size": args.batch_size, "repeats": args.repeats, "rows": rows, "memory_budget": memory_budget()}
    (args.output_dir / "baseline_comparison.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = [key for key in rows[0] if key != "confusion_matrix"]
    with (args.output_dir / "baseline_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in rows])
    (args.output_dir / "memory_budget.json").write_text(json.dumps(summary["memory_budget"], indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(json.dumps({key: row[key] for key in ("model", "seed", "accuracy", "balanced_accuracy", "macro_f1", "checkpoint_bytes", "host_inference_median_ms", "host_inference_p95_ms")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
