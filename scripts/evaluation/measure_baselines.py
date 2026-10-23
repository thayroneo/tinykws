"""Fecha T2.5: mede latência no host e exporta artefatos dos baselines."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.compression.run_compression import (
    LABELS,
    CompressedModel,
    export_variant,
    load_archive,
    load_checkpoint,
)
from scripts.evaluation.evaluate import compute_metrics


def measure(model, loader, device, repeats):
    for _ in range(2):
        with torch.inference_mode():
            for features, _ in loader:
                model(features.to(device))
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        with torch.inference_mode():
            for features, _ in loader:
                model(features.to(device))
        durations.append((time.perf_counter() - start) * 1000)
    return {
        "median_ms": float(np.median(durations)),
        "p95_ms": float(np.percentile(durations, 95)),
        "all_ms": [float(value) for value in durations],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("data/processed/features.npz"))
    parser.add_argument("--mlp-dir", type=Path, default=Path("models/mlp"))
    parser.add_argument("--transformer-dir", type=Path, default=Path("models/transformer"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/baselines"))
    parser.add_argument("--batch-reference", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device("cpu")
    data, _ = load_archive(args.features)
    test_mask = data["splits"] == "test"
    test_x = torch.from_numpy(data["x"][test_mask])
    test_y = torch.from_numpy(data["y"][test_mask])
    rows = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for model_name, model_dir, prefix in (("mlp", args.mlp_dir, "mlp"), ("tiny_transformer", args.transformer_dir, "transformer")):
        for seed in (20261022, 20261023, 20261024):
            checkpoint = model_dir / f"{prefix}_seed_{seed}.pt"
            base = load_checkpoint(model_name, checkpoint, device).eval()
            model = CompressedModel(base, model_name, None, "fp32").eval()
            truth, prediction = [], []
            ref_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=args.batch_reference, shuffle=False)
            with torch.inference_mode():
                for features, labels in ref_loader:
                    truth.append(labels.numpy())
                    prediction.append(model(features).argmax(dim=1).numpy())
            metrics = compute_metrics(np.concatenate(truth), np.concatenate(prediction), LABELS)
            for batch_size in (1, args.batch_reference):
                loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False)
                timing = measure(model, loader, device, args.repeats)
                rows.append({
                    "model": model_name,
                    "seed": seed,
                    "variant": "fp32",
                    "batch_size": batch_size,
                    "test_examples": len(test_y),
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "latency_median_ms": timing["median_ms"],
                    "latency_p95_ms": timing["p95_ms"],
                    "latency_us_per_example": timing["median_ms"] * 1000 / len(test_y),
                })
            export_dir = args.output_dir / "exports" / model_name
            export_dir.mkdir(parents=True, exist_ok=True)
            export = export_variant(export_dir / f"fp32_seed_{seed}.npz", base, model_name, None, "fp32", 8)
            for row in rows[-2:]:
                row["export_file_bytes"] = export["artifact_file_bytes"]
                row["fp32_parameter_bytes"] = export["fp32_parameter_bytes"]
                row["dense_quantized_bytes"] = export["dense_quantized_bytes"]
    payload = {
        "protocol": {
            "device": "cpu",
            "torch_threads": 1,
            "batch_sizes": [1, args.batch_reference],
            "repeats": args.repeats,
            "latency_scope": "full fixed public test split",
            "export_scope": "weight tensors and float32 non-quantized tensors in NPZ",
        },
        "rows": rows,
    }
    (args.output_dir / "baseline_measurements.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = sorted({key for row in rows for key in row})
    with (args.output_dir / "baseline_measurements.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "output": str(args.output_dir / "baseline_measurements.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
