"""Treina e avalia o TinyTransformer baseline do VoiceTiny RP2040."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from scripts.evaluation.evaluate import compute_metrics
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.evaluation.evaluate import compute_metrics

LABELS = ["yes", "no", "stop", "unknown", "silence"]


class TinyTransformer(nn.Module):
    """Um bloco de atenção com 40 tokens MFCC e duas cabeças."""

    def __init__(self, input_dim: int = 13, tokens: int = 40, d_model: int = 32, heads: int = 2, ffn_dim: int = 64, classes: int = 5) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position = nn.Parameter(torch.zeros(1, tokens, d_model))
        self.attention = nn.MultiheadAttention(d_model, heads, dropout=0.0, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.ReLU(), nn.Linear(ffn_dim, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_projection(x) + self.position
        attended, _ = self.attention(h, h, h, need_weights=False)
        h = self.norm1(h + attended)
        h = self.norm2(h + self.ffn(h))
        return self.classifier(h.mean(dim=1))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_data(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    archive = np.load(path, allow_pickle=False)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return {key: archive[key] for key in ("x", "y", "splits", "paths")}, metadata


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    true, predicted = [], []
    with torch.inference_mode():
        for features, labels in loader:
            predicted.append(model(features.to(device)).argmax(dim=1).cpu().numpy())
            true.append(labels.numpy())
    return np.concatenate(true), np.concatenate(predicted)


def architecture_report(model: nn.Module) -> dict:
    parameters = sum(parameter.numel() for parameter in model.parameters())
    fp32_weights = parameters * 4
    token_buffer = 40 * 32 * 4
    attention_scores = 2 * 40 * 40 * 4
    return {
        "model": "TinyTransformer",
        "input_shape": [40, 13],
        "projection_shape": [40, 32],
        "heads": 2,
        "ffn_dim": 64,
        "layers": 1,
        "output_shape": [5],
        "parameter_count": parameters,
        "fp32_weight_bytes": fp32_weights,
        "int8_weight_bytes_estimate": parameters,
        "activation_bytes_preliminary": {"token_buffer": token_buffer, "attention_scores": attention_scores, "sum": token_buffer + attention_scores},
        "note": "Estimativa preliminar; SRAM de pico deve ser medida no runtime embarcado.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("data/processed/features.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/transformer"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261022, 20261023, 20261024])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    data, metadata = load_data(args.features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split = data["splits"]
    weights = torch.tensor([metadata["class_weights"][label] for label in LABELS], dtype=torch.float32, device=device)
    report_model = TinyTransformer()
    (args.output_dir / "architecture.json").write_text(json.dumps(architecture_report(report_model), indent=2) + "\n", encoding="utf-8")
    results = []
    for seed in args.seeds:
        seed_everything(seed)
        masks = {name: split == name for name in ("train", "validation", "test")}
        datasets = {name: TensorDataset(torch.from_numpy(data["x"][mask]), torch.from_numpy(data["y"][mask])) for name, mask in masks.items()}
        loaders = {
            "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=args.workers),
            "validation": DataLoader(datasets["validation"], batch_size=args.batch_size, shuffle=False, num_workers=args.workers),
            "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False, num_workers=args.workers),
        }
        model = TinyTransformer().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss(weight=weights)
        best_score = (-1.0, -1.0)
        stale = 0
        history = []
        checkpoint = args.output_dir / f"transformer_seed_{seed}.pt"
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum = 0.0
            for features, labels in loaders["train"]:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(features.to(device)), labels.to(device))
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(labels)
            val_true, val_pred = predict(model, loaders["validation"], device)
            val_metrics = compute_metrics(val_true, val_pred, LABELS)
            score = (val_metrics["balanced_accuracy"], val_metrics["macro_f1"])
            history.append({"epoch": epoch, "loss": loss_sum / len(datasets["train"]), "val_balanced_accuracy": score[0], "val_macro_f1": score[1]})
            if score > best_score:
                best_score = score
                stale = 0
                torch.save({"model_state": model.state_dict(), "model": "TinyTransformer", "seed": seed, "epoch": epoch, "config": vars(args), "labels": LABELS, "parameter_count": sum(p.numel() for p in model.parameters())}, checkpoint)
            else:
                stale += 1
            if stale >= args.patience:
                break
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        best_model = TinyTransformer().to(device)
        best_model.load_state_dict(saved["model_state"])
        test_true, test_pred = predict(best_model, loaders["test"], device)
        metrics = compute_metrics(test_true, test_pred, LABELS)
        metrics.update({"seed": seed, "best_epoch": saved["epoch"], "parameter_count": saved["parameter_count"], "checkpoint": str(checkpoint), "history": history})
        (args.output_dir / f"metrics_seed_{seed}.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        np.save(args.output_dir / f"predictions_seed_{seed}.npy", test_pred)
        np.save(args.output_dir / f"truth_seed_{seed}.npy", test_true)
        results.append({key: metrics[key] for key in ("seed", "accuracy", "balanced_accuracy", "macro_f1", "parameter_count", "best_epoch")})
        print(json.dumps(results[-1], ensure_ascii=False))
    (args.output_dir / "summary.json").write_text(json.dumps({"results": results, "device": str(device), "labels": LABELS}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
