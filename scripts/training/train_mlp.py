"""Treina e avalia a MLP baseline do VoiceTiny RP2040."""
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


class MLP(nn.Module):
    def __init__(self, input_dim: int = 520, hidden1: int = 64, hidden2: int = 32, classes: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_data(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    archive = np.load(path, allow_pickle=False)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    data = {key: archive[key] for key in ("x", "y", "splits", "paths")}
    return data, metadata


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    true, predicted = [], []
    with torch.inference_mode():
        for features, labels in loader:
            logits = model(features.to(device))
            true.append(labels.numpy())
            predicted.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(true), np.concatenate(predicted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("data/processed/features.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/mlp"))
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
    results = []
    for seed in args.seeds:
        seed_everything(seed)
        train_mask, val_mask, test_mask = split == "train", split == "validation", split == "test"
        train_ds = TensorDataset(torch.from_numpy(data["x"][train_mask]), torch.from_numpy(data["y"][train_mask]))
        val_ds = TensorDataset(torch.from_numpy(data["x"][val_mask]), torch.from_numpy(data["y"][val_mask]))
        test_ds = TensorDataset(torch.from_numpy(data["x"][test_mask]), torch.from_numpy(data["y"][test_mask]))
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=args.workers)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        model = MLP().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss(weight=weights)
        best_score = -1.0
        best_epoch = 0
        stale = 0
        history = []
        checkpoint = args.output_dir / f"mlp_seed_{seed}.pt"
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum = 0.0
            for features, labels in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(features.to(device)), labels.to(device))
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(labels)
            val_true, val_pred = predict(model, val_loader, device)
            val_metrics = compute_metrics(val_true, val_pred, LABELS)
            row = {"epoch": epoch, "loss": loss_sum / len(train_ds), "val_balanced_accuracy": val_metrics["balanced_accuracy"], "val_macro_f1": val_metrics["macro_f1"]}
            history.append(row)
            score = (val_metrics["balanced_accuracy"], val_metrics["macro_f1"])
            best_tuple = (best_score, history[best_epoch - 1]["val_macro_f1"] if best_epoch else -1.0)
            if score > best_tuple:
                best_score = score[0]
                best_epoch = epoch
                stale = 0
                torch.save({"model_state": model.state_dict(), "model": "MLP", "seed": seed, "epoch": epoch, "config": vars(args), "labels": LABELS, "parameter_count": sum(p.numel() for p in model.parameters())}, checkpoint)
            else:
                stale += 1
            if stale >= args.patience:
                break
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        best_model = MLP().to(device)
        best_model.load_state_dict(saved["model_state"])
        test_true, test_pred = predict(best_model, test_loader, device)
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
