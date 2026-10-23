"""Avalia a sensibilidade do baseline ao limite de exemplos unknown no treino."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


LABELS = ["yes", "no", "stop", "unknown", "silence"]


def make_subset(source: Path, target: Path, unknown_cap: int) -> None:
    archive = np.load(source, allow_pickle=False)
    x, y, splits, paths = archive["x"], archive["y"], archive["splits"], archive["paths"]
    train_unknown = np.where((splits == "train") & (y == LABELS.index("unknown")))[0]
    ordered = train_unknown[np.argsort(paths[train_unknown])]
    selected = set(ordered[:unknown_cap].tolist())
    keep = np.array(
        [
            split_name != "train"
            or int(label) != LABELS.index("unknown")
            or index in selected
            for index, (split_name, label) in enumerate(zip(splits.tolist(), y.tolist()))
        ],
        dtype=bool,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, x=x[keep], y=y[keep], splits=splits[keep], paths=paths[keep])
    counts = {label: int(((splits[keep] == "train") & (y[keep] == index)).sum()) for index, label in enumerate(LABELS)}
    total = sum(counts.values())
    metadata = {
        "labels": LABELS,
        "source": str(source),
        "unknown_cap": unknown_cap,
        "counts_train": counts,
        "class_weights": {label: total / (len(LABELS) * count) for label, count in counts.items()},
    }
    target.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_training(features: Path, output_root: Path) -> None:
    for model, module in (("mlp", "scripts.training.train_mlp"), ("transformer", "scripts.training.train_transformer")):
        output = output_root / model
        command = [
            sys.executable,
            "-m",
            module,
            "--features",
            str(features),
            "--output-dir",
            str(output),
            "--epochs",
            "30",
            "--batch-size",
            "128",
            "--workers",
            "0",
        ]
        subprocess.run(command, check=True)


def main() -> None:
    source = Path("data/processed/features.npz")
    root = Path("experiments/sensitivity_unknown")
    cap = 2000
    features = root / f"features_unknown_{cap}.npz"
    make_subset(source, features, cap)
    run_training(features, root / f"unknown_{cap}")
    rows = []
    for model in ("mlp", "transformer"):
        summary = json.loads(
            (root / f"unknown_{cap}" / model / "summary.json").read_text(encoding="utf-8")
        )
        for row in summary["results"]:
            rows.append({"unknown_cap": cap, "model": model, **row})
    (root / "summary.json").write_text(
        json.dumps({"rows": rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "output": str(root / "summary.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
