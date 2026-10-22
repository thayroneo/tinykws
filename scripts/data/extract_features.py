"""Extrai MFCCs e aplica normalização ajustada somente no treino."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import yaml

LABELS = ("yes", "no", "stop", "unknown", "silence")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_audio_path(project_root: Path, relative_path: str) -> Path:
    if relative_path.startswith("processed/"):
        return project_root / "data" / relative_path
    return project_root / "data" / "raw" / "speech_commands" / relative_path


def extract_one(path: Path, sample_rate: int, duration_seconds: float, params: dict) -> np.ndarray:
    target_samples = int(sample_rate * duration_seconds)
    audio, source_rate = librosa.load(path, sr=None, mono=True)
    if source_rate != sample_rate:
        audio = librosa.resample(audio, orig_sr=source_rate, target_sr=sample_rate)
    audio = np.asarray(audio, dtype=np.float32)
    audio = audio[:target_samples]
    if audio.size < target_samples:
        audio = np.pad(audio, (0, target_samples - audio.size))
    # With center=False, librosa uses n_fft as the effective frame length.
    analysis_pad = params["n_fft"] - params["win_length"]
    if analysis_pad > 0:
        audio = np.pad(audio, (0, analysis_pad))
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=params["n_mfcc"],
        n_fft=params["n_fft"],
        hop_length=params["hop_length"],
        win_length=params["win_length"],
        n_mels=params["n_mels"],
        fmin=params["fmin_hz"],
        fmax=params["fmax_hz"],
        center=False,
    ).T.astype(np.float32)
    expected_shape = (params["frames"], params["n_mfcc"])
    if mfcc.shape != expected_shape:
        raise ValueError(f"{path}: MFCC {mfcc.shape}, esperado {expected_shape}")
    return mfcc


def choose_rows(rows: list[dict], max_train_unknown: int | None, seed: int, limit_per_split: int | None) -> list[dict]:
    rng = np.random.default_rng(seed)
    selected = []
    for split in ("train", "validation", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        if split == "train" and max_train_unknown is not None:
            unknown = [row for row in split_rows if row["label"] == "unknown"]
            other = [row for row in split_rows if row["label"] != "unknown"]
            if len(unknown) > max_train_unknown:
                unknown = [unknown[i] for i in rng.choice(len(unknown), max_train_unknown, replace=False)]
            split_rows = other + unknown
        if limit_per_split is not None:
            split_rows = split_rows[:limit_per_split]
        selected.extend(sorted(split_rows, key=lambda row: row["path"]))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/speech_commands.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/features.npz"))
    parser.add_argument("--config", type=Path, default=Path("configs/features.yaml"))
    parser.add_argument("--max-train-unknown", type=int, default=4000)
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    rows = choose_rows(load_manifest(args.manifest), args.max_train_unknown, seed, args.limit_per_split)
    project_root = Path.cwd()
    features = []
    labels = []
    splits = []
    paths = []
    def extract_row(row: dict) -> np.ndarray:
        return extract_one(resolve_audio_path(project_root, row["path"]), config["sample_rate_hz"], config["duration_seconds"], config["mfcc"])

    iterator = rows
    executor = ThreadPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    if executor is not None:
        iterator = zip(rows, executor.map(extract_row, rows))
    else:
        iterator = ((row, extract_row(row)) for row in rows)
    for index, (row, feature) in enumerate(iterator, start=1):
        features.append(feature)
        labels.append(LABEL_TO_ID[row["label"]])
        splits.append(row["split"])
        paths.append(row["path"])
        if index % 1000 == 0:
            print(f"Extraídos: {index}/{len(rows)}")
    if executor is not None:
        executor.shutdown()
    x = np.stack(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    split_array = np.asarray(splits)
    train_mask = split_array == "train"
    mean = x[train_mask].mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = x[train_mask].std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-6)
    x = ((x - mean[None, None, :]) / std[None, None, :]).astype(np.float32)
    counts = Counter(y[train_mask].tolist())
    total = sum(counts.values())
    class_weights = {LABELS[i]: total / (len(LABELS) * counts.get(i, 1)) for i in range(len(LABELS))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, x=x, y=y, splits=split_array, paths=np.asarray(paths), mean=mean, std=std)
    metadata = {"labels": LABELS, "seed": seed, "mfcc": config["mfcc"], "class_weights": class_weights, "counts_train": {LABELS[i]: counts.get(i, 0) for i in range(len(LABELS))}}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Features: {args.output} shape={x.shape}")
    print(f"Treino: {metadata['counts_train']}")
    print(f"Pesos de classe: {class_weights}")


if __name__ == "__main__":
    main()

