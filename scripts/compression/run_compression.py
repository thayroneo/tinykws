"""Executa T2.5 e T3.2--T3.5 para MLP e TinyTransformer.

O pipeline quantiza pesos simetricamente por tensor em INT8. As variantes iterativas
usam máscaras globais fixas por magnitude e straight-through estimator:
P→Q aplica a máscara antes da quantização; Q→P quantiza antes da máscara.
A variante Q→P faz duas atualizações no mesmo mini-batch, conforme a fonte auditada.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.evaluation.evaluate import compute_metrics
from scripts.training.train_mlp import MLP
from scripts.training.train_transformer import TinyTransformer

LABELS = ["yes", "no", "stop", "unknown", "silence"]
SEEDS = (20261022, 20261023, 20261024)
ELIGIBLE_MLP = ("net.1.weight", "net.3.weight", "net.5.weight")
ELIGIBLE_TRANSFORMER = (
    "input_projection.weight",
    "position",
    "attention.in_proj_weight",
    "attention.out_proj.weight",
    "ffn.0.weight",
    "ffn.2.weight",
    "classifier.weight",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_archive(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    archive = np.load(path, allow_pickle=False)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    data = {key: archive[key] for key in ("x", "y", "splits", "paths")}
    return data, metadata


def build_model(name: str) -> nn.Module:
    return MLP() if name == "mlp" else TinyTransformer()


def load_checkpoint(name: str, path: Path, device: torch.device) -> nn.Module:
    model = build_model(name).to(device)
    saved = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state"])
    return model


def eligible_names(name: str) -> tuple[str, ...]:
    return ELIGIBLE_MLP if name == "mlp" else ELIGIBLE_TRANSFORMER


def named_parameter_map(model: nn.Module) -> dict[str, nn.Parameter]:
    return dict(model.named_parameters())


def build_masks(model: nn.Module, name: str, sparsity: float) -> dict[str, torch.Tensor]:
    parameters = named_parameter_map(model)
    names = eligible_names(name)
    if sparsity <= 0:
        return {key: torch.ones_like(parameters[key]) for key in names}
    total = sum(parameters[key].numel() for key in names)
    keep = max(1, int(round(total * (1.0 - sparsity))))
    magnitudes = torch.cat([parameters[key].detach().abs().flatten().cpu() for key in names])
    threshold_index = max(0, total - keep)
    threshold = torch.kthvalue(magnitudes, threshold_index + 1).values.item()
    mask = {}
    remaining = keep
    for key in names:
        values = parameters[key].detach().abs()
        current = int((values > threshold).sum().item())
        mask[key] = (values > threshold).to(dtype=parameters[key].dtype)
        remaining -= current
    if remaining > 0:
        candidates: list[tuple[float, str, int]] = []
        for key in names:
            values = parameters[key].detach().abs().flatten()
            for index, value in enumerate(values.tolist()):
                if value == threshold and mask[key].flatten()[index].item() == 0:
                    candidates.append((value, key, index))
        for _, key, index in candidates[:remaining]:
            mask[key].flatten()[index] = 1
    return mask


def symmetric_quantize(tensor: torch.Tensor, bits: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    levels = float(2 ** (bits - 1) - 1)
    max_abs = tensor.detach().abs().max()
    scale = torch.clamp(max_abs / levels, min=torch.finfo(tensor.dtype).eps)
    quantized = torch.clamp(torch.round(tensor / scale), -levels - 1, levels).to(torch.int8)
    return quantized, scale


def fake_quant_ste(tensor: torch.Tensor, bits: int = 8) -> torch.Tensor:
    quantized, scale = symmetric_quantize(tensor, bits)
    dequantized = quantized.to(dtype=tensor.dtype) * scale
    return tensor + (dequantized - tensor).detach()


def transform_weight(
    tensor: torch.Tensor,
    mask: torch.Tensor | None,
    method: str,
    bits: int = 8,
) -> torch.Tensor:
    if method in {"fp32", "none"}:
        return tensor
    if method in {"int8_weight_ptq", "qp", "qp_sequential_control"}:
        quantized = fake_quant_ste(tensor, bits)
        return quantized if mask is None else quantized * mask
    if method in {"pq", "pq_sequential_control"}:
        pruned = tensor if mask is None else tensor * mask
        return fake_quant_ste(pruned, bits)
    raise ValueError(f"Método desconhecido: {method}")


class CompressedModel(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        model_name: str,
        masks: dict[str, torch.Tensor] | None,
        method: str,
        bits: int = 8,
    ) -> None:
        super().__init__()
        self.base = base
        self.model_name = model_name
        self.masks = masks or {}
        self.method = method
        self.bits = bits

    def weight(self, name: str) -> torch.Tensor:
        parameter = dict(self.base.named_parameters())[name]
        mask = self.masks.get(name)
        return transform_weight(parameter, mask, self.method, self.bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_name == "mlp":
            h = x.flatten(1)
            for layer_index in (1, 3, 5):
                layer = self.base.net[layer_index]
                weight = self.weight(f"net.{layer_index}.weight")
                h = F.linear(h, weight, layer.bias)
                if layer_index != 5:
                    h = torch.relu(h)
            return h

        base = self.base
        h = F.linear(x, self.weight("input_projection.weight"), base.input_projection.bias)
        h = h + self.weight("position")
        in_proj = self.weight("attention.in_proj_weight")
        qkv = F.linear(h, in_proj, base.attention.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        batch, tokens, d_model = q.shape
        heads = base.attention.num_heads
        head_dim = d_model // heads
        q = q.view(batch, tokens, heads, head_dim).transpose(1, 2)
        k = k.view(batch, tokens, heads, head_dim).transpose(1, 2)
        v = v.view(batch, tokens, heads, head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        attention = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attention, v)
        attended = attended.transpose(1, 2).contiguous().view(batch, tokens, d_model)
        attended = F.linear(
            attended,
            self.weight("attention.out_proj.weight"),
            base.attention.out_proj.bias,
        )
        h = base.norm1(h + attended)
        ffn_hidden = F.linear(h, self.weight("ffn.0.weight"), base.ffn[0].bias)
        ffn_hidden = torch.relu(ffn_hidden)
        ffn_output = F.linear(ffn_hidden, self.weight("ffn.2.weight"), base.ffn[2].bias)
        h = base.norm2(h + ffn_output)
        return F.linear(h.mean(dim=1), self.weight("classifier.weight"), base.classifier.bias)


def project_masks(base: nn.Module, masks: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        parameters = dict(base.named_parameters())
        for name, mask in masks.items():
            parameters[name].mul_(mask)


def make_loaders(
    data: dict[str, np.ndarray],
    batch_size: int,
    seed: int,
) -> dict[str, DataLoader]:
    loaders = {}
    for split_name in ("train", "validation", "test"):
        split_mask = data["splits"] == split_name
        dataset = TensorDataset(
            torch.from_numpy(data["x"][split_mask]),
            torch.from_numpy(data["y"][split_mask]),
        )
        generator = torch.Generator().manual_seed(seed)
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split_name == "train",
            generator=generator if split_name == "train" else None,
            num_workers=0,
        )
    return loaders


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    repeats: int = 0,
) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    truth, predicted = [], []
    with torch.inference_mode():
        for features, labels in loader:
            logits = model(features.to(device))
            truth.append(labels.numpy())
            predicted.append(logits.argmax(dim=1).cpu().numpy())
    y_true = np.concatenate(truth)
    y_pred = np.concatenate(predicted)
    metrics = compute_metrics(y_true, y_pred, LABELS)
    if repeats:
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
        metrics["latency_median_ms"] = float(np.median(durations))
        metrics["latency_p95_ms"] = float(np.percentile(durations, 95))
        metrics["latency_all_ms"] = [float(item) for item in durations]
        metrics["latency_us_per_example"] = float(
            metrics["latency_median_ms"] * 1000 / len(y_true)
        )
    return metrics, y_true, y_pred


def variant_storage(
    base: nn.Module,
    name: str,
    masks: dict[str, torch.Tensor] | None,
    method: str,
    bits: int,
) -> dict:
    parameters = named_parameter_map(base)
    eligible = set(eligible_names(name))
    fp32_bytes = sum(parameter.numel() * 4 for parameter in parameters.values())
    dense_bytes = 0
    sparse_bytes = 0
    nonzero = 0
    total_prunable = 0
    scales = 0
    for param_name, parameter in parameters.items():
        if param_name not in eligible:
            dense_bytes += parameter.numel() * 4
            sparse_bytes += parameter.numel() * 4
            continue
        quantized, _ = symmetric_quantize(parameter.detach(), bits)
        mask = masks.get(param_name) if masks else None
        if mask is None:
            effective = torch.ones_like(parameter)
        else:
            effective = mask
        count = int(effective.sum().item())
        total_prunable += parameter.numel()
        nonzero += count
        dense_bytes += quantized.numel() * (bits // 8) + 4
        scales += 4
        # COO int32 estimate: one int8 value plus one int32 flat index per nonzero.
        sparse_bytes += count * (bits // 8 + 4) + 16
    sparsity = 1.0 - (nonzero / total_prunable) if total_prunable else 0.0
    return {
        "fp32_parameter_bytes": int(fp32_bytes),
        "dense_fp32_bytes": int(fp32_bytes),
        "dense_quantized_bytes": int(dense_bytes),
        "physical_bytes_dense": int(fp32_bytes if method == "fp32" else dense_bytes),
        "physical_bytes_sparse_coo_int32_estimate": int(sparse_bytes),
        "quantized_scale_bytes": int(scales),
        "prunable_parameters": int(total_prunable),
        "nonzero_parameters": int(nonzero),
        "logical_sparsity": float(sparsity),
        "method": method,
        "bits": bits,
        "sparse_format": "COO with int32 flat indices (estimate)",
    }


def export_variant(
    path: Path,
    base: nn.Module,
    name: str,
    masks: dict[str, torch.Tensor] | None,
    method: str,
    bits: int,
) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    scales: dict[str, float] = {}
    parameters = named_parameter_map(base)
    eligible = set(eligible_names(name))
    for param_name, parameter in parameters.items():
        key = param_name.replace(".", "__")
        if method == "fp32":
            arrays[key] = parameter.detach().cpu().numpy().astype(np.float32)
        elif param_name in eligible:
            transformed = parameter.detach()
            mask = masks.get(param_name) if masks else None
            if method in {"pq", "pq_sequential_control"} and mask is not None:
                transformed = transformed * mask
            quantized, scale = symmetric_quantize(transformed, bits)
            arrays[key] = quantized.cpu().numpy()
            scales[key] = float(scale.item())
        else:
            arrays[key] = parameter.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(path, **arrays)
    return {
        "artifact": str(path),
        "artifact_file_bytes": path.stat().st_size,
        "scales": scales,
        **variant_storage(base, name, masks, method, bits),
    }


def train_iterative(
    base: nn.Module,
    name: str,
    method: str,
    masks: dict[str, torch.Tensor],
    loaders: dict[str, DataLoader],
    class_weights: torch.Tensor,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> tuple[dict, dict]:
    wrapper = CompressedModel(base, name, masks, method).to(device)
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best_score = (-1.0, -1.0)
    best_state = None
    history = []
    updates_per_batch = 2 if method == "qp" else 1
    for epoch in range(1, epochs + 1):
        wrapper.train()
        loss_sum = 0.0
        batch_count = 0
        for features, labels in loaders["train"]:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(wrapper(features), labels)
            loss.backward()
            optimizer.step()
            project_masks(base, masks)
            total_loss = float(loss.item())
            if method == "qp":
                optimizer.zero_grad(set_to_none=True)
                second_loss = criterion(wrapper(features), labels)
                second_loss.backward()
                optimizer.step()
                project_masks(base, masks)
                total_loss += float(second_loss.item())
            loss_sum += total_loss * len(labels)
            batch_count += 1
        wrapper.eval()
        val_metrics, _, _ = evaluate_model(wrapper, loaders["validation"], device)
        row = {
            "epoch": epoch,
            "loss": loss_sum / len(loaders["train"].dataset),
            "validation_balanced_accuracy": val_metrics["balanced_accuracy"],
            "validation_macro_f1": val_metrics["macro_f1"],
            "updates_per_batch": updates_per_batch,
            "batches": batch_count,
        }
        history.append(row)
        score = (row["validation_balanced_accuracy"], row["validation_macro_f1"])
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in base.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Nenhum estado foi salvo durante o ajuste fino")
    base.load_state_dict(best_state)
    project_masks(base, masks)
    return {"history": history, "updates_per_batch": updates_per_batch}, {"best_score": best_score}


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.torch_threads)
    device = torch.device("cpu")
    data, metadata = load_archive(args.features)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    rows: list[dict] = []
    for model_name, model_dir, prefix in (
        ("mlp", args.mlp_dir, "mlp"),
        ("tiny_transformer", args.transformer_dir, "transformer"),
    ):
        for seed in args.seeds:
            seed_everything(seed)
            checkpoint = model_dir / f"{prefix}_seed_{seed}.pt"
            baseline = load_checkpoint(model_name, checkpoint, device)
            loaders = make_loaders(data, args.batch_size, seed)
            class_weights = torch.tensor(
                [metadata["class_weights"][label] for label in LABELS],
                dtype=torch.float32,
                device=device,
            )
            base_export_dir = output_dir / "exports" / model_name / f"seed_{seed}"
            base_wrapper = CompressedModel(baseline, model_name, None, "fp32").to(device)
            fp32_metrics, _, _ = evaluate_model(
                base_wrapper, loaders["test"], device, repeats=args.repeats
            )
            fp32_storage = variant_storage(baseline, model_name, None, "fp32", 8)
            fp32_path = base_export_dir / "fp32.npz"
            fp32_export = export_variant(
                fp32_path, baseline, model_name, None, "fp32", 8
            )
            rows.append({
                "model": model_name,
                "seed": seed,
                "method": "fp32",
                "sparsity_target": 0.0,
                "method_status": "baseline",
                "operation_order": "none",
                "calibration_split": "not_applicable",
                **{key: fp32_metrics[key] for key in ("accuracy", "balanced_accuracy", "macro_f1", "confusion_matrix", "latency_median_ms", "latency_p95_ms", "latency_us_per_example")},
                **fp32_storage,
                "export_file_bytes": fp32_export["artifact_file_bytes"],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "history": [],
            })
            int8_wrapper = CompressedModel(baseline, model_name, None, "int8_weight_ptq").to(device)
            int8_metrics, _, _ = evaluate_model(
                int8_wrapper, loaders["test"], device, repeats=args.repeats
            )
            int8_storage = variant_storage(
                baseline, model_name, None, "int8_weight_ptq", 8
            )
            int8_export = export_variant(
                base_export_dir / "int8_weight_ptq.npz",
                baseline,
                model_name,
                None,
                "int8_weight_ptq",
                8,
            )
            rows.append({
                "model": model_name,
                "seed": seed,
                "method": "int8_weight_ptq",
                "sparsity_target": 0.0,
                "method_status": "ptq_weight_only_control",
                "operation_order": "quantization",
                "calibration_split": "train_weights",
                **{key: int8_metrics[key] for key in ("accuracy", "balanced_accuracy", "macro_f1", "confusion_matrix", "latency_median_ms", "latency_p95_ms", "latency_us_per_example")},
                **int8_storage,
                "export_file_bytes": int8_export["artifact_file_bytes"],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "history": [],
            })
            for sparsity in args.sparsities:
                masks = build_masks(baseline, model_name, sparsity)
                for method, status, order in (
                    ("pq_sequential_control", "sequential_baseline", "prune_then_quantize"),
                    ("qp_sequential_control", "sequential_baseline", "quantize_then_prune"),
                ):
                    control = load_checkpoint(model_name, checkpoint, device)
                    control_masks = {key: value.to(device) for key, value in masks.items()}
                    control_wrapper = CompressedModel(
                        control, model_name, control_masks, method
                    ).to(device)
                    metrics, _, _ = evaluate_model(
                        control_wrapper, loaders["test"], device, repeats=args.repeats
                    )
                    storage = variant_storage(control, model_name, control_masks, method, 8)
                    exported = export_variant(
                        base_export_dir / f"{method}_{int(sparsity * 100):02d}.npz",
                        control,
                        model_name,
                        control_masks,
                        method,
                        8,
                    )
                    rows.append({
                        "model": model_name,
                        "seed": seed,
                        "method": method,
                        "sparsity_target": sparsity,
                        "method_status": status,
                        "operation_order": order,
                        "calibration_split": "train_weights",
                        **{key: metrics[key] for key in ("accuracy", "balanced_accuracy", "macro_f1", "confusion_matrix", "latency_median_ms", "latency_p95_ms", "latency_us_per_example")},
                        **storage,
                        "export_file_bytes": exported["artifact_file_bytes"],
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": sha256(checkpoint),
                        "history": [],
                    })
                for method, order in (("pq", "prune_then_quantize"), ("qp", "quantize_then_prune")):
                    seed_everything(seed)
                    trained = load_checkpoint(model_name, checkpoint, device)
                    train_masks = {key: value.to(device) for key, value in build_masks(trained, model_name, sparsity).items()}
                    train_info, selection = train_iterative(
                        trained,
                        model_name,
                        method,
                        train_masks,
                        loaders,
                        class_weights,
                        device,
                        args.finetune_epochs,
                        args.learning_rate,
                        args.weight_decay,
                    )
                    trained_wrapper = CompressedModel(
                        trained, model_name, train_masks, method
                    ).to(device)
                    metrics, _, _ = evaluate_model(
                        trained_wrapper, loaders["test"], device, repeats=args.repeats
                    )
                    storage = variant_storage(trained, model_name, train_masks, method, 8)
                    export_path = base_export_dir / f"{method}_{int(sparsity * 100):02d}.npz"
                    exported = export_variant(
                        export_path, trained, model_name, train_masks, method, 8
                    )
                    model_path = output_dir / "checkpoints" / model_name / f"{method}_{int(sparsity * 100):02d}_seed_{seed}.pt"
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "model_state": trained.state_dict(),
                            "model": model_name,
                            "method": method,
                            "sparsity_target": sparsity,
                            "masks": {key: value.cpu() for key, value in train_masks.items()},
                            "history": train_info["history"],
                            "selection": selection,
                            "source_protocol": "Fernandes-Kung IJCNN 2021 adapted",
                        },
                        model_path,
                    )
                    rows.append({
                        "model": model_name,
                        "seed": seed,
                        "method": method,
                        "sparsity_target": sparsity,
                        "method_status": "adaptation_iterative",
                        "operation_order": order,
                        "calibration_split": "train_weights",
                        **{key: metrics[key] for key in ("accuracy", "balanced_accuracy", "macro_f1", "confusion_matrix", "latency_median_ms", "latency_p95_ms", "latency_us_per_example")},
                        **storage,
                        "export_file_bytes": exported["artifact_file_bytes"],
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": sha256(checkpoint),
                        "history": train_info["history"],
                    })
                    print(json.dumps({
                        "model": model_name,
                        "seed": seed,
                        "method": method,
                        "sparsity": sparsity,
                        "accuracy": metrics["accuracy"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "macro_f1": metrics["macro_f1"],
                        "dense_bytes": storage["dense_quantized_bytes"],
                        "sparse_coo_bytes": storage["physical_bytes_sparse_coo_int32_estimate"],
                    }, ensure_ascii=False))
    (output_dir / "compression_results.json").write_text(
        json.dumps(
            {
                "protocol": {
                    "labels": LABELS,
                    "bits": 8,
                    "quantization": "symmetric per tensor, weight-only PTQ/fake-quantized forward",
                    "pruning": "global magnitude fixed mask on eligible weight tensors",
                    "sparsities": args.sparsities,
                    "seeds": args.seeds,
                    "finetune_epochs": args.finetune_epochs,
                    "test_used_after_selection": True,
                    "same_test_for_all": True,
                    "method_status": {
                        "int8_weight_ptq": "ptq_weight_only_control",
                        "pq": "adaptation_iterative",
                        "qp": "adaptation_iterative",
                        "pq_sequential_control": "sequential_baseline",
                        "qp_sequential_control": "sequential_baseline",
                    },
                },
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    fields = sorted({key for row in rows for key in row if key != "confusion_matrix" and key != "history"})
    with (output_dir / "compression_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    print(json.dumps({"rows": len(rows), "output": str(output_dir / "compression_results.json")}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("data/processed/features.npz"))
    parser.add_argument("--mlp-dir", type=Path, default=Path("models/mlp"))
    parser.add_argument("--transformer-dir", type=Path, default=Path("models/transformer"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/compression"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--sparsities", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--finetune-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
