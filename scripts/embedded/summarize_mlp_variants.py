"""Merge host quality metrics with measured RP2040 deployment results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(compression: Path, reports_dir: Path, output: Path) -> dict:
    host = json.loads(compression.read_text(encoding="utf-8"))
    rows = []
    for row in host["rows"]:
        if row.get("model") != "mlp" or row.get("seed") != 20261022:
            continue
        method = row["method"]
        if method == "fp32":
            continue
        name = method if method == "int8_weight_ptq" else f"{method}_{int(row['sparsity_target'] * 100):02d}"
        report_path = reports_dir / name / "report.json"
        if not report_path.exists():
            continue
        device = json.loads(report_path.read_text(encoding="utf-8"))
        rows.append({
            "variant": name,
            "method": method,
            "sparsity_target": row["sparsity_target"],
            "host_accuracy": row["accuracy"],
            "host_balanced_accuracy": row["balanced_accuracy"],
            "host_macro_f1": row["macro_f1"],
            "host_artifact_bytes": row["export_file_bytes"],
            "device_binary_bytes": device["reference"]["binary_bytes"],
            "device_latency_us": device["device"]["latency_us"],
            "device_mem_free_before": device["device"]["mem_free_before"],
            "device_mem_free_after": device["device"]["mem_free_after"],
            "device_max_abs_logit_error": device["max_abs_logit_error"],
            "device_predicted_index_match": device["predicted_index_match"],
            "device_within_tolerance": device["within_tolerance"],
        })
    result = {
        "protocol": {
            "host_seed": 20261022,
            "device": "BitDogLab RP2040 via COM5",
            "device_input": "one fixed normalized MFCC vector used for T5.1 equivalence",
            "classification_metrics": "full host test split; device equivalence checked before associating deployment metrics",
            "tolerance_abs": 1e-4,
        },
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compression", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.compression, args.reports_dir, args.output)
    print(json.dumps({"rows": len(result["rows"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
