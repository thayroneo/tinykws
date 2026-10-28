"""Valida integridade, taxa e faixa de uma captura recuperada."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def validate(raw_path: Path, metadata_path: Path, target_rate_hz: int = 16000) -> dict:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw = raw_path.read_bytes()
    expected_bytes = int(metadata["sample_count"]) * 2
    digest = hashlib.sha256(raw).hexdigest()
    samples = np.frombuffer(raw, dtype="<u2")
    effective_rate = float(metadata["effective_rate_hz"])
    rate_error = abs(effective_rate - target_rate_hz) / target_rate_hz
    report = {
        "raw_bytes_ok": len(raw) == expected_bytes,
        "sha256_ok": digest == metadata.get("raw_sha256", digest),
        "sample_count_ok": len(samples) == int(metadata["sample_count"]),
        "rate_hz": effective_rate,
        "rate_error_fraction": rate_error,
        "rate_ok_for_mfcc": rate_error <= 0.01,
        "clipped_low": bool((samples == 0).any()),
        "clipped_high": bool((samples >= 65535).any()),
        "sample_min": int(samples.min()) if len(samples) else None,
        "sample_max": int(samples.max()) if len(samples) else None,
        "sample_mean": float(samples.mean()) if len(samples) else None,
    }
    report["valid_for_mfcc"] = all(
        (report[key] for key in ("raw_bytes_ok", "sha256_ok", "sample_count_ok", "rate_ok_for_mfcc"))
    ) and not report["clipped_low"] and not report["clipped_high"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("metadata", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.raw, args.metadata), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
