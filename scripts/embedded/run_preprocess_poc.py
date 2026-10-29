"""Run and compare the fixed preprocessing vector on MicroPython."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def mpremote(port: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "mpremote", "connect", port, *args], check=True, text=True, capture_output=True)


def run(port: str, artifact_dir: Path) -> dict:
    mpremote(port, "fs", "cp", "firmware/micropython/preprocess_check.py", ":preprocess_check.py")
    for name in ("mfcc_raw.f32", "mfcc_mean.f32", "mfcc_std.f32"):
        mpremote(port, "fs", "cp", str(artifact_dir / name), ":" + name)
    device = json.loads(mpremote(port, "exec", "import preprocess_check; preprocess_check.check()").stdout.strip().splitlines()[-1])
    reference = json.loads((artifact_dir / "preprocess_reference.json").read_text(encoding="utf-8"))
    errors = [abs(a - b) for a, b in zip(device["normalized"], reference["normalized"])]
    report = {"normalization_latency_us": device["normalization_latency_us"], "max_abs_error": max(errors), "mean_abs_error": sum(errors) / len(errors), "within_tolerance": max(errors) <= reference["tolerance_abs"]}
    (artifact_dir / "preprocess_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--artifact-dir", type=Path, default=Path("experiments/embedded/mlp_poc"))
    args = parser.parse_args()
    print(json.dumps(run(args.port, args.artifact_dir), indent=2))


if __name__ == "__main__":
    main()
