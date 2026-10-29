"""Upload the MLP POC to MicroPython and compare logits with the host."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def mpremote(port: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "mpremote", "connect", port, *args], check=True, text=True, capture_output=True)


def run(port: str, artifact_dir: Path) -> dict:
    mpremote(port, "fs", "cp", "firmware/micropython/mlp_infer.py", ":mlp_infer.py")
    mpremote(port, "fs", "cp", str(artifact_dir / "mlp_fp32.bin"), ":mlp_fp32.bin")
    mpremote(port, "fs", "cp", str(artifact_dir / "input.f32"), ":input.f32")
    result = mpremote(port, "exec", "import mlp_infer; mlp_infer.infer()")
    device = json.loads(result.stdout.strip().splitlines()[-1])
    reference = json.loads((artifact_dir / "reference.json").read_text(encoding="utf-8"))
    errors = [abs(a - b) for a, b in zip(device["logits"], reference["logits"])]
    report = {
        "device": device,
        "reference": reference,
        "max_abs_logit_error": max(errors),
        "mean_abs_logit_error": sum(errors) / len(errors),
        "predicted_label_match": device["predicted_index"] == reference["labels"].index(reference["predicted_label"]),
        "within_tolerance": max(errors) <= reference["tolerance_abs"],
    }
    (artifact_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--artifact-dir", type=Path, default=Path("experiments/embedded/mlp_poc"))
    args = parser.parse_args()
    print(json.dumps(run(args.port, args.artifact_dir), indent=2))


if __name__ == "__main__":
    main()
