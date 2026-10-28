"""Compara metadados das capturas MicroPython e FIFO/DMA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("experiments/device_capture/comparison.json"))
    args = parser.parse_args()
    rows = []
    for path in args.metadata:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "metadata": str(path),
                "source": metadata.get("source", metadata.get("firmware", "unknown")),
                "requested_rate_hz": metadata.get("requested_rate_hz"),
                "effective_rate_hz": metadata.get("effective_rate_hz"),
                "interval_mean_us": metadata.get("interval_mean_us"),
                "interval_jitter_rms_us": metadata.get("interval_jitter_rms_us"),
                "raw_bytes": metadata.get("raw_bytes"),
                "raw_sha256": metadata.get("raw_sha256"),
            }
        )
    payload = {"rows": rows, "target_rate_hz": 16000, "notes": "Compare somente capturas com mesmo duration_ms e sample_count."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
