"""Run the register-level DMA capture on a MicroPython board."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path


def mpremote(port: str, *args: str) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-m", "mpremote", "connect", port, *args]
    return subprocess.run(command, check=True, text=True, capture_output=True)


def capture(port: str, output_dir: Path, rate_hz: int, duration_ms: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    device_stem = "/capture_dma"
    mpremote(port, "fs", "cp", "firmware/micropython/capture_dma.py", ":capture_dma.py")
    result = mpremote(
        port,
        "exec",
        f"import capture_dma; capture_dma.capture_once({rate_hz}, {duration_ms}, '{device_stem}')",
    )
    metadata = json.loads(result.stdout.strip().splitlines()[-1])
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    stem = output_dir / f"capture_dma_micropython_{stamp}"
    mpremote(port, "fs", "cp", ":capture_dma.raw", str(stem) + ".raw")
    mpremote(port, "fs", "cp", ":capture_dma.json", str(stem) + ".json")
    raw_path = Path(str(stem) + ".raw")
    raw = bytearray(raw_path.read_bytes())
    for index in range(0, len(raw), 2):
        value = min(65535, (raw[index] | (raw[index + 1] << 8)) << 4)
        raw[index] = value & 0xFF
        raw[index + 1] = value >> 8
    metadata["serial_port"] = port
    metadata["host_retrieved_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["raw_bytes"] = raw_path.stat().st_size
    metadata["raw_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    metadata["wav_sample_rate_hz"] = int(round(metadata["effective_rate_hz"]))
    Path(str(stem) + ".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    with wave.open(str(stem) + ".wav", "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(metadata["wav_sample_rate_hz"])
        stream.writeframes(raw)
    return {"raw": str(stem) + ".raw", "metadata": str(stem) + ".json"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/bitdoglab_dma"))
    parser.add_argument("--rate", type=int, default=16000)
    parser.add_argument("--duration-ms", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(capture(args.port, args.output_dir, args.rate, args.duration_ms)))


if __name__ == "__main__":
    main()
