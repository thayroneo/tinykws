"""Executa a captura MicroPython e recupera WAV, bruto e metadados."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import wave
from datetime import datetime, timezone
from pathlib import Path


def mpremote(port: str, *arguments: str) -> None:
    command = [sys.executable, "-m", "mpremote", "connect", port, *arguments]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM5")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/bitdoglab"))
    parser.add_argument("--rate", type=int, default=16000)
    parser.add_argument("--duration-ms", type=int, default=1000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    firmware = Path("firmware/micropython/capture.py")
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    stem = f"capture_{timestamp}"
    with tempfile.TemporaryDirectory() as temporary:
        remote_raw = "capture.raw"
        remote_metadata = "capture.json"
        mpremote(args.port, "fs", "cp", str(firmware), ":capture.py")
        command = (
            "import capture; capture.capture("
            f"{remote_raw!r}, {remote_metadata!r}, {args.rate}, {args.duration_ms})"
        )
        mpremote(args.port, "exec", command)
        raw_path = Path(temporary) / remote_raw
        metadata_path = Path(temporary) / remote_metadata
        mpremote(args.port, "fs", "cp", f":{remote_raw}", str(raw_path))
        mpremote(args.port, "fs", "cp", f":{remote_metadata}", str(metadata_path))

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(
            {
                "host_retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "serial_port": args.port,
                "source": "BitDogLab RP2040 microphone ADC",
            }
        )
        output_raw = args.output_dir / f"{stem}.raw"
        output_json = args.output_dir / f"{stem}.json"
        output_wav = args.output_dir / f"{stem}.wav"
        output_raw.write_bytes(raw_path.read_bytes())
        metadata["wav_sample_rate_hz"] = int(round(metadata["effective_rate_hz"]))
        output_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        with wave.open(str(output_wav), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(metadata["wav_sample_rate_hz"])
            stream.writeframes(output_raw.read_bytes())

    print(json.dumps({"raw": str(output_raw), "wav": str(output_wav), "metadata": str(output_json)}))


if __name__ == "__main__":
    main()
