"""Captura ADC temporizada por FIFO/DMA via USB CDC."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import serial


def read_until(port: serial.Serial, prefix: bytes, timeout_s: float = 5.0) -> bytes:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        line = port.readline()
        if line.startswith(prefix):
            return line
    raise TimeoutError(f"Não recebi {prefix!r} em {timeout_s}s")


def capture(port_name: str, output_dir: Path, rate_hz: int, duration_ms: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    stem = f"capture_dma_{stamp}"
    with serial.Serial(port_name, 115200, timeout=0.2) as port:
        port.reset_input_buffer()
        port.write(f"CAPTURE {rate_hz} {duration_ms}\n".encode("ascii"))
        header = read_until(port, b"VTKWS_READY ", timeout_s=8.0)
        metadata = json.loads(header[len(b"VTKWS_READY ") :].decode("ascii"))
        expected_bytes = int(metadata["sample_count"]) * 2
        raw = port.read(expected_bytes)
        if len(raw) != expected_bytes:
            raise RuntimeError(f"Recebi {len(raw)} bytes; esperava {expected_bytes}")
        read_until(port, b"VTKWS_DONE", timeout_s=8.0)

    raw_path = output_dir / f"{stem}.raw"
    json_path = output_dir / f"{stem}.json"
    wav_path = output_dir / f"{stem}.wav"
    raw_path.write_bytes(raw)
    metadata.update(
        {
            "source": "BitDogLab RP2040 ADC FIFO/DMA",
            "serial_port": port_name,
            "host_retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "raw_bytes": len(raw),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "wav_sample_rate_hz": int(round(metadata["effective_rate_hz"])),
        }
    )
    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    samples = bytearray(raw)
    for index in range(0, len(samples), 2):
        value = samples[index] | (samples[index + 1] << 8)
        value = min(65535, value << 4)
        samples[index] = value & 0xFF
        samples[index + 1] = value >> 8
    with wave.open(str(wav_path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(metadata["wav_sample_rate_hz"])
        stream.writeframes(samples)
    return {"raw": str(raw_path), "wav": str(wav_path), "metadata": str(json_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/bitdoglab"))
    parser.add_argument("--rate", type=int, default=16000)
    parser.add_argument("--duration-ms", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(capture(args.port, args.output_dir, args.rate, args.duration_ms)))


if __name__ == "__main__":
    main()
