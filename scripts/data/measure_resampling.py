"""Measure host-side resampling cost for a captured WAV."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def measure(path: Path, target_rate_hz: int, repeats: int, duration_seconds: float | None = None) -> dict:
    audio, source_rate_hz = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if duration_seconds is not None:
        audio = audio[: int(round(duration_seconds * source_rate_hz))]
    # Warm up numba-backed filters so compilation time is not counted.
    librosa.resample(
        audio,
        orig_sr=int(source_rate_hz),
        target_sr=target_rate_hz,
        res_type="kaiser_fast",
    )
    times = []
    output = None
    for _ in range(repeats):
        start = time.perf_counter()
        output = librosa.resample(
            audio,
            orig_sr=int(source_rate_hz),
            target_sr=target_rate_hz,
            res_type="kaiser_fast",
        )
        times.append(time.perf_counter() - start)
    mean_s = float(np.mean(times))
    duration_s = len(audio) / float(source_rate_hz)
    return {
        "input": str(path),
        "source_rate_hz": int(source_rate_hz),
        "target_rate_hz": target_rate_hz,
        "input_samples": int(len(audio)),
        "output_samples": int(len(output)),
        "input_duration_s": duration_s,
        "repeats": repeats,
        "resample_time_s_mean": mean_s,
        "resample_time_s_std": float(np.std(times)),
        "resample_time_s_min": float(np.min(times)),
        "resample_time_s_max": float(np.max(times)),
        "realtime_factor": duration_s / mean_s if mean_s else None,
        "resample_type": "librosa kaiser_fast",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", type=Path)
    parser.add_argument("--target-rate", type=int, default=16000)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = measure(args.wav, args.target_rate, args.repeats, args.duration_seconds)
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
