"""Coleta exemplos rotulados independentes para T4.3 usando o firmware DMA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.embedded.capture_dma import capture as capture_c
from scripts.embedded.capture_dma_micropython import capture as capture_micropython


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/bitdoglab_eval"))
    parser.add_argument("--takes-per-label", type=int, default=5)
    parser.add_argument("--labels", nargs="+", default=["yes", "no", "stop", "silence", "noise"])
    parser.add_argument(
        "--backend",
        choices=["c", "micropython"],
        default="c",
        help="c para o firmware UF2 ou micropython para DMA via registradores",
    )
    args = parser.parse_args()
    manifest = args.output_dir / "manifest.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with manifest.open("a", encoding="utf-8") as stream:
        for label in args.labels:
            for take in range(args.takes_per_label):
                input(f"Pressione Enter e produza {label!r} ({take + 1}/{args.takes_per_label})...")
                capture_fn = capture_c if args.backend == "c" else capture_micropython
                result = capture_fn(args.port, args.output_dir / label, 16000, 1000)
                metadata = json.loads(Path(result["metadata"]).read_text(encoding="utf-8"))
                rate_error = abs(metadata["effective_rate_hz"] - 16000) / 16000
                if rate_error > 0.01:
                    raise RuntimeError(
                        f"Captura rejeitada para {label}: taxa efetiva "
                        f"{metadata['effective_rate_hz']:.2f} Hz"
                    )
                metadata.update({"label": label, "take": take + 1, "evaluation_only": True})
                stream.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                stream.flush()
    print(json.dumps({"manifest": str(manifest), "labels": args.labels}))


if __name__ == "__main__":
    main()
