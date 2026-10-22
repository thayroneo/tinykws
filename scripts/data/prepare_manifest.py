"""Monta e valida o manifesto do Google Speech Commands.

Classes públicas: yes, no, stop, unknown e silence.
A classe silence é gerada a partir de _background_noise_ em janelas de 1 s.
Arquivos de ruído são separados por origem entre train/validation/test para
não colocar segmentos do mesmo áudio em divisões diferentes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import soundfile as sf

TARGETS = ("yes", "no", "stop")
NOISE_DIR = "_background_noise_"
EXTENSIONS = {".wav", ".flac", ".ogg"}
SPLITS = ("train", "validation", "test")


def read_list(path: Path) -> set[str]:
    return {
        line.strip().replace("\\", "/")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_silence(root: Path, output_root: Path, sample_rate: int) -> list[dict]:
    noise_files = sorted((root / NOISE_DIR).glob("*.wav"))
    if len(noise_files) < 3:
        raise SystemExit("São necessários pelo menos três arquivos em _background_noise_.")
    # Separação por arquivo de origem: 2/3 treino, 1/6 validação, 1/6 teste.
    assignments = {}
    for index, source in enumerate(noise_files):
        split = "train" if index < len(noise_files) - 2 else "validation" if index == len(noise_files) - 2 else "test"
        assignments[source] = split

    rows = []
    for source in noise_files:
        audio, rate = sf.read(source, dtype="float32", always_2d=False)
        if rate != sample_rate:
            raise SystemExit(f"Taxa inesperada em {source}: {rate} Hz")
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        split = assignments[source]
        split_dir = output_root / "silence" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        total = len(audio) // sample_rate
        for start_second in range(total):
            start = start_second * sample_rate
            segment = audio[start : start + sample_rate]
            out = split_dir / f"{source.stem}_{start_second:04d}.wav"
            sf.write(out, segment, sample_rate, subtype="PCM_16")
            rows.append({
                "path": out.relative_to(output_root.parent).as_posix(),
                "label": "silence",
                "split": split,
                "sha256": sha256(out),
                "source": source.relative_to(root).as_posix(),
                "start_second": start_second,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/manifests/speech_commands.jsonl"))
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    validation = read_list(root / "validation_list.txt")
    testing = read_list(root / "testing_list.txt")
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        rel = path.relative_to(root).as_posix()
        source_class = path.parent.name
        if source_class == NOISE_DIR:
            continue
        label = source_class if source_class in TARGETS else "unknown"
        split = "validation" if rel in validation else "test" if rel in testing else "train"
        rows.append({"path": rel, "label": label, "split": split, "sha256": sha256(path)})

    silence_rows = make_silence(root, root.parent.parent / "processed", args.sample_rate)
    rows.extend(silence_rows)
    if not rows:
        raise SystemExit(f"Nenhum áudio encontrado em {root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    classes = {row["label"] for row in rows}
    if classes != {"yes", "no", "stop", "unknown", "silence"}:
        raise SystemExit(f"Classes inválidas: {sorted(classes)}")
    paths = [row["path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise SystemExit("Há caminhos duplicados no manifesto.")
    print(f"Manifesto: {output} ({len(rows)} arquivos)")
    for (split, label), count in sorted(Counter((r["split"], r["label"]) for r in rows).items()):
        print(f"{split:10} {label:8} {count}")


if __name__ == "__main__":
    main()
