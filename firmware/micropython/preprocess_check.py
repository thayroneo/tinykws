"""Check the training-time MFCC normalization on MicroPython."""
import json
import struct
import time


def _read(path, count):
    with open(path, "rb") as stream:
        data = stream.read()
    return [struct.unpack_from("<f", data, i * 4)[0] for i in range(count)]


def check():
    raw = _read("mfcc_raw.f32", 520)
    mean = _read("mfcc_mean.f32", 13)
    std = _read("mfcc_std.f32", 13)
    start = time.ticks_us()
    normalized = [0.0] * 520
    for i, value in enumerate(raw):
        normalized[i] = (value - mean[i % 13]) / std[i % 13]
    elapsed_us = time.ticks_diff(time.ticks_us(), start)
    result = {"normalized": normalized, "normalization_latency_us": elapsed_us}
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    check()
