"""MicroPython diagnostic runner for the VTKTR1 TinyTransformer export."""
import gc
import json
import math
import struct
import time
from array import array


MAGIC = b"VTKTR1\0\0"
SPECS = (
    ("input_projection.weight", 416), ("input_projection.bias", 32),
    ("position", 1280), ("attention.in_proj_weight", 3072),
    ("attention.in_proj_bias", 96), ("attention.out_proj.weight", 1024),
    ("attention.out_proj.bias", 32), ("norm1.weight", 32),
    ("norm1.bias", 32), ("ffn.0.weight", 2048), ("ffn.0.bias", 64),
    ("ffn.2.weight", 2048), ("ffn.2.bias", 32), ("norm2.weight", 32),
    ("norm2.bias", 32), ("classifier.weight", 160), ("classifier.bias", 5),
)


def _read(stream, size):
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("truncated transformer export")
    return data


def _values_from_stream(stream, count):
    values = array("f")
    for _ in range(count):
        values.append(struct.unpack("<f", _read(stream, 4))[0])
    return values


def _values(data):
    return [struct.unpack_from("<f", data, index * 4)[0] for index in range(len(data) // 4)]


def _linear(weight, bias, rows, cols, vector):
    result = array("f")
    for row in range(rows):
        value = bias[row]
        offset = row * cols
        for col in range(cols):
            value += weight[offset + col] * vector[col]
        result.append(value)
    return result


def _norm(values, weight, bias):
    output = array("f")
    for _ in range(len(values)):
        output.append(0.0)
    for token in range(40):
        offset = token * 32
        mean = sum(values[offset + col] for col in range(32)) / 32.0
        variance = sum((values[offset + col] - mean) ** 2 for col in range(32)) / 32.0
        inverse = 1.0 / math.sqrt(variance + 1e-5)
        for col in range(32):
            output[offset + col] = (values[offset + col] - mean) * inverse * weight[col] + bias[col]
    return output


def infer(weights_path="transformer_fp32.bin", input_path="input.f32"):
    with open(weights_path, "rb") as stream:
        if stream.read(8) != MAGIC:
            raise ValueError("invalid transformer export")
        _read(stream, 24)
        tensors = {}
        for name, count in SPECS:
            tensors[name] = _values_from_stream(stream, count)
    x = _values(open(input_path, "rb").read())
    gc.collect()
    mem_before = gc.mem_free()
    start_us = time.ticks_us()
    projection = tensors["input_projection.weight"]
    h = array("f")
    for token in range(40):
        row = _linear(projection, tensors["input_projection.bias"], 32, 13, x[token * 13:(token + 1) * 13])
        for col in range(32):
            row[col] += tensors["position"][token * 32 + col]
        h.extend(row)
    qkv = array("f")
    for token in range(40):
        qkv.extend(_linear(tensors["attention.in_proj_weight"], tensors["attention.in_proj_bias"], 96, 32, h[token * 32:(token + 1) * 32]))
    attended = array("f")
    for _ in range(40 * 32):
        attended.append(0.0)
    for head in range(2):
        for query in range(40):
            scores = []
            for key in range(40):
                score = 0.0
                for col in range(16):
                    score += qkv[query * 96 + head * 16 + col] * qkv[key * 96 + 32 + head * 16 + col]
                scores.append(score / math.sqrt(16.0))
            maximum = max(scores)
            scores = [math.exp(value - maximum) for value in scores]
            total = sum(scores)
            for key in range(40):
                scores[key] /= total
                for col in range(16):
                    attended[query * 32 + head * 16 + col] += scores[key] * qkv[key * 96 + 64 + head * 16 + col]
    for token in range(40):
        projected = _linear(tensors["attention.out_proj.weight"], tensors["attention.out_proj.bias"], 32, 32, attended[token * 32:(token + 1) * 32])
        for col in range(32):
            h[token * 32 + col] += projected[col]
    h = _norm(h, tensors["norm1.weight"], tensors["norm1.bias"])
    for token in range(40):
        hidden = _linear(tensors["ffn.0.weight"], tensors["ffn.0.bias"], 64, 32, h[token * 32:(token + 1) * 32])
        hidden = [value if value > 0.0 else 0.0 for value in hidden]
        projected = _linear(tensors["ffn.2.weight"], tensors["ffn.2.bias"], 32, 64, hidden)
        for col in range(32):
            h[token * 32 + col] += projected[col]
    h = _norm(h, tensors["norm2.weight"], tensors["norm2.bias"])
    pooled = [sum(h[token * 32 + col] for token in range(40)) / 40.0 for col in range(32)]
    logits = _linear(tensors["classifier.weight"], tensors["classifier.bias"], 5, 32, pooled)
    result = {"logits": [float(value) for value in logits], "predicted_index": max(range(5), key=lambda i: logits[i]), "latency_us": time.ticks_diff(time.ticks_us(), start_us), "mem_free_before": mem_before, "mem_free_after": gc.mem_free(), "weights_bytes": 41748, "input_bytes": 2080}
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    infer()
