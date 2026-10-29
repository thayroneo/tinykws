"""Weight-only INT8 MLP inference for the RP2040 proof of concept."""
import gc
import json
import struct
import time


MAGIC = b"VTKMLP8\0"


def _read(stream, size):
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("truncated INT8 MLP export")
    return data


def _f32(data, index):
    return struct.unpack_from("<f", data, index * 4)[0]


def _i8(data, index):
    return struct.unpack_from("<b", data, index)[0]


def infer(weights_path="mlp_int8.bin", input_path="input.f32"):
    with open(weights_path, "rb") as stream:
        if stream.read(8) != MAGIC:
            raise ValueError("invalid INT8 MLP export")
        input_dim, hidden1, hidden2, classes = struct.unpack("<4I", _read(stream, 16))
        layers = []
        for rows, cols in ((hidden1, input_dim), (hidden2, hidden1), (classes, hidden2)):
            weights = _read(stream, rows * cols)
            scale = struct.unpack("<f", _read(stream, 4))[0]
            bias = _read(stream, rows * 4)
            layers.append((weights, scale, bias, rows, cols))
    with open(input_path, "rb") as stream:
        x_data = _read(stream, input_dim * 4)
    gc.collect()
    mem_before = gc.mem_free()
    start_us = time.ticks_us()
    h1 = [0.0 for _ in range(hidden1)]
    h2 = [0.0 for _ in range(hidden2)]
    logits = [0.0 for _ in range(classes)]
    activations = x_data
    outputs = (h1, h2, logits)
    for layer_index, (weights, scale, bias, rows, cols) in enumerate(layers):
        output = outputs[layer_index]
        for row in range(rows):
            accumulator = 0.0
            offset = row * cols
            for col in range(cols):
                accumulator += _i8(weights, offset + col) * _f32(activations, col)
            value = _f32(bias, row) + scale * accumulator
            output[row] = value if layer_index < 2 and value > 0.0 else value
        if layer_index < 2:
            activations = bytearray()
            for value in output:
                activations.extend(struct.pack("<f", value if value > 0.0 else 0.0))
    latency_us = time.ticks_diff(time.ticks_us(), start_us)
    mem_after = gc.mem_free()
    result = {
        "logits": [float(value) for value in logits],
        "predicted_index": max(range(classes), key=lambda i: logits[i]),
        "latency_us": latency_us,
        "mem_free_before": mem_before,
        "mem_free_after": mem_after,
        "weights_bytes": 35904,
        "input_bytes": input_dim * 4,
    }
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    infer()
