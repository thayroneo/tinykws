"""FP32 MLP inference for the deterministic T5.1 host comparison."""
import json
import struct


MAGIC = b"VTKMLP1\0"


def _read_f32(stream, count):
    data = stream.read(count * 4)
    if len(data) != count * 4:
        raise ValueError("truncated MLP export")
    return data


def _value(data, index):
    return struct.unpack_from("<f", data, index * 4)[0]


def infer(weights_path="mlp_fp32.bin", input_path="input.f32"):
    with open(weights_path, "rb") as stream:
        if stream.read(8) != MAGIC:
            raise ValueError("invalid MLP export")
        input_dim, hidden1, hidden2, classes = struct.unpack("<4I", stream.read(16))
        w1 = _read_f32(stream, hidden1 * input_dim)
        b1 = _read_f32(stream, hidden1)
        w2 = _read_f32(stream, hidden2 * hidden1)
        b2 = _read_f32(stream, hidden2)
        w3 = _read_f32(stream, classes * hidden2)
        b3 = _read_f32(stream, classes)
    with open(input_path, "rb") as stream:
        x = _read_f32(stream, input_dim)
    h1 = [0.0 for _ in range(hidden1)]
    h2 = [0.0 for _ in range(hidden2)]
    logits = [0.0 for _ in range(classes)]
    for row in range(hidden1):
        value = _value(b1, row)
        offset = row * input_dim
        for col in range(input_dim):
            value += _value(w1, offset + col) * _value(x, col)
        h1[row] = value if value > 0.0 else 0.0
    for row in range(hidden2):
        value = _value(b2, row)
        offset = row * hidden1
        for col in range(hidden1):
            value += _value(w2, offset + col) * h1[col]
        h2[row] = value if value > 0.0 else 0.0
    for row in range(classes):
        value = _value(b3, row)
        offset = row * hidden2
        for col in range(hidden2):
            value += _value(w3, offset + col) * h2[col]
        logits[row] = value
    result = {"logits": [float(value) for value in logits], "predicted_index": max(range(classes), key=lambda i: logits[i])}
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    infer()
