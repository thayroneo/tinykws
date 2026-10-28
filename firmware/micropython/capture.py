"""Captura audio do microfone analogico da BitDogLab no ADC GPIO28."""
from array import array
import gc
import json
import math
import time

import machine


ADC_PIN = 28
DEFAULT_RATE_HZ = 16000
DEFAULT_DURATION_MS = 1000


def _wait_until(deadline_us):
    """Espera ate o prazo sem alocar estruturas durante a captura."""
    while True:
        now_us = time.ticks_us()
        remaining_us = time.ticks_diff(deadline_us, now_us)
        if remaining_us <= 0:
            return now_us
        if remaining_us > 100:
            time.sleep_us(remaining_us - 20)
        else:
            time.sleep_us(1)


def capture(
    raw_path="capture.raw",
    metadata_path="capture.json",
    rate_hz=DEFAULT_RATE_HZ,
    duration_ms=DEFAULT_DURATION_MS,
):
    """Captura amostras uint16 little-endian e grava metadados de temporizacao."""
    sample_count = (rate_hz * duration_ms) // 1000
    samples = array("H")
    for _ in range(sample_count):
        samples.append(0)
    adc = machine.ADC(ADC_PIN)
    gc.collect()

    start_us = time.ticks_us()
    previous_us = None
    first_sample_us = None
    last_sample_us = None
    interval_count = 0
    interval_sum = 0
    interval_sum_sq = 0
    interval_min = 1 << 30
    interval_max = 0
    late_count = 0
    max_lateness_us = 0
    sample_sum = 0
    sample_min = 65535
    sample_max = 0

    for index in range(sample_count):
        target_us = time.ticks_add(start_us, (index * 1000000) // rate_hz)
        actual_us = _wait_until(target_us)
        value = adc.read_u16()
        samples[index] = value
        sample_sum += value
        if value < sample_min:
            sample_min = value
        if value > sample_max:
            sample_max = value
        if first_sample_us is None:
            first_sample_us = actual_us
        if previous_us is not None:
            interval_us = time.ticks_diff(actual_us, previous_us)
            interval_count += 1
            interval_sum += interval_us
            interval_sum_sq += interval_us * interval_us
            if interval_us < interval_min:
                interval_min = interval_us
            if interval_us > interval_max:
                interval_max = interval_us
        lateness_us = time.ticks_diff(actual_us, target_us)
        if lateness_us > 1:
            late_count += 1
        if lateness_us > max_lateness_us:
            max_lateness_us = lateness_us
        previous_us = actual_us
        last_sample_us = actual_us

    with open(raw_path, "wb") as stream:
        stream.write(memoryview(samples))

    elapsed_us = time.ticks_diff(last_sample_us, first_sample_us)
    mean_interval_us = interval_sum / interval_count if interval_count else 0.0
    variance = 0.0
    if interval_count:
        variance = (interval_sum_sq / interval_count) - (mean_interval_us * mean_interval_us)
    metadata = {
        "adc_pin": ADC_PIN,
        "adc_channel": 2,
        "sample_count": sample_count,
        "requested_rate_hz": rate_hz,
        "duration_ms": duration_ms,
        "sample_format": "uint16_le",
        "adc_range": [0, 65535],
        "effective_rate_hz": ((sample_count - 1) * 1000000 / elapsed_us) if elapsed_us else 0.0,
        "elapsed_us": elapsed_us,
        "interval_mean_us": mean_interval_us,
        "interval_jitter_rms_us": math.sqrt(max(0.0, variance)),
        "interval_min_us": interval_min if interval_count else 0,
        "interval_max_us": interval_max if interval_count else 0,
        "late_samples_gt_1us": late_count,
        "max_lateness_us": max_lateness_us,
        "sample_min": sample_min,
        "sample_max": sample_max,
        "sample_mean": sample_sum / sample_count,
        "firmware": "micropython",
    }
    with open(metadata_path, "w") as stream:
        stream.write(json.dumps(metadata))
    return metadata
