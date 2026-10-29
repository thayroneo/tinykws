"""Capture ADC2 with the RP2040 ADC FIFO and DMA, without Python timing."""
from array import array
import json
import time
import uctypes
from machine import ADC, mem32


ADC_BASE = 0x4004C000
ADC_CS = ADC_BASE + 0x00
ADC_FCS = ADC_BASE + 0x08
ADC_FIFO = ADC_BASE + 0x0C
ADC_DIV = ADC_BASE + 0x10

DMA_BASE = 0x50000000
DMA_CHANNEL = 0
DMA_CH = DMA_BASE + DMA_CHANNEL * 0x40
DMA_READ_ADDR = DMA_CH + 0x00
DMA_WRITE_ADDR = DMA_CH + 0x04
DMA_TRANS_COUNT = DMA_CH + 0x08
DMA_CTRL_TRIG = DMA_CH + 0x0C

ADC_CS_EN = 1 << 0
ADC_CS_START_MANY = 1 << 3
ADC_CS_AINSEL_MASK = 0x7 << 12
ADC_FCS_EN = 1 << 0
ADC_FCS_DREQ_EN = 1 << 3
ADC_FCS_OVER = 1 << 11
ADC_FCS_LEVEL_MASK = 0xF << 16
DMA_CTRL_EN = 1 << 0
DMA_CTRL_DATA_SIZE_HALFWORD = 1 << 2
DMA_CTRL_INCR_WRITE = 1 << 5
DMA_CTRL_BUSY = 1 << 24
DMA_CTRL_ERROR_MASK = 0xE0000000
DMA_TREQ_ADC = 36 << 15


def _drain_fifo():
    while (mem32[ADC_FCS] & ADC_FCS_LEVEL_MASK) != 0:
        _ = mem32[ADC_FIFO]


def capture_once(rate_hz=16000, duration_ms=1000, path="/capture_dma"):
    sample_count = (rate_hz * duration_ms) // 1000
    if sample_count <= 0 or sample_count > 16000:
        raise ValueError("sample count must be in 1..16000")

    # Constructing ADC(2) initializes the ADC block and GPIO28/ADC2.
    _adc = ADC(2)
    samples = array("H")
    for _ in range(sample_count):
        samples.append(0)

    _drain_fifo()
    mem32[ADC_FCS] = 0
    mem32[ADC_CS] = (mem32[ADC_CS] & ~ADC_CS_AINSEL_MASK) | (2 << 12) | ADC_CS_EN
    divider = (48000000.0 / rate_hz) - 1.0
    mem32[ADC_DIV] = int(divider * 256.0 + 0.5)
    mem32[ADC_FCS] = ADC_FCS_EN | ADC_FCS_DREQ_EN | (1 << 24)

    buffer_address = uctypes.addressof(samples)
    mem32[DMA_READ_ADDR] = ADC_FIFO
    mem32[DMA_WRITE_ADDR] = buffer_address
    mem32[DMA_TRANS_COUNT] = sample_count
    control = (
        DMA_CTRL_EN
        | DMA_CTRL_DATA_SIZE_HALFWORD
        | DMA_CTRL_INCR_WRITE
        | DMA_TREQ_ADC
    )

    start_us = time.ticks_us()
    mem32[DMA_CTRL_TRIG] = control
    mem32[ADC_CS] |= ADC_CS_START_MANY
    while mem32[DMA_CTRL_TRIG] & DMA_CTRL_BUSY:
        time.sleep_us(20)
    end_us = time.ticks_us()
    mem32[ADC_CS] &= ~ADC_CS_START_MANY
    mem32[ADC_FCS] = 0
    fifo_status = mem32[ADC_FCS]
    _drain_fifo()

    elapsed_us = time.ticks_diff(end_us, start_us)
    effective_rate = ((sample_count - 1) * 1000000.0 / elapsed_us) if elapsed_us else 0.0
    metadata = {
        "source": "BitDogLab RP2040 ADC FIFO/DMA via MicroPython registers",
        "sample_count": sample_count,
        "requested_rate_hz": rate_hz,
        "duration_ms": duration_ms,
        "adc_gpio": 28,
        "adc_channel": 2,
        "adc_bits": 12,
        "dma_channel": DMA_CHANNEL,
        "elapsed_us": elapsed_us,
        "effective_rate_hz": effective_rate,
        "adc_fifo_overflow": bool(fifo_status & ADC_FCS_OVER),
        "dma_error": bool(mem32[DMA_CTRL_TRIG] & DMA_CTRL_ERROR_MASK),
        "sample_format": "uint16_le_adc12",
    }
    raw_path = path + ".raw"
    json_path = path + ".json"
    with open(raw_path, "wb") as stream:
        stream.write(samples)
    with open(json_path, "w") as stream:
        stream.write(json.dumps(metadata))
    print(json.dumps(metadata))
    return metadata


if __name__ == "__main__":
    capture_once()
