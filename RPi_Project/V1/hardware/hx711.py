# hardware/hx711.py
from __future__ import annotations

import time
from dataclasses import dataclass

import RPi.GPIO as GPIO


@dataclass
class HX711Config:
    dout_pin: int = 5          # DATA -> GPIO5 (BCM)
    sck_pin: int = 6           # SCK  -> GPIO6 (BCM)
    gain: int = 128            # 128 for channel A
    read_timeout_s: float = 0.2
    settle_s: float = 0.05


class HX711:
    """
    Minimal HX711 driver using RPi.GPIO (BCM numbering).
    Reads 24-bit two's complement values.
    """

    def __init__(self, cfg: HX711Config):
        self.cfg = cfg

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.cfg.sck_pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.cfg.dout_pin, GPIO.IN)

        time.sleep(self.cfg.settle_s)

        # Prime gain/channel
        _ = self.read_raw()

    def close(self) -> None:
        try:
            GPIO.cleanup((self.cfg.sck_pin, self.cfg.dout_pin))
        except Exception:
            pass

    def is_ready(self) -> bool:
        return GPIO.input(self.cfg.dout_pin) == 0

    def _gain_pulses(self) -> int:
        if self.cfg.gain == 128:
            return 1
        if self.cfg.gain == 32:
            return 2
        if self.cfg.gain == 64:
            return 3
        raise ValueError("gain must be one of {128, 64, 32}")

    def read_raw(self) -> int:
        t0 = time.monotonic()
        while not self.is_ready():
            if time.monotonic() - t0 > self.cfg.read_timeout_s:
                raise TimeoutError("HX711 not ready (check wiring/power)")
            time.sleep(0.001)

        value = 0
        for _ in range(24):
            GPIO.output(self.cfg.sck_pin, GPIO.HIGH)
            time.sleep(0.000001)
            value = (value << 1) | GPIO.input(self.cfg.dout_pin)
            GPIO.output(self.cfg.sck_pin, GPIO.LOW)
            time.sleep(0.000001)

        for _ in range(self._gain_pulses()):
            GPIO.output(self.cfg.sck_pin, GPIO.HIGH)
            time.sleep(0.000001)
            GPIO.output(self.cfg.sck_pin, GPIO.LOW)
            time.sleep(0.000001)

        if value & 0x800000:
            value -= 1 << 24

        return int(value)

    def read_average(self, samples: int = 10, delay_s: float = 0.01) -> float:
        vals = []
        for _ in range(max(1, samples)):
            vals.append(self.read_raw())
            if delay_s > 0:
                time.sleep(delay_s)
        return sum(vals) / len(vals)