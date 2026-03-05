# V1/hardware/hx711.py
from __future__ import annotations

import time
from dataclasses import dataclass
import RPi.GPIO as GPIO


@dataclass
class HX711Config:
    dout_pin: int = 5
    sck_pin: int = 6
    gain: int = 128
    read_timeout_s: float = 0.2
    settle_s: float = 0.05
    retries: int = 12  # more retries to ride through occasional bad frames


class HX711:
    """
    HX711 driver with explicit rejection of known-invalid frames:
      - 0xFFFFFF -> signed -1
      - 0x7FFFFF -> signed +8388607
    """

    def __init__(self, cfg: HX711Config):
        self.cfg = cfg

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.cfg.sck_pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.cfg.dout_pin, GPIO.IN)

        time.sleep(self.cfg.settle_s)
        _ = self.read_raw()  # prime

    def close(self) -> None:
        try:
            GPIO.cleanup((self.cfg.sck_pin, self.cfg.dout_pin))
        except Exception:
            pass

    def is_ready(self) -> bool:
        return GPIO.input(self.cfg.dout_pin) == 0

    def _gain_pulses(self) -> int:
        return {128: 1, 32: 2, 64: 3}.get(self.cfg.gain) or 1

    @staticmethod
    def _to_signed_24(u24: int) -> int:
        u24 &= 0xFFFFFF
        if u24 & 0x800000:
            u24 -= 1 << 24
        return int(u24)

    def _read_u24_once(self) -> int:
        # wait ready
        t0 = time.monotonic()
        while not self.is_ready():
            if time.monotonic() - t0 > self.cfg.read_timeout_s:
                raise TimeoutError("HX711 not ready (check wiring/power)")
            time.sleep(0.001)

        value = 0
        for _ in range(24):
            GPIO.output(self.cfg.sck_pin, GPIO.HIGH)
            time.sleep(1e-6)
            value = (value << 1) | (GPIO.input(self.cfg.dout_pin) & 1)
            GPIO.output(self.cfg.sck_pin, GPIO.LOW)
            time.sleep(1e-6)

        # set gain/channel for next conversion
        for _ in range(self._gain_pulses()):
            GPIO.output(self.cfg.sck_pin, GPIO.HIGH)
            time.sleep(1e-6)
            GPIO.output(self.cfg.sck_pin, GPIO.LOW)
            time.sleep(1e-6)

        return int(value) & 0xFFFFFF

    def read_raw(self) -> int:
        last_exc: Exception | None = None

        for _ in range(max(1, self.cfg.retries)):
            try:
                u24 = self._read_u24_once()
                signed = self._to_signed_24(u24)

                # Reject exact edge patterns (these are what you printed)
                if signed == -1 or signed == 8388607:
                    continue

                return signed
            except Exception as e:
                last_exc = e
                continue

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("HX711 read failed (too many invalid frames)")