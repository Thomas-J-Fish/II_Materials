# hardware/thermocouple.py

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional

import spidev


@dataclass
class ThermocoupleConfig:
    spi_bus: int = 0
    spi_dev: int = 0          # CE0 = 0, CE1 = 1
    channel: int = 0          # MCP3008 CH0..CH7
    vref: float = 3.3         # MUST match MCP3008 VREF wiring (3.3 or 5.0)
    max_speed_hz: int = 1350000
    avg_samples: int = 10     # basic smoothing


class ThermocoupleAD8495:
    """
    Reads AD8495 amplifier output via MCP3008 and converts to degC.

    AD8495 transfer:
      Vout = 1.25 V + 0.005 V/°C * T
      => T = (Vout - 1.25) / 0.005
    """

    def __init__(self, cfg: Optional[ThermocoupleConfig] = None) -> None:
        self.cfg = cfg or ThermocoupleConfig()
        if not (0 <= self.cfg.channel <= 7):
            raise ValueError("MCP3008 channel must be 0..7")

        self.spi = spidev.SpiDev()
        self.spi.open(self.cfg.spi_bus, self.cfg.spi_dev)
        self.spi.max_speed_hz = self.cfg.max_speed_hz

    def close(self) -> None:
        try:
            self.spi.close()
        except Exception:
            pass

    def _read_adc_once(self) -> int:
        ch = self.cfg.channel
        r = self.spi.xfer2([1, (8 + ch) << 4, 0])
        return ((r[1] & 3) << 8) + r[2]

    def read_voltage(self) -> float:
        n = max(1, int(self.cfg.avg_samples))
        s = 0
        for _ in range(n):
            s += self._read_adc_once()
        adc = s / n
        return float(adc) * float(self.cfg.vref) / 1023.0

    def read_temp_c(self) -> float:
        v = self.read_voltage()
        return (v - 1.25) / 0.005