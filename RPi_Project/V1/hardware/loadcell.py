# hardware/loadcell.py
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Optional, List

from .hx711 import HX711, HX711Config
from .loadcell_calibration import load_latest_calibration


@dataclass
class LoadCellConfig:
    cal_dir: str = "calibration"
    median_samples: int = 25
    sample_delay_s: float = 0.003
    g: float = 9.80665


class LoadCell:
    """
    Loads the latest calibration JSON and returns NET mass (g) relative to the
    holder baseline (i.e. holder ≈ 0 g after calibration/tare).

    You can additionally tare at runtime (e.g. at start of each run).
    """

    def __init__(self, simulation: bool = False, cfg: Optional[LoadCellConfig] = None):
        self.simulation = simulation
        self.cfg = cfg or LoadCellConfig()

        self.offset_counts: float = 0.0
        self.scale_counts_per_g: float = 1.0
        self.calibration_path: Optional[str] = None

        self._hx: Optional[HX711] = None
        self._t0 = time.monotonic()

        if not self.simulation:
            cal = load_latest_calibration(self.cfg.cal_dir)
            self.offset_counts = float(cal["offset_counts"])
            self.scale_counts_per_g = float(cal["scale_counts_per_g"])
            self.calibration_path = cal.get("_path")

            dout = int(cal["dout_pin"])
            sck = int(cal["sck_pin"])
            self._hx = HX711(HX711Config(dout_pin=dout, sck_pin=sck, gain=128))

    def close(self) -> None:
        try:
            if self._hx is not None:
                self._hx.close()
        finally:
            self._hx = None

    def _read_raw_median(self) -> Optional[int]:
        if self.simulation:
            # gentle simulated signal
            import math
            t = time.monotonic() - self._t0
            m = 50.0 + 50.0 * (0.5 * (1.0 + math.sin(2 * math.pi * t / 4.0)))
            return int(self.offset_counts + m * self.scale_counts_per_g)

        if self._hx is None:
            return None

        vals: List[int] = []
        for _ in range(max(5, self.cfg.median_samples)):
            try:
                vals.append(int(self._hx.read_raw()))
            except Exception:
                pass
            time.sleep(self.cfg.sample_delay_s)

        if len(vals) < 5:
            return None

        return int(statistics.median(vals))

    def read_mass_g(self) -> Optional[float]:
        raw = self._read_raw_median()
        if raw is None:
            return None
        net = raw - self.offset_counts
        return net / self.scale_counts_per_g

    def read_force_n(self) -> Optional[float]:
        m = self.read_mass_g()
        if m is None:
            return None
        return (m / 1000.0) * self.cfg.g

    def tare_to_current(self) -> bool:
        """
        Set offset to the current raw reading.
        Use: holder installed, no extra weights.
        """
        raw = self._read_raw_median()
        if raw is None:
            return False
        self.offset_counts = float(raw)
        return True