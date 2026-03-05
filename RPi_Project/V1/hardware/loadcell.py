# V1/hardware/loadcell.py
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Deque
from collections import deque

from .hx711 import HX711, HX711Config
from .loadcell_calibration import load_latest_calibration


def _default_cal_dir() -> str:
    here = Path(__file__).resolve()
    v1_dir = here.parents[1]  # .../V1
    return str(v1_dir / "calibration")


@dataclass
class LoadCellConfig:
    cal_dir: str = ""
    median_samples: int = 25
    sample_delay_s: float = 0.003
    g: float = 9.80665

    # Simple robustness controls (prevents insane spikes)
    max_raw_step: int = 2_000_000     # reject raw jumps bigger than this (counts)
    history_len: int = 15             # recent raw history for jump comparison
    max_bad_in_row: int = 20          # after this, return None

    def __post_init__(self) -> None:
        if not self.cal_dir:
            self.cal_dir = _default_cal_dir()


class LoadCell:
    """
    Calibration-aware HX711 reader.
    Returns NET mass (g) relative to offset (tare/calibration baseline).
    Hardened against occasional corrupted raw reads by rejecting large jumps.
    """

    def __init__(self, simulation: bool = False, cfg: Optional[LoadCellConfig] = None):
        self.simulation = simulation
        self.cfg = cfg or LoadCellConfig()

        self.offset_counts: float = 0.0
        self.scale_counts_per_g: float = 1.0
        self.calibration_path: Optional[str] = None

        self._hx: Optional[HX711] = None
        self._t0 = time.monotonic()

        self._raw_hist: Deque[int] = deque(maxlen=self.cfg.history_len)
        self._bad_in_row = 0

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

    def _sim_raw(self) -> int:
        import math
        t = time.monotonic() - self._t0
        m = 50.0 + 50.0 * (0.5 * (1.0 + math.sin(2 * math.pi * t / 4.0)))
        return int(self.offset_counts + m * self.scale_counts_per_g)

    def _read_raw_once(self) -> Optional[int]:
        if self.simulation:
            return self._sim_raw()
        if self._hx is None:
            return None
        try:
            return int(self._hx.read_raw())
        except Exception:
            return None

    def _raw_is_plausible(self, raw: int) -> bool:
        """
        Reject raw values that jump wildly compared to recent history median.
        """
        if not self._raw_hist:
            return True
        med = int(statistics.median(self._raw_hist))
        return abs(raw - med) <= self.cfg.max_raw_step

    def _read_raw_median(self) -> Optional[int]:
        """
        Collect samples, reject implausible spikes, return median of good samples.
        """
        good: List[int] = []
        attempts = 0

        # try until we have enough good samples or too many failures
        while len(good) < max(5, self.cfg.median_samples) and attempts < (self.cfg.median_samples * 3):
            attempts += 1
            raw = self._read_raw_once()
            if raw is None:
                time.sleep(self.cfg.sample_delay_s)
                continue

            if self._raw_is_plausible(raw):
                good.append(raw)
                self._raw_hist.append(raw)
            else:
                # spike rejected, don't add to history
                pass

            time.sleep(self.cfg.sample_delay_s)

        if len(good) < 5:
            self._bad_in_row += 1
            if self._bad_in_row >= self.cfg.max_bad_in_row:
                return None
            return None

        self._bad_in_row = 0
        return int(statistics.median(good))

    def read_mass_g(self) -> Optional[float]:
        """
        Return NET mass in grams relative to the current offset_counts.
        Use absolute value of calibration scale so that a negative scale in the
        JSON (caused by a reversed sign during a prior calibration) won't
        flip the sign of reported mass.
        """
        raw = self._read_raw_median()
        if raw is None:
            return None
        net = raw - self.offset_counts

        # Use magnitude of calibration scale for conversion (avoid sign inversion)
        scale = abs(self.scale_counts_per_g) if self.scale_counts_per_g != 0 else 1.0
        return net / scale

    def read_force_n(self) -> Optional[float]:
        m = self.read_mass_g()
        if m is None:
            return None
        return (m / 1000.0) * self.cfg.g

    def tare_to_current(self) -> bool:
        """
        Set offset_counts to the current raw reading.
        After this call, read_mass_g() will return values relative to the
        current raw baseline (holder-only ≈ 0 g).
        """
        raw = self._read_raw_median()
        if raw is None:
            return False
        self.offset_counts = float(raw)
        return True