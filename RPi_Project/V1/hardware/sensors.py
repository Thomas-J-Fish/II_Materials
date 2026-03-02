# hardware/sensors.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .motor import MotorController
from .loadcell import LoadCell


@dataclass
class ForceCal:
    # Fallback calibration for motor proxy
    scale_N_per_count: float = 1.0
    offset_counts: float = 0.0


class Sensors:
    """
    Displacement comes from motor.
    Force comes from HX711 (preferred). Fallback to motor proxy if loadcell unavailable.
    """

    def __init__(self, motor: MotorController, simulation: bool = True, cal: Optional[ForceCal] = None) -> None:
        self.motor = motor
        self.simulation = simulation
        self.cal = cal or ForceCal()

        # Try to use load cell if possible
        self.loadcell: Optional[LoadCell] = None
        try:
            self.loadcell = LoadCell(simulation=simulation)
        except Exception:
            self.loadcell = None

    def close(self) -> None:
        try:
            if self.loadcell is not None:
                self.loadcell.close()
        except Exception:
            pass

    # ---- Load cell helpers ----
    def tare_loadcell(self) -> bool:
        if self.loadcell is None:
            return False
        return self.loadcell.tare_to_current()

    def read_mass_g(self) -> Optional[float]:
        if self.loadcell is None:
            return None
        return self.loadcell.read_mass_g()

    # ---- Existing API ----
    def read_displacement_m(self) -> float:
        return float(self.motor.get_displacement_m())

    def read_force_N(self) -> float:
        # Preferred: HX711
        if self.loadcell is not None:
            f = self.loadcell.read_force_n()
            if f is not None:
                return float(f)

        # Fallback: motor proxy
        if self.simulation:
            return 0.0

        self.motor.update_feedback()
        raw = float(self.motor.last_load_raw)
        if raw != raw:
            return float("nan")

        net = raw - float(self.cal.offset_counts)
        return float(net * self.cal.scale_N_per_count)