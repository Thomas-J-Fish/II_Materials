# hardware/sensors.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .motor import MotorController


@dataclass
class ForceCal:
    scale_N_per_count: float = 1.0
    offset_counts: float = 0.0


class Sensors:
    """
    Displacement comes from motor's unwrapped position.
    "Force" comes from motor's last_load_raw (torque proxy), scaled.
    """

    def __init__(self, motor: MotorController, simulation: bool = True, cal: Optional[ForceCal] = None) -> None:
        self.motor = motor
        self.simulation = simulation
        self.cal = cal or ForceCal()

    def close(self) -> None:
        pass

    def tare(self, samples: int = 25) -> None:
        if self.simulation:
            self.cal.offset_counts = 0.0
            return

        vals = []
        for _ in range(max(1, samples)):
            ok = self.motor.update_feedback()
            if ok:
                vals.append(float(self.motor.last_load_raw))
        self.cal.offset_counts = (sum(vals) / len(vals)) if vals else 0.0

    def read_displacement_m(self) -> float:
        # MotorController maintains unwrapped angle internally
        return float(self.motor.get_displacement_m())

    def read_force_N(self) -> float:
        if self.simulation:
            return 0.0

        # Ensure feedback is reasonably fresh
        self.motor.update_feedback()
        raw = float(self.motor.last_load_raw)

        if raw != raw:  # NaN
            return float("nan")

        net = raw - float(self.cal.offset_counts)
        return float(net * self.cal.scale_N_per_count)