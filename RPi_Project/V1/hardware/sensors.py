# hardware/sensors.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .motor import MotorController


@dataclass
class ForceCal:
    # Servo load is a proxy (unitless). You can tune this later.
    # Start with 1.0 to just plot raw-ish values in "N".
    scale_N_per_count: float = 1.0
    offset_counts: float = 0.0


class Sensors:
    """
    Reads displacement (from motor position) + "force" from servo load register.

    This keeps your GUI + acquisition pipeline unchanged:
      - read_displacement_m()
      - read_force_N()  (really "load proxy", scaled)
    """

    def __init__(
        self,
        motor: MotorController,
        simulation: bool = True,
        cal: Optional[ForceCal] = None,
    ) -> None:
        self.motor = motor
        self.simulation = simulation
        self.cal = cal or ForceCal()

    def close(self) -> None:
        pass

    def tare(self, samples: int = 25) -> None:
        """
        Set offset based on current load (no external load).
        """
        if self.simulation:
            self.cal.offset_counts = 0.0
            return

        vals = []
        for _ in range(max(1, samples)):
            v = self.motor.read_load_raw()
            if v is not None:
                vals.append(float(v))
        self.cal.offset_counts = (sum(vals) / len(vals)) if vals else 0.0

    def read_displacement_m(self) -> float:
        # Update motor unwrap using latest position so displacement is live
        if not self.simulation:
            p = self.motor.read_pos_units()
            if p is not None:
                # update internal unwrap
                # (motor.step_deg already updates too, but this keeps it continuous)
                self.motor._update_unwrapped(p)  # simple + practical for now
        return float(self.motor.get_displacement_m())

    def read_force_N(self) -> float:
        if self.simulation:
            return 0.0

        raw = self.motor.read_load_raw()
        if raw is None:
            return float("nan")

        net = float(raw) - float(self.cal.offset_counts)
        return float(net * self.cal.scale_N_per_count)