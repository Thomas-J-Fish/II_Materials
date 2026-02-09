# hardware/sensors.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from .motor import MotorController

if TYPE_CHECKING:
    from .hx711 import HX711, HX711Config
else:
    HX711 = object          # type: ignore
    HX711Config = object    # type: ignore


@dataclass
class ForceCal:
    tare_offset_counts: float = 0.0
    scale_N_per_count: float = 1.0  # you will calibrate this!


class Sensors:
    """
    Reads displacement (from motor model) + force (from HX711 or simulation).
    """

    def __init__(
        self,
        motor: MotorController,
        simulation: bool = True,
        spring_k_N_per_m: float = 100.0,
        hx711_cfg: Optional[HX711Config] = None,
        cal: Optional[ForceCal] = None,
        avg_samples: int = 10,
    ) -> None:
        self.motor = motor
        self.simulation = simulation
        self.spring_k = spring_k_N_per_m

        self.avg_samples = avg_samples
        self.cal = cal or ForceCal()

        self.hx = None
        if not self.simulation:
            # Import only when needed (so laptops don't require RPi.GPIO)
            try:
                from .hx711 import HX711, HX711Config  # type: ignore
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "HX711 requested (simulation=False) but hx711/RPi.GPIO not available on this system."
                ) from e

            self.hx = HX711(hx711_cfg or HX711Config(dout_pin=6, sck_pin=5, gain=128))

    def close(self) -> None:
        if self.hx is not None:
            self.hx.close()

    def tare(self, samples: int = 25) -> None:
        """
        Set tare offset based on current (no-load) reading.
        Call this with nothing on the load cell.
        """
        if self.simulation:
            self.cal.tare_offset_counts = 0.0
            return
        assert self.hx is not None
        self.cal.tare_offset_counts = float(self.hx.read_average(samples=samples))

    def read_displacement_m(self) -> float:
        # Displacement from the motor model/feedback
        return float(self.motor.get_displacement_m())

    def read_force_N(self) -> float:
        if self.simulation:
            x = self.read_displacement_m()
            return float(self.spring_k * x)

        assert self.hx is not None
        raw = float(self.hx.read_average(samples=self.avg_samples))
        net = raw - self.cal.tare_offset_counts
        return float(net * self.cal.scale_N_per_count)