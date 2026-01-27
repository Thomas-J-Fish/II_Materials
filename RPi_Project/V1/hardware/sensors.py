# hardware/sensors.py

from __future__ import annotations

from typing import Optional

from .motor import MotorController


class Sensors:
    """
    Sensor abstraction.

    Default is simulation mode: force is computed as F = k * x where
    x is the motor's displacement. Replace TODO sections with real ADC /
    sensor reads.
    """

    def __init__(
        self,
        motor: MotorController,
        simulation: bool = True,
        spring_k_N_per_m: float = 100.0,
    ) -> None:
        """
        :param motor: MotorController instance used to get displacement.
        :param simulation: if True, use a simple linear spring model.
        :param spring_k_N_per_m: spring constant used in the simulation.
        """
        self.motor = motor
        self.simulation = simulation
        self.spring_k = spring_k_N_per_m

        if not self.simulation:
            # TODO: set up ADC / load cell amplifier (e.g. HX711) here.
            # For example:
            #   self.hx = HX711(dout_pin, sck_pin)
            #   self.hx.set_scale(calibration_factor)
            #   self.hx.tare()
            raise NotImplementedError(
                "Real sensor reading not implemented yet. "
                "Set simulation=True for software testing."
            )

    def read_displacement_m(self) -> float:
        """
        Return the current displacement [m].

        In this design we take displacement from the motor position.
        If you have a separate displacement sensor, change this.
        """
        return self.motor.get_displacement_m()

    def read_force_N(self) -> float:
        """
        Return the current force [N] on the spring.

        Simulation: F = k * x. Real: read ADC and convert using calibration.
        """
        if self.simulation:
            x = self.read_displacement_m()
            return self.spring_k * x
        else:
            # TODO: read from ADC and apply calibration.
            raise NotImplementedError("read_force_N for real hardware not implemented")