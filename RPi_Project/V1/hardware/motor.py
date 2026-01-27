# hardware/motor.py

from __future__ import annotations

import time
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ServoConfig:
    device_name: str = "/dev/ttyUSB0"      # Pi: "/dev/ttyUSB0" or "/dev/serial/by-id/..."
    baudrate: int = 1_000_000
    servo_id: int = 1

    # PWM control for multi-turn
    pwm_value: int = 500                   # magnitude; sign gives direction
    poll_s: float = 0.02                   # feedback polling interval
    move_timeout_s: float = 10.0           # safety timeout per step

    # Geometry
    radius_m: float = 0.02                 # drum/pulley radius [m]


class MotorController:
    """
    Multi-turn control using PWM mode + position feedback.

    Strategy:
      - Put servo into PWMMode.
      - Use WritePWM to rotate continuously.
      - Use ReadPosSpeed feedback to track position.
      - Unwrap modulo-4096 position to accumulate turns.
      - Implement step_deg(delta) by rotating until target delta is reached.

    simulation=True uses a simple software model.
    """

    def __init__(self, simulation: bool = True, config: Optional[ServoConfig] = None):
        self.simulation = simulation
        self.config = config or ServoConfig()

        # Feedback tracking (unwrapped)
        self._last_pos_units: Optional[int] = None
        self._turns: int = 0                # number of full wraps (+/-)
        self.total_pos_units: int = 0       # unwrapped in "units" (can exceed 0..4095)

        # Derived state for convenience
        self.total_angle_deg: float = 0.0

        # Real hardware handles/constants
        self._portHandler = None
        self._packetHandler = None
        self._COMM_SUCCESS = None

        if not self.simulation:
            self._init_real()

    # ---------------- Real hardware setup ----------------
    def _init_real(self) -> None:
        from scservo_sdk import PortHandler, scscl, COMM_SUCCESS  # type: ignore

        self._COMM_SUCCESS = COMM_SUCCESS
        self._portHandler = PortHandler(self.config.device_name)
        self._packetHandler = scscl(self._portHandler)

        if not self._portHandler.openPort():
            raise RuntimeError(f"Failed to open port: {self.config.device_name}")
        if not self._portHandler.setBaudRate(self.config.baudrate):
            raise RuntimeError(f"Failed to set baudrate: {self.config.baudrate}")

        # Put servo into PWM mode (from wheel.py)
        comm, err = self._packetHandler.PWMMode(self.config.servo_id)
        if comm != self._COMM_SUCCESS:
            raise RuntimeError(self._packetHandler.getTxRxResult(comm))
        if err != 0:
            raise RuntimeError(self._packetHandler.getRxPacketError(err))

        # Seed position tracking
        pos = self.read_pos_units()
        if pos is not None:
            self._last_pos_units = pos
            self._turns = 0
            self._update_unwrapped(pos)

    def close(self) -> None:
        if not self.simulation and self._portHandler is not None:
            self.stop()
            self._portHandler.closePort()

    # ---------------- Low-level servo I/O ----------------
    def set_pwm(self, pwm: int) -> None:
        """pwm sign controls direction; pwm=0 stops."""
        if self.simulation:
            return

        if self._packetHandler is None or self._COMM_SUCCESS is None:
            raise RuntimeError("Motor not initialised")

        comm, err = self._packetHandler.WritePWM(self.config.servo_id, int(pwm))
        if comm != self._COMM_SUCCESS:
            raise RuntimeError(self._packetHandler.getTxRxResult(comm))
        if err != 0:
            raise RuntimeError(self._packetHandler.getRxPacketError(err))

    def stop(self) -> None:
        self.set_pwm(0)

    def read_pos_units(self) -> Optional[int]:
        """
        Read current position in 0..4095 units via ReadPosSpeed.
        Assumes this works sensibly in PWMMode (your stated assumption).
        """
        if self.simulation:
            # simulate modulo position from total_angle_deg
            pos = int(round(((self.total_angle_deg % 360.0) / 360.0) * 4095.0))
            return max(0, min(4095, pos))

        if self._packetHandler is None or self._COMM_SUCCESS is None:
            return None

        if not hasattr(self._packetHandler, "ReadPosSpeed"):
            return None

        pos, speed, comm, err = self._packetHandler.ReadPosSpeed(self.config.servo_id)
        if comm != self._COMM_SUCCESS or err != 0:
            return None
        return int(pos)

    # ---------------- Unwrapping logic ----------------
    def _update_unwrapped(self, pos_units: int) -> None:
        """
        Convert modulo 0..4095 to unwrapped multi-turn units by detecting wrap.
        """
        if self._last_pos_units is None:
            self._last_pos_units = pos_units
            self._turns = 0
        else:
            delta = pos_units - self._last_pos_units

            # If it jumps a lot, assume wrap-around happened.
            # Threshold 2048 = half turn; robust against noise.
            if delta > 2048:
                # e.g. went from ~0 to ~4095 (wrapped backwards)
                self._turns -= 1
            elif delta < -2048:
                # e.g. went from ~4095 to ~0 (wrapped forwards)
                self._turns += 1

            self._last_pos_units = pos_units

        self.total_pos_units = self._turns * 4096 + pos_units
        self.total_angle_deg = (self.total_pos_units / 4095.0) * 360.0

    # ---------------- Public API expected by your code ----------------
    def step_deg(self, delta_deg: float, step_time_s: float = 0.0) -> None:
        """
        Rotate by approximately delta_deg (multi-turn), using feedback to stop at target.

        step_time_s is kept for compatibility; we don't rely on it for motion amount.
        """
        if self.simulation:
            # simulation: exact increment
            self.total_angle_deg += delta_deg
            return

        direction = 1 if delta_deg >= 0 else -1
        target_delta = abs(delta_deg)

        # Get starting angle from feedback
        start_pos = self.read_pos_units()
        if start_pos is None:
            raise RuntimeError("Failed to read position (required for multi-turn stepping).")
        self._update_unwrapped(start_pos)
        start_angle = self.total_angle_deg

        # Start moving
        self.set_pwm(direction * abs(self.config.pwm_value))

        t0 = time.time()
        try:
            while True:
                if time.time() - t0 > self.config.move_timeout_s:
                    raise TimeoutError("Servo move timed out (check PWM value, load, power).")

                pos = self.read_pos_units()
                if pos is None:
                    continue  # transient comms; keep trying

                self._update_unwrapped(pos)
                moved = abs(self.total_angle_deg - start_angle)

                if moved >= target_delta:
                    break

                time.sleep(self.config.poll_s)
        finally:
            self.stop()

        if step_time_s > 0:
            time.sleep(step_time_s)

    def get_displacement_m(self) -> float:
        """
        Displacement from total multi-turn rotation: s = r * theta_total.
        """
        theta_rad = math.radians(self.total_angle_deg)
        return self.config.radius_m * theta_rad

    def go_home(self) -> None:
        """
        For motor-mode systems you usually need a physical home switch or manual zeroing.
        For now: set current position as zero reference.
        """
        if self.simulation:
            self.total_angle_deg = 0.0
            return

        pos = self.read_pos_units()
        if pos is None:
            raise RuntimeError("Cannot home: failed to read position.")

        # Reset unwrapped tracking to treat current as zero
        self._last_pos_units = pos
        self._turns = 0
        self._update_unwrapped(pos)

        # Zero relative
        self.total_pos_units = 0
        self.total_angle_deg = 0.0