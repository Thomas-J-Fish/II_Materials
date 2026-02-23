# hardware/motor.py
from __future__ import annotations

import time
import math
from dataclasses import dataclass
from typing import Optional

import serial


@dataclass
class ServoConfig:
    device_name: str = "/dev/ttyUSB0"
    baudrate: int = 115200
    servo_id: int = 1

    # Geometry
    radius_m: float = 0.02  # drum/pulley radius [m]

    # Position model: assume 12-bit position (0..4095) per revolution
    units_per_rev: int = 4096
    pos_mask: int = 0x0FFF  # keep low 12 bits

    # Registers (SCSCL-like)
    TORQUE_ENABLE: int = 40        # 0x28
    GOAL_POSITION_L: int = 0x2A    # 42
    PRESENT_POSITION_L: int = 0x38 # 56
    PRESENT_LOAD_L: int = 0x3C     # 60

    # Motion settings
    move_time_ms: int = 1000
    speed: int = 800

    # Polling
    poll_s: float = 0.02


class MotorController:
    """
    SIMPLE SCServo controller using raw packets over pyserial.

    Exposes the API your GUI already expects:
      - step_deg(delta)
      - get_displacement_m()
      - go_home()
      - close()

    Plus extra useful reads:
      - read_pos_units() (0..4095 masked)
      - read_load_raw()  (0..65535 raw "load/torque proxy")
    """

    INST_READ = 0x02
    INST_WRITE = 0x03

    def __init__(self, simulation: bool = True, config: Optional[ServoConfig] = None):
        self.simulation = simulation
        self.config = config or ServoConfig()

        self._ser: Optional[serial.Serial] = None

        # State tracking
        self._pos_units_sim: int = 0
        self._pos_units_last: Optional[int] = None
        self._turns: int = 0
        self.total_pos_units: int = 0
        self.total_angle_deg: float = 0.0

        if not self.simulation:
            self._ser = serial.Serial(self.config.device_name, self.config.baudrate, timeout=0.05)
            time.sleep(0.15)
            # Ensure torque enabled so it can move + report load
            self.torque_enable(True)

            # Seed position tracking
            p = self.read_pos_units()
            if p is not None:
                self._seed_unwrap(p)

    # ---------------- Packet helpers ----------------
    @staticmethod
    def _checksum(sid: int, length: int, inst_or_err: int, params: list[int]) -> int:
        return (~(sid + length + inst_or_err + sum(params))) & 0xFF

    @classmethod
    def _packet(cls, sid: int, inst: int, params: list[int]) -> bytes:
        length = 2 + len(params)
        chk = cls._checksum(sid, length, inst, params)
        return bytes([0xFF, 0xFF, sid, length, inst] + params + [chk])

    def _read_status_packet(self, timeout_s: float = 0.05):
        """
        Read one status packet:
          FF FF ID LEN ERR PARAMS... CHK
        Return (sid, err, params_bytes) or None.
        """
        if self._ser is None:
            return None

        deadline = time.time() + timeout_s

        # Find header
        while time.time() < deadline:
            b = self._ser.read(1)
            if not b or b[0] != 0xFF:
                continue
            b2 = self._ser.read(1)
            if not b2 or b2[0] != 0xFF:
                continue

            sid_b = self._ser.read(1)
            length_b = self._ser.read(1)
            err_b = self._ser.read(1)
            if len(sid_b) < 1 or len(length_b) < 1 or len(err_b) < 1:
                continue

            sid = sid_b[0]
            length = length_b[0]
            err = err_b[0]

            params_len = max(0, length - 2)
            params = self._ser.read(params_len) if params_len else b""
            chk_b = self._ser.read(1)
            if len(params) != params_len or len(chk_b) != 1:
                continue

            chk = chk_b[0]
            calc = self._checksum(sid, length, err, list(params))
            if chk != calc:
                continue

            return sid, err, params

        return None

    def _read_regs(self, start_addr: int, nbytes: int) -> Optional[bytes]:
        if self._ser is None:
            return None
        # Clear stale bytes to avoid old responses confusing us
        try:
            self._ser.reset_input_buffer()
        except Exception:
            pass

        self._ser.write(self._packet(self.config.servo_id, self.INST_READ, [start_addr, nbytes]))
        self._ser.flush()

        resp = self._read_status_packet(timeout_s=0.05)
        if resp is None:
            return None
        sid, err, params = resp
        if sid != self.config.servo_id or err != 0 or len(params) < nbytes:
            return None
        return params[:nbytes]

    def _write_bytes(self, params: list[int]) -> None:
        if self._ser is None:
            return
        self._ser.write(self._packet(self.config.servo_id, self.INST_WRITE, params))
        self._ser.flush()

    @staticmethod
    def _u16(lo: int, hi: int) -> int:
        return (lo & 0xFF) | ((hi & 0xFF) << 8)

    # ---------------- Torque / move ----------------
    def torque_enable(self, enable: bool) -> None:
        if self.simulation:
            return
        self._write_bytes([self.config.TORQUE_ENABLE, 1 if enable else 0])

    def move_to_units(self, pos_units: int) -> None:
        """
        Command a goal position (units are 0..4095 typically).
        """
        if self.simulation:
            self._pos_units_sim = int(pos_units) & self.config.pos_mask
            self._update_unwrapped(self._pos_units_sim)
            return

        pos_units = int(pos_units) & 0xFFFF
        params = [
            self.config.GOAL_POSITION_L,
            pos_units & 0xFF, (pos_units >> 8) & 0xFF,
            self.config.move_time_ms & 0xFF, (self.config.move_time_ms >> 8) & 0xFF,
            self.config.speed & 0xFF, (self.config.speed >> 8) & 0xFF,
        ]
        self._write_bytes(params)

    def stop(self) -> None:
        """
        Your proven "stop" method is torque disable.
        """
        self.torque_enable(False)

    # ---------------- Reads ----------------
    def read_pos_units(self) -> Optional[int]:
        """
        Read present position and mask to low 12 bits (0..4095).
        """
        if self.simulation:
            return self._pos_units_sim & self.config.pos_mask

        data = self._read_regs(self.config.PRESENT_POSITION_L, 2)
        if data is None or len(data) < 2:
            return None
        raw = self._u16(data[0], data[1])
        return raw & self.config.pos_mask

    def read_load_raw(self) -> Optional[int]:
        """
        Read present load (torque proxy). This is usually NOT calibrated.
        """
        if self.simulation:
            return 0

        data = self._read_regs(self.config.PRESENT_LOAD_L, 2)
        if data is None or len(data) < 2:
            return None
        return self._u16(data[0], data[1])

    # ---------------- Unwrap + displacement ----------------
    def _seed_unwrap(self, pos_units: int) -> None:
        self._pos_units_last = pos_units
        self._turns = 0
        self._update_unwrapped(pos_units)

    def _update_unwrapped(self, pos_units: int) -> None:
        """
        Unwrap modulo position into multi-turn units (very simple).
        """
        if self._pos_units_last is None:
            self._pos_units_last = pos_units
            self._turns = 0
        else:
            delta = pos_units - self._pos_units_last
            half = self.config.units_per_rev // 2
            if delta > half:
                self._turns -= 1
            elif delta < -half:
                self._turns += 1
            self._pos_units_last = pos_units

        self.total_pos_units = self._turns * self.config.units_per_rev + pos_units
        self.total_angle_deg = (self.total_pos_units / self.config.units_per_rev) * 360.0

    def get_displacement_m(self) -> float:
        """
        Displacement = r * theta_total.
        """
        theta_rad = math.radians(self.total_angle_deg)
        return float(self.config.radius_m * theta_rad)

    # ---------------- API used by your acquisition ----------------
    def step_deg(self, delta_deg: float, step_time_s: float = 0.0) -> None:
        """
        VERY simple: read current pos, command new goal, then sleep a bit.
        No feedback stop logic (you asked to keep it simple).
        """
        if self.simulation:
            self.total_angle_deg += float(delta_deg)
            # update sim pos
            u = int((self.total_angle_deg / 360.0) * self.config.units_per_rev) & self.config.pos_mask
            self._pos_units_sim = u
            self._update_unwrapped(u)
            return

        # Ensure torque on while moving
        self.torque_enable(True)

        p0 = self.read_pos_units()
        if p0 is None:
            # If read fails, just command relative using last known / 0
            p0 = 0

        # Update unwrap from this reading
        if self._pos_units_last is None:
            self._seed_unwrap(p0)
        else:
            self._update_unwrapped(p0)

        delta_units = int(round((float(delta_deg) / 360.0) * self.config.units_per_rev))
        target_units = (p0 + delta_units) & self.config.pos_mask

        self.move_to_units(target_units)

        # crude wait for motion
        time.sleep(max(0.05, self.config.move_time_ms / 1000.0))

        # optionally additional dwell
        if step_time_s > 0:
            time.sleep(step_time_s)

        # Update unwrap after move
        p1 = self.read_pos_units()
        if p1 is not None:
            self._update_unwrapped(p1)

    def go_home(self) -> None:
        """
        Set current position as zero reference.
        """
        if self.simulation:
            self.total_angle_deg = 0.0
            self._pos_units_sim = 0
            self._seed_unwrap(0)
            return

        p = self.read_pos_units()
        if p is None:
            p = 0
        self._seed_unwrap(p)
        self.total_pos_units = 0
        self.total_angle_deg = 0.0

    def close(self) -> None:
        if not self.simulation:
            try:
                self.stop()
            except Exception:
                pass
            try:
                if self._ser is not None:
                    self._ser.close()
            except Exception:
                pass
            self._ser = None