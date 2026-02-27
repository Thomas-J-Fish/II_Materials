from __future__ import annotations

import time
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import serial


@dataclass
class ServoConfig:
    device_name: str = "/dev/ttyUSB0"
    baudrate: int = 115200
    servo_id: int = 1

    # Geometry
    radius_m: float = 0.02  # drum/pulley radius [m]

    # ---- POSITION FORMAT GUESS ----
    # Most common: 12-bit (0..4095)
    units_per_rev: int = 1024
    pos_mask: int = 0x03FF
    # If your servo is 10-bit (0..1023), change to:
    # units_per_rev = 1024
    # pos_mask = 0x03FF
    # -------------------------------

    # Registers (SCSCL-like map; may vary by model)
    TORQUE_ENABLE: int = 0x28          # 40
    GOAL_POSITION_L: int = 0x2A        # 42
    PRESENT_POSITION_L: int = 0x38     # 56
    PRESENT_LOAD_L: int = 0x3C         # 60
    PRESENT_CURRENT_L: int = 0x45      # 69 (Hex 0x45) - Valid for STS/SC15

    # Motion settings for goal-position writes
    move_time_ms: int = 800
    speed: int = 800

    # Polling
    poll_s: float = 0.01  # faster sampling (100 Hz target), adjust if needed


class MotorController:
    """
    Simple SC/SCS serial servo interface using raw packets over pyserial.

    Key features:
      - Reads POSITION+LOAD+CURRENT in ONE read (faster + fewer desync issues)
      - Handles Signed (Two's Complement) conversion for Load/Current
      - Unwrap logic kept here; only updated when you call update_feedback()
    """

    INST_READ = 0x02
    INST_WRITE = 0x03

    def __init__(self, simulation: bool = True, config: Optional[ServoConfig] = None):
        self.simulation = simulation
        self.config = config or ServoConfig()

        self._ser: Optional[serial.Serial] = None

        # Unwrap tracking
        self._last_pos_units: Optional[int] = None
        self._turns: int = 0
        self.total_pos_units: int = 0
        self.total_angle_deg: float = 0.0

        # last load
        self.last_load_raw: float = float("nan")

        if not self.simulation:
            try:
                self._ser = serial.Serial(self.config.device_name, self.config.baudrate, timeout=0.05)
                time.sleep(0.15)
                self.torque_enable(True)

                # seed
                pos, load = self.read_pos_and_load()
                if pos is not None:
                    self._seed_unwrap(pos)
                if load is not None:
                    self.last_load_raw = float(load)
            except Exception as e:
                print(f"[Motor] Init failed: {e}")

    # ---------------- Packet helpers ----------------
    @staticmethod
    def _checksum(sid: int, length: int, inst_or_err: int, params: list[int]) -> int:
        return (~(sid + length + inst_or_err + sum(params))) & 0xFF

    @classmethod
    def _packet(cls, sid: int, inst: int, params: list[int]) -> bytes:
        length = 2 + len(params)
        chk = cls._checksum(sid, length, inst, params)
        return bytes([0xFF, 0xFF, sid, length, inst] + params + [chk])

    @staticmethod
    def _u16(lo: int, hi: int) -> int:
        return (lo & 0xFF) | ((hi & 0xFF) << 8)

    @staticmethod
    def _to_signed(val: int) -> int:
        """Convert 16-bit unsigned to signed integer (Two's Complement)."""
        if val > 32767:
            val -= 65536
        return val

    def _read_status_packet(self, timeout_s: float = 0.05):
        """
        Read one status packet: FF FF ID LEN ERR PARAMS... CHK
        Return (sid, err, params_bytes) or None.
        """
        if self._ser is None:
            return None

        deadline = time.time() + timeout_s

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
        """
        Send READ(start_addr, nbytes) and return param bytes.
        """
        if self._ser is None:
            return None

        # flush any old bytes to reduce desync
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

    # ---------------- Public control ----------------
    def torque_enable(self, enable: bool) -> None:
        if self.simulation:
            return
        self._write_bytes([self.config.TORQUE_ENABLE, 1 if enable else 0])

    def stop(self) -> None:
        self.torque_enable(False)

    def move_to_units(self, pos_units: int) -> None:
        """
        Write a goal position (pos units in servo scale).
        """
        if self.simulation:
            # sim: just set internal state
            pos_units = int(pos_units) & self.config.pos_mask
            self._update_unwrapped(pos_units)
            return

        pos_units = int(pos_units) & 0xFFFF
        params = [
            self.config.GOAL_POSITION_L,
            pos_units & 0xFF,
            (pos_units >> 8) & 0xFF,
            self.config.move_time_ms & 0xFF,
            (self.config.move_time_ms >> 8) & 0xFF,
            self.config.speed & 0xFF,
            (self.config.speed >> 8) & 0xFF,
        ]
        self._write_bytes(params)

    # ---------------- Read POSITION+LOAD+CURRENT (single packet) ----------------
    def read_pos_and_load(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Single read starting at PRESENT_POSITION_L.
        NOW UPDATED: Reads through to CURRENT_L/H to get better torque data.
        Returns: (Position [raw], Current [mA])
        """
        if self.simulation:
            # simple sim
            pos = (self.total_pos_units % self.config.units_per_rev) & self.config.pos_mask
            return pos, 0

        # Read from Position (56) up to Current (69+1=70)
        # Span: 70 - 56 + 1 = 15 bytes. We read 16 bytes for safety/alignment.
        start = self.config.PRESENT_POSITION_L
        end = self.config.PRESENT_CURRENT_L + 1
        nbytes = (end - start) + 1
        
        data = self._read_regs(start, nbytes)
        if data is None or len(data) < nbytes:
            return None, None

        # 1. Position (Offset 0)
        pos_raw16 = self._u16(data[0], data[1])
        pos = pos_raw16 & self.config.pos_mask

        # 2. Current (Offset = CurrentAddr - PosAddr)
        # Current is preferred over 'Load' for scientific plotting
        off_curr = self.config.PRESENT_CURRENT_L - self.config.PRESENT_POSITION_L
        
        # Check if we successfully read enough bytes to reach Current
        if len(data) >= (off_curr + 2):
            curr_raw = self._u16(data[off_curr], data[off_curr + 1])
            val_out = self._to_signed(curr_raw) # Current in mA
        else:
            # Fallback to Load if Current bytes missing
            off_load = self.config.PRESENT_LOAD_L - self.config.PRESENT_POSITION_L
            load_raw = self._u16(data[off_load], data[off_load + 1])
            val_out = self._to_signed(load_raw) # Unitless load

        return pos, val_out

    # ---------------- Feedback update / unwrapping ----------------
    def _seed_unwrap(self, pos_units: int) -> None:
        self._last_pos_units = int(pos_units)
        self._turns = 0
        self._update_unwrapped(int(pos_units))

    def _update_unwrapped(self, pos_units: int) -> None:
        pos_units = int(pos_units) & self.config.pos_mask

        if self._last_pos_units is None:
            self._last_pos_units = pos_units
            self._turns = 0
        else:
            delta = pos_units - self._last_pos_units
            half = self.config.units_per_rev // 2

            if delta > half:
                self._turns -= 1
            elif delta < -half:
                self._turns += 1

            self._last_pos_units = pos_units

        self.total_pos_units = self._turns * self.config.units_per_rev + pos_units
        self.total_angle_deg = (self.total_pos_units / self.config.units_per_rev) * 360.0

    def update_feedback(self) -> bool:
        """
        Read pos+load once, update internal unwrap and last_load_raw.
        Returns True if a valid read occurred.
        """
        pos, load = self.read_pos_and_load()
        if pos is None or load is None:
            return False

        self._update_unwrapped(pos)
        self.last_load_raw = float(load)
        return True

    # ---------------- API expected by your stack ----------------
    def step_deg(self, delta_deg: float, step_time_s: float = 0.0) -> None:
        """
        Very simple: read once, command a new goal, sleep for move_time_ms.
        """
        if self.simulation:
            self.total_angle_deg += float(delta_deg)
            # update unwrapped using modulo
            pos = int((self.total_angle_deg / 360.0) * self.config.units_per_rev) & self.config.pos_mask
            self._update_unwrapped(pos)
            return

        self.torque_enable(True)

        # get current pos (best effort)
        ok = self.update_feedback()
        if not ok and self._last_pos_units is None:
            # fallback if we have absolutely no feedback yet
            self._seed_unwrap(0)

        p0 = self._last_pos_units if self._last_pos_units is not None else 0
        delta_units = int(round((float(delta_deg) / 360.0) * self.config.units_per_rev))
        target = (p0 + delta_units) & self.config.pos_mask

        self.move_to_units(target)

        # crude wait
        time.sleep(max(0.05, self.config.move_time_ms / 1000.0))
        if step_time_s > 0:
            time.sleep(step_time_s)

        # refresh after move
        self.update_feedback()

    def get_displacement_m(self) -> float:
        theta_rad = math.radians(self.total_angle_deg)
        return float(self.config.radius_m * theta_rad)

    def go_home(self) -> None:
        if self.simulation:
            self._seed_unwrap(0)
            self.total_pos_units = 0
            self.total_angle_deg = 0.0
            return

        self.update_feedback()
        if self._last_pos_units is None:
            self._seed_unwrap(0)
        else:
            self._seed_unwrap(self._last_pos_units)

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