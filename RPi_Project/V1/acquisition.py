# acquisition.py

from __future__ import annotations

import time
from typing import Generator, Tuple, Optional

from hardware import MotorController, Sensors

DataPoint = Tuple[float, float, float]  # (t [s], displacement [m], force [N])


def _safe_read(sensors: Sensors) -> Optional[Tuple[float, float]]:
    """Return (disp, force) or None if a read fails."""
    try:
        disp = float(sensors.read_displacement_m())
        force = float(sensors.read_force_N())
        return disp, force
    except Exception:
        return None


def run_loading_cycle(
    motor: MotorController,
    sensors: Sensors,
    stop_flag,
    step_deg: float = 1.0,
    n_steps_up: int = 50,
    dwell_s: float = 0.05,
) -> Generator[DataPoint, None, None]:
    """
    Run a single loading–unloading cycle (one datapoint per step).

    :yield: (t, displacement, force) tuples after each move.
    """
    t0 = time.monotonic()

    # Optional initial datapoint (comment out if you truly want ONLY per-step)
    reading = _safe_read(sensors)
    if reading is not None:
        disp, force = reading
        yield 0.0, disp, force

    # Loading
    for _ in range(n_steps_up):
        if stop_flag.is_set():
            return

        motor.step_deg(step_deg)

        if dwell_s > 0:
            time.sleep(dwell_s)

        reading = _safe_read(sensors)
        if reading is None:
            continue
        disp, force = reading
        t = time.monotonic() - t0
        yield t, disp, force

    # Unloading
    for _ in range(n_steps_up):
        if stop_flag.is_set():
            return

        motor.step_deg(-step_deg)

        if dwell_s > 0:
            time.sleep(dwell_s)

        reading = _safe_read(sensors)
        if reading is None:
            continue
        disp, force = reading
        t = time.monotonic() - t0
        yield t, disp, force