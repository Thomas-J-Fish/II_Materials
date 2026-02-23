# acquisition.py
from __future__ import annotations

import time
from typing import Generator, Tuple, Optional

from hardware import MotorController, Sensors

DataPoint = Tuple[float, float, float]  # (t [s], displacement [m], force [N])


def _safe_read(sensors: Sensors) -> Optional[Tuple[float, float]]:
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
    Run a single loading–unloading cycle.

    Key change vs your old version:
      - After each step command, we stream multiple sensor points for ~dwell_s seconds
      - This makes the GUI plot look "live" and you can see load/position evolve
    """
    t0 = time.monotonic()

    def stream_for(duration_s: float) -> Generator[DataPoint, None, None]:
        t_end = time.monotonic() + max(0.0, duration_s)
        while time.monotonic() < t_end:
            if stop_flag.is_set():
                return
            reading = _safe_read(sensors)
            if reading is not None:
                disp, force = reading
                yield (time.monotonic() - t0), disp, force
            time.sleep(0.02)  # ~50 Hz like your telemetry stream

    # initial baseline stream
    yield from stream_for(0.2)

    # Loading
    for _ in range(n_steps_up):
        if stop_flag.is_set():
            break

        motor.step_deg(step_deg)

        # stream during/after move
        yield from stream_for(max(dwell_s, 0.1))

    # Unloading
    for _ in range(n_steps_up):
        if stop_flag.is_set():
            break

        motor.step_deg(-step_deg)

        yield from stream_for(max(dwell_s, 0.1))

    # Stop at the end (matches your working method)
    try:
        motor.stop()
    except Exception:
        pass