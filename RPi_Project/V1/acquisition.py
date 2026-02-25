# acquisition.py
from __future__ import annotations

import time
from typing import Generator, Tuple, Optional

from hardware import MotorController, Sensors

DataPoint = Tuple[float, float, float]  # (t [s], displacement [m], force [N])


def _safe_read(motor: MotorController, sensors: Sensors) -> Optional[Tuple[float, float]]:
    try:
        # Centralised feedback update (prevents unwrap corruption)
        motor.update_feedback()
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
    Loading–unloading cycle with smoother streaming.
    """
    t0 = time.monotonic()

    sample_dt = 0.01  # 100 Hz target; raise to 0.02 if it burdens the Pi/bridge

    def stream_for(duration_s: float) -> Generator[DataPoint, None, None]:
        t_end = time.monotonic() + max(0.0, duration_s)
        while time.monotonic() < t_end:
            if stop_flag.is_set():
                return
            reading = _safe_read(motor, sensors)
            if reading is not None:
                disp, force = reading
                yield (time.monotonic() - t0), disp, force
            time.sleep(sample_dt)

    # baseline
    yield from stream_for(0.2)

    # Loading
    for _ in range(n_steps_up):
        if stop_flag.is_set():
            break
        motor.step_deg(step_deg)
        yield from stream_for(max(dwell_s, 0.15))

    # Unloading
    for _ in range(n_steps_up):
        if stop_flag.is_set():
            break
        motor.step_deg(-step_deg)
        yield from stream_for(max(dwell_s, 0.15))

    # stop at end
    try:
        motor.stop()
    except Exception:
        pass