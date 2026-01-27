# scripts/read_hx711.py

from __future__ import annotations

import time

from hardware.motor import MotorController
from hardware.sensors import Sensors, ForceCal
from hardware.hx711 import HX711Config


def main() -> None:
    motor = MotorController(simulation=True)
    sensors = Sensors(
        motor=motor,
        simulation=False,
        hx711_cfg=HX711Config(dout_pin=6, sck_pin=5, gain=128),
        cal=ForceCal(tare_offset_counts=0.0, scale_N_per_count=1.0),
        avg_samples=10,
    )

    try:
        sensors.tare(samples=30)
        print("Reading force (uncalibrated unless you set scale). Ctrl+C to stop.")
        while True:
            F = sensors.read_force_N()
            print(f"{F: .6f} N")
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        sensors.close()


if __name__ == "__main__":
    main()