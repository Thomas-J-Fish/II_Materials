# scripts/calibrate_hx711.py

from __future__ import annotations

import time

from hardware.motor import MotorController
from hardware.sensors import Sensors, ForceCal
from hardware.hx711 import HX711Config


def main() -> None:
    # No motor needed for calibration (but Sensors expects one)
    motor = MotorController(simulation=True)

    sensors = Sensors(
        motor=motor,
        simulation=False,
        hx711_cfg=HX711Config(dout_pin=6, sck_pin=5, gain=128),
        cal=ForceCal(tare_offset_counts=0.0, scale_N_per_count=1.0),
        avg_samples=15,
    )

    try:
        print("HX711 calibration")
        print("1) Ensure NOTHING is on the load cell.")
        input("Press Enter to tare...")
        sensors.tare(samples=30)
        print(f"Tare offset counts: {sensors.cal.tare_offset_counts:.3f}")

        print("\n2) Place a known mass on the load cell (e.g. 500 g or 1 kg).")
        mass_kg = float(input("Enter mass in kg: ").strip())
        known_force_N = mass_kg * 9.80665
        print(f"Known force: {known_force_N:.4f} N")

        input("Press Enter when the mass is stable...")
        time.sleep(0.5)

        # Read net counts under load
        raw = sensors.hx.read_average(samples=40)  # type: ignore
        net = raw - sensors.cal.tare_offset_counts
        print(f"Net counts: {net:.3f}")

        scale = known_force_N / net
        sensors.cal.scale_N_per_count = scale

        print("\nCalibration complete.")
        print(f"scale_N_per_count = {scale:.12e}")
        print("Copy this number into your config (e.g. in sensors init).")

    finally:
        sensors.close()


if __name__ == "__main__":
    main()