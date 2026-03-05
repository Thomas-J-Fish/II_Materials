#!/usr/bin/env python3
from __future__ import annotations

import time
import json
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List

# Use your in-repo HX711 GPIO driver
from hardware.hx711 import HX711, HX711Config


@dataclass
class CalibrationResult:
    dout_pin: int
    sck_pin: int
    offset_counts: int                # raw counts with holder only (baseline)
    scale_counts_per_g: float         # counts per gram of added mass
    baseline_raw_median: int
    cal_raw_median: int
    added_mass_g: float
    net_counts: int
    timestamp: str


def read_raw_median(hx: HX711, samples: int = 25, delay_s: float = 0.003) -> Optional[int]:
    vals: List[int] = []
    for _ in range(max(5, samples)):
        try:
            vals.append(int(hx.read_raw()))
        except Exception:
            pass
        time.sleep(delay_s)
    if len(vals) < 5:
        return None
    return int(statistics.median(vals))


def measure_median_over_time(hx: HX711, duration_s: float = 5.0) -> int:
    vals: List[int] = []
    t0 = time.time()
    while time.time() - t0 < duration_s:
        v = read_raw_median(hx, samples=15, delay_s=0.003)
        if v is not None:
            vals.append(v)
        time.sleep(0.05)

    if len(vals) < 8:
        raise RuntimeError("Not enough valid samples. Check wiring / stability.")
    return int(statistics.median(vals))


def main() -> None:
    # ---- Pins (BCM) ----
    # Must match your wiring: DATA->GPIO5, SCK->GPIO6
    DOUT = 5
    SCK = 6

    # Always save into repo-root calibration folder (../calibration relative to scripts/)
    out_dir = Path(__file__).resolve().parents[1] / "calibration"
    out_dir.mkdir(exist_ok=True)

    hx = HX711(HX711Config(dout_pin=DOUT, sck_pin=SCK, gain=128))

    print("\n=== HX711 Two-Point Calibration (holder = baseline) ===")
    print("Step 1: Put ONLY the sample holder on the load cell.")
    print("        No extra weights. Keep everything still.\n")
    input("Press Enter to start baseline measurement...")

    baseline = measure_median_over_time(hx, duration_s=5.0)
    print(f"\nBaseline raw median (holder only) = {baseline}")

    print("\nStep 2: Add a known weight stack ON TOP of the holder.")
    print("        Enter ONLY the ADDED mass (not including holder).")
    added_mass_g = float(input("\nAdded mass [g] (e.g. 150): ").strip())
    if added_mass_g <= 0:
        raise ValueError("Added mass must be > 0 g")

    input("\nPress Enter to start calibration measurement...")

    cal_raw = measure_median_over_time(hx, duration_s=5.0)
    print(f"\nCalibration raw median (holder + added mass) = {cal_raw}")

    net_counts = cal_raw - baseline
    if net_counts == 0:
        raise RuntimeError("Net counts is 0. No signal change detected.")

    scale_counts_per_g = net_counts / added_mass_g

    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    result = CalibrationResult(
        dout_pin=DOUT,
        sck_pin=SCK,
        offset_counts=baseline,
        scale_counts_per_g=scale_counts_per_g,
        baseline_raw_median=baseline,
        cal_raw_median=cal_raw,
        added_mass_g=added_mass_g,
        net_counts=net_counts,
        timestamp=ts,
    )

    out_path = out_dir / f"hx711_calibration_{ts}.json"
    out_path.write_text(json.dumps(asdict(result), indent=2))

    print("\n=== Results ===")
    print(f"OFFSET (holder baseline) counts: {baseline}")
    print(f"NET counts (cal - baseline):     {net_counts}")
    print(f"SCALE counts per gram:          {scale_counts_per_g:.6f} counts/g")
    print(f"\nSaved calibration to: {out_path}")

    try:
        hx.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()