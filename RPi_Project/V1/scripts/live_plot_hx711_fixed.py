#!/usr/bin/env python3
from __future__ import annotations

import time
import csv
import statistics
from pathlib import Path
from typing import Optional, List, Dict, Any
from collections import deque
import json
import glob

import matplotlib.pyplot as plt
from V1.hardware.loadcell import LoadCell
from V1.hardware.loadcell import LoadCellConfig


# Helper to load JSON path (for info only)
def load_latest_calibration_info(cal_dir: str = None) -> Dict[str, Any]:
    if cal_dir is None:
        # anchor to package
        here = Path(__file__).resolve().parents[1]
        cal_dir = str(here / "calibration")
    paths = sorted(glob.glob(str(Path(cal_dir) / "hx711_calibration_*.json")))
    if not paths:
        raise FileNotFoundError(f"No calibration files found in '{cal_dir}/'. Run calibrate_hx711 first.")
    latest = paths[-1]
    with open(latest, "r") as f:
        data = json.load(f)
    data["_path"] = latest
    return data


def main() -> None:
    # Load calibration info just for printing
    cal = load_latest_calibration_info()

    print("Loaded calibration:", cal["_path"])
    print(f"GPIO DOUT={cal['dout_pin']}, SCK={cal['sck_pin']}")
    print(f"OFFSET (saved)={cal['offset_counts']}, SCALE (saved)={cal['scale_counts_per_g']:.6f} counts/g")
    print("Note: this test script will perform a runtime tare so the plot baseline is 0 g.")

    # Create LoadCell (real)
    lc = LoadCell(simulation=False, cfg=LoadCellConfig())

    # Perform runtime tare: set current raw value as baseline so plot shows 0 g now
    input("\nEnsure holder only (no extra weight). Press Enter to perform runtime tare...")
    ok = lc.tare_to_current()
    print("Runtime tare OK" if ok else "Tare FAILED (check wiring/stability)")
    print("Now plotting. Apply/release force to confirm response.\n")

    # CSV logging
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_name = f"hx711_net_mass_log_{ts}.csv"
    print(f"Logging to {csv_name}")

    # Plot setup
    plt.ion()
    fig, ax = plt.subplots(figsize=(8,5))
    ax.set_title("HX711 net mass vs time (relative to holder baseline)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Net mass (g)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)

    line, = ax.plot([], [], lw=2)
    window_s = 30.0
    t0 = time.time()
    times = deque()
    masses = deque()

    with open(csv_name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "raw_counts", "offset_counts", "scale_counts_per_g", "net_mass_g"])

        try:
            while plt.fignum_exists(fig.number):
                m_g = lc.read_mass_g()
                # if read failed, skip
                if m_g is None:
                    # small sleep to avoid tight loop if hardware noisy
                    time.sleep(0.05)
                    continue

                # read raw via the history (approx): we can't get raw here easily, so log mass
                t = time.time() - t0

                times.append(t)
                masses.append(m_g)

                # keep last N seconds
                while times and (times[-1] - times[0]) > window_s:
                    times.popleft()
                    masses.popleft()

                # write row
                w.writerow([f"{t:.3f}", "", f"{lc.offset_counts:.0f}", f"{lc.scale_counts_per_g:.9f}", f"{m_g:.3f}"])
                f.flush()

                # update plot
                line.set_data(list(times), list(masses))
                ax.relim()
                ax.autoscale_view()
                plt.pause(0.01)

        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            lc.close()

    print("Done.")


if __name__ == "__main__":
    main()