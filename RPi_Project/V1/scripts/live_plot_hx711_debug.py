#!/usr/bin/env python3
from __future__ import annotations

import time
import csv
from pathlib import Path
from typing import Optional
from collections import deque
import json
import glob
import math

import matplotlib.pyplot as plt
from V1.hardware.loadcell import LoadCell, LoadCellConfig


def load_latest_cal_info(cal_dir: str = None):
    if cal_dir is None:
        here = Path(__file__).resolve().parents[1]
        cal_dir = str(here / "calibration")
    paths = sorted(glob.glob(str(Path(cal_dir) / "hx711_calibration_*.json")))
    if not paths:
        return None
    with open(paths[-1], "r") as f:
        d = json.load(f)
    d["_path"] = paths[-1]
    return d


def main() -> None:
    info = load_latest_cal_info()
    if info is None:
        print("No calibration JSON found. Run calibrate_hx711 first.")
        return

    print("Using calibration:", info["_path"])
    print(f"Saved OFFSET={info['offset_counts']}, SCALE={info['scale_counts_per_g']:.9g} counts/g")
    print(f"DOUT={info['dout_pin']}  SCK={info['sck_pin']}")
    print("Creating LoadCell (real).")

    lc = LoadCell(simulation=False, cfg=LoadCellConfig())

    input("\nPlace holder only, steady. Press Enter to perform runtime tare (sets baseline to 0 g)...")
    ok = lc.tare_to_current()
    print("Runtime tare OK" if ok else "Runtime tare FAILED")
    print("(Now apply/remove force to see changes. Ctrl+C to stop.)\n")

    # CSV logging
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = Path.cwd() / f"hx711_debug_{ts}.csv"
    print("Logging CSV ->", csv_path)

    # Plot setup
    plt.ion()
    fig, ax = plt.subplots(figsize=(8,5))
    ax.set_title("HX711 net mass vs time (debug)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Net mass (g)")
    ax.grid(True, linestyle="--", alpha=0.5)
    line, = ax.plot([], [], lw=2)

    # sensible initial window so small changes are visible
    y_vis = 200.0  # grams (±). If your weights are smaller use lower value
    ax.set_ylim(-y_vis, y_vis)

    window_s = 30.0
    t0 = time.time()
    times = deque()
    masses = deque()

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "raw_counts", "offset_counts", "scale_counts_per_g", "net_mass_g", "force_N"])

        try:
            while plt.fignum_exists(fig.number):
                # get raw median (internal private call — OK for debugging)
                # If your LoadCell implementation removed _read_raw_median, call read_mass_g() and derive raw (not ideal).
                try:
                    raw = getattr(lc, "_read_raw_median")()
                except Exception:
                    raw = None

                m_g = lc.read_mass_g()
                f_N = lc.read_force_n()

                t = time.time() - t0

                # Print debug line to console for diagnosis
                print(f"t={t:6.2f}s  raw={raw!s:>10}  offset={lc.offset_counts:10.0f}  "
                      f"scale={lc.scale_counts_per_g:12.6g}  mass_g={m_g!s:>9}  force_N={f_N!s:>9}")

                # write CSV (use empty strings if none)
                w.writerow([
                    f"{t:.3f}",
                    str(raw) if raw is not None else "",
                    f"{lc.offset_counts:.0f}",
                    f"{lc.scale_counts_per_g:.9g}",
                    f"{m_g:.6f}" if m_g is not None else "",
                    f"{f_N:.6f}" if f_N is not None else ""
                ])
                f.flush()

                # plotting only when we have a mass value
                if m_g is not None:
                    times.append(t)
                    masses.append(m_g)

                    # keep window
                    while times and (times[-1] - times[0]) > window_s:
                        times.popleft()
                        masses.popleft()

                    line.set_data(list(times), list(masses))

                    # If values exceed visible y-limits, expand them to include data
                    ymin, ymax = ax.get_ylim()
                    curmin = min(masses) if masses else 0.0
                    curmax = max(masses) if masses else 0.0
                    margin = max(1.0, 0.05 * (curmax - curmin if curmax != curmin else y_vis))
                    if curmin - margin < ymin or curmax + margin > ymax:
                        ax.set_ylim(min(ymin, curmin - margin), max(ymax, curmax + margin))

                    ax.relim()
                    ax.autoscale_view(scalex=False, scaley=False)

                plt.pause(0.05)

        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            lc.close()

    print("CSV saved to:", csv_path)
    print("Done.")


if __name__ == "__main__":
    main()