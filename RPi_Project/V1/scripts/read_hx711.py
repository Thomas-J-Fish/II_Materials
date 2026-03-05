#!/usr/bin/env python3
from __future__ import annotations

import time
from hardware.loadcell import LoadCell


def main() -> None:
    lc = LoadCell(simulation=False)
    print(f"Using calibration: {lc.calibration_path}")

    print("Taring to current... (holder only, no extra load)")
    ok = lc.tare_to_current()
    print("Tare OK" if ok else "Tare FAILED")

    try:
        while True:
            m_g = lc.read_mass_g()
            f_N = lc.read_force_n()

            if m_g is not None and f_N is not None:
                print(f"{m_g:10.2f} g   {f_N:10.4f} N")
            else:
                print("No reading")

            time.sleep(0.2)

    except KeyboardInterrupt:
        pass
    finally:
        lc.close()


if __name__ == "__main__":
    main()