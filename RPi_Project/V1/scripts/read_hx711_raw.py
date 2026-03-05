#!/usr/bin/env python3
from __future__ import annotations

import time
import statistics

from V1.hardware.hx711 import HX711, HX711Config


def main() -> None:
    hx = HX711(HX711Config(dout_pin=5, sck_pin=6, gain=128))
    print("Reading raw HX711 counts. Ctrl+C to stop.\n")

    try:
        while True:
            vals = []
            for _ in range(25):
                vals.append(hx.read_raw())
                time.sleep(0.003)

            med = int(statistics.median(vals))
            mn = int(min(vals))
            mx = int(max(vals))
            span = mx - mn

            print(f"median={med:>10d}  min={mn:>10d}  max={mx:>10d}  span={span:>8d}")
            time.sleep(0.2)

    except KeyboardInterrupt:
        pass
    finally:
        hx.close()


if __name__ == "__main__":
    main()