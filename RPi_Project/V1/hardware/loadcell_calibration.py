# hardware/loadcell_calibration.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_latest_calibration(cal_dir: str = "calibration") -> Dict[str, Any]:
    """
    Load the newest hx711_calibration_*.json from ./calibration.
    """
    p = Path(cal_dir)
    files = sorted(p.glob("hx711_calibration_*.json"))
    if not files:
        raise FileNotFoundError(
            f"No calibration JSON found in '{cal_dir}/'. "
            f"Run calibrate_hx711.py first."
        )
    latest = files[-1]
    data = json.loads(latest.read_text())
    data["_path"] = str(latest)
    return data