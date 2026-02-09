# hardware/__init__.py

from .motor import MotorController
from .sensors import Sensors, ForceCal

# Optional exports (Pi-only hardware)
try:
    from .thermocouple import ThermocoupleAD8495, ThermocoupleConfig
except ModuleNotFoundError:
    # Allow import on non-Raspberry Pi systems
    ThermocoupleAD8495 = None  # type: ignore
    ThermocoupleConfig = None  # type: ignore