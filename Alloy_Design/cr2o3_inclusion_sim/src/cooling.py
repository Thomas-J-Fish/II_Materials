"""
cooling.py
----------
Maps physical casting parameters to a dimensionless simulation stopping time.

Experimental conditions modelled
----------------------------------
  Melt temperature:  2100 degC
  Cooling method:    copper block contact
  Cooling rate:      5 – 100 degC/s
  delta_T:           820 degC  (T_melt - T_solidus = 2100 - 1280)

Physical model
--------------
    t_phys [s]  =  t_dimless  *  dx^2 / (M_phys * sigma)

    t_available = delta_T / cooling_rate

    t_dimless_max = t_available * M_phys * sigma / dx^2

    n_steps = t_dimless_max / dt

Mobility at 2100 degC
----------------------
At this temperature M is approximately 70x higher than at 1500 degC
(Arrhenius scaling, Q ~ 250 kJ/mol).  With M_base ~ 3e-13 at 1500 degC,
M_2100 ~ 1e-12 m^3 J^-1 s^-1.
"""

import numpy as np

DEFAULT_SIGMA        = 1.5
DEFAULT_MOBILITY     = 3e-14
DEFAULT_DELTA_T      = 820.0
DEFAULT_COOLING_RATE = 20.0


def dimless_time_budget(dx_m, cooling_rate=DEFAULT_COOLING_RATE,
                        delta_T=DEFAULT_DELTA_T, mobility=DEFAULT_MOBILITY,
                        sigma=DEFAULT_SIGMA):
    t_physical = delta_T / cooling_rate
    return t_physical * mobility * sigma / dx_m**2


def steps_from_cooling(dx_m, dt_dimless=0.1, cooling_rate=DEFAULT_COOLING_RATE,
                       delta_T=DEFAULT_DELTA_T, mobility=DEFAULT_MOBILITY,
                       sigma=DEFAULT_SIGMA, min_steps=500, max_steps=200_000):
    t_physical = delta_T / cooling_rate
    t_dimless  = t_physical * mobility * sigma / dx_m**2
    n_steps    = int(t_dimless / dt_dimless)
    n_steps    = max(min_steps, min(max_steps, n_steps))
    return n_steps, t_dimless, t_physical


def relaxation_time_scale(radius_m, mobility=DEFAULT_MOBILITY, sigma=DEFAULT_SIGMA):
    return radius_m**2 / (mobility * sigma)


def print_time_scales(dx_m, radius_m, cooling_rate=DEFAULT_COOLING_RATE,
                      delta_T=DEFAULT_DELTA_T, mobility=DEFAULT_MOBILITY,
                      sigma=DEFAULT_SIGMA):
    t_solid = delta_T / cooling_rate
    t_relax = relaxation_time_scale(radius_m, mobility, sigma)
    t_dimless, _, _ = steps_from_cooling(dx_m, 0.1, cooling_rate,
                                         delta_T, mobility, sigma)
    print("\n  --- Physical time scales ---")
    print(f"  Solidification time  τ_solid  = {t_solid:.2f} s  "
          f"(δT={delta_T:.0f}°C, dT/dt={cooling_rate:.1f}°C/s)")
    print(f"  Relaxation time      τ_relax  = {t_relax:.4f} s  "
          f"(R={radius_m*1e6:.1f}µm, M={mobility:.1e}, σ={sigma:.1f})")
    ratio = t_solid / t_relax
    verdict = (">>" if ratio > 5 else "~") + " 1"
    print(f"  Ratio τ_solid/τ_relax = {ratio:.2f}  ({verdict})")
    print(f"  Dimensionless time budget: {t_dimless:.1f}")