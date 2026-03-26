"""
cooling.py
----------
Maps physical casting parameters (cooling rate, mushy zone width, interfacial
mobility) to a dimensionless simulation stopping time.

Physical model
--------------
The Allen-Cahn equation in physical units is:

    dphi/dt_phys = M_phys * [ sigma * kappa_curv - ... ]

where M_phys [m^3 J^-1 s^-1] is the interface mobility and sigma [J m^-2] is
the interfacial energy.  In our dimensionless pixel-unit solver, the
correspondence is:

    t_phys [s]  =  t_dimless  *  dx^2 / (M_phys * sigma)

The inclusion shape is frozen when the matrix solidifies.  The available
physical time for shape relaxation is:

    t_available [s]  =  delta_T [°C]  /  dT_dt [°C s^-1]

where:
  delta_T  - width of the mushy zone (liquidus - solidus temperature)  [°C]
  dT_dt    - cooling rate  [°C s^-1]

Combining:

    t_dimless_max  =  t_available * M_phys * sigma / dx^2
                   =  (delta_T / dT_dt) * M_phys * sigma / dx^2

This is the dimensionless time budget — the simulation should run for this
many dimensionless time units before the shape is frozen.

The corresponding number of simulation steps is:

    n_steps  =  t_dimless_max / dt

where dt = 0.1 is the stable time step.

Mobility uncertainty
--------------------
The physical mobility M_phys of a Cr2O3/Fe-alloy melt interface is not
directly measured in the literature.  Physically reasonable estimates for
solid oxide / liquid metal interfaces range from ~1e-10 to ~1e-8 m^3 J^-1 s^-1.
This uncertainty spans two orders of magnitude and is the dominant source of
quantitative error in any time-scale prediction.  The simulation therefore
treats M_phys as a parameter to vary, allowing you to ask:
  "For what mobility does the model reproduce my observed degree of rounding?"

Typical casting parameters for this alloy
------------------------------------------
  Liquidus temperature:   ~1380 °C   (estimated from CALPHAD for this composition)
  Solidus temperature:    ~1280 °C   (estimated)
  Mushy zone width:       ~100 °C
  Cooling rate (casting): 1 – 20 °C/s  (investment/sand casting)
  Cooling rate (quench):  100 – 1000 °C/s
"""

import numpy as np


# ---------------------------------------------------------------------------
# Default physical parameters — override in CONFIG
# ---------------------------------------------------------------------------

DEFAULT_SIGMA        = 1.5     # Cr2O3/melt interfacial energy  [J m^-2]
DEFAULT_MOBILITY     = 3e-13    # Interface mobility  [m^3 J^-1 s^-1]  (mid-range estimate)
DEFAULT_DELTA_T      = 100.0   # Mushy zone width  [°C]
DEFAULT_COOLING_RATE = 5.0     # Cooling rate  [°C s^-1]  (moderate investment casting)


# ---------------------------------------------------------------------------
# Core calculation
# ---------------------------------------------------------------------------

def dimless_time_budget(
    dx_m:          float,
    cooling_rate:  float = DEFAULT_COOLING_RATE,
    delta_T:       float = DEFAULT_DELTA_T,
    mobility:      float = DEFAULT_MOBILITY,
    sigma:         float = DEFAULT_SIGMA,
) -> float:
    """
    Compute the dimensionless time budget available for shape relaxation before
    the matrix solidifies.

    Parameters
    ----------
    dx_m          : grid spacing in metres
    cooling_rate  : dT/dt in °C s^-1  (positive value)
    delta_T       : mushy zone width in °C  (liquidus - solidus)
    mobility      : physical interface mobility  [m^3 J^-1 s^-1]
    sigma         : interfacial energy  [J m^-2]

    Returns
    -------
    t_dimless : float
        Dimensionless time available for shape relaxation.
    """
    t_physical  = delta_T / cooling_rate                  # seconds available
    t_dimless   = t_physical * mobility * sigma / dx_m**2
    return t_dimless


def steps_from_cooling(
    dx_m:          float,
    dt_dimless:    float  = 0.1,
    cooling_rate:  float  = DEFAULT_COOLING_RATE,
    delta_T:       float  = DEFAULT_DELTA_T,
    mobility:      float  = DEFAULT_MOBILITY,
    sigma:         float  = DEFAULT_SIGMA,
    min_steps:     int    = 500,
    max_steps:     int    = 200_000,
) -> tuple:
    """
    Compute the number of simulation steps corresponding to the available
    physical solidification time.

    Parameters
    ----------
    dx_m          : grid spacing in metres
    dt_dimless    : dimensionless time step (default 0.1)
    cooling_rate  : dT/dt in °C s^-1
    delta_T       : mushy zone width in °C
    mobility      : physical interface mobility  [m^3 J^-1 s^-1]
    sigma         : interfacial energy  [J m^-2]
    min_steps     : floor on returned step count (avoids trivially short runs)
    max_steps     : ceiling on returned step count

    Returns
    -------
    n_steps      : int    number of simulation steps
    t_dimless    : float  corresponding dimensionless time
    t_physical   : float  physical time available  [s]
    """
    t_physical = delta_T / cooling_rate
    t_dimless  = t_physical * mobility * sigma / dx_m**2
    n_steps    = int(t_dimless / dt_dimless)
    n_steps    = max(min_steps, min(max_steps, n_steps))
    return n_steps, t_dimless, t_physical


def mobility_at_T(T_celsius, M0, Q=250e3, T_ref_celsius=2100.0):
    """
    Arrhenius temperature-dependent mobility, normalised so that
    M(T_ref_celsius) = M0 exactly.

    Uses the ratio form to avoid numerical issues with the raw exponential:

        M(T) = M0 * exp(-Q/R * (1/T - 1/T_ref))

    As the alloy cools from T_melt toward the solidus, M drops by roughly
    two orders of magnitude.  A fast-cooled sample passes through the
    high-mobility temperature range more quickly, so its effective rounding
    rate per unit physical time is genuinely lower than a slow-cooled sample.

    Parameters
    ----------
    T_celsius     : float  current temperature [°C]
    M0            : float  mobility at T_ref_celsius [m^3 J^-1 s^-1]
    Q             : float  activation energy [J/mol], default 250 kJ/mol
    T_ref_celsius : float  reference temperature at which M = M0 [°C],
                           should match T_melt in CONFIG

    Returns
    -------
    M : float  effective mobility at T_celsius  [m^3 J^-1 s^-1]
    """
    R_gas     = 8.314
    T_k       = T_celsius       + 273.15
    T_ref_k   = T_ref_celsius   + 273.15
    return M0 * np.exp(-Q / R_gas * (1.0 / T_k - 1.0 / T_ref_k))


def relaxation_time_scale(
    radius_m:   float,
    mobility:   float = DEFAULT_MOBILITY,
    sigma:      float = DEFAULT_SIGMA,
) -> float:
    """
    Estimate the characteristic shape-relaxation time for an inclusion of
    radius `radius_m`.

    The relaxation time scales as R^2 / (M * sigma), which means:
      - larger inclusions relax more slowly
      - higher mobility or higher surface energy speeds up relaxation

    Parameters
    ----------
    radius_m  : inclusion radius in metres
    mobility  : physical mobility  [m^3 J^-1 s^-1]
    sigma     : interfacial energy  [J m^-2]

    Returns
    -------
    tau : float  relaxation time in seconds
    """
    return radius_m**2 / (mobility * sigma)


def print_time_scales(
    dx_m:           float,
    radius_m:       float,
    cooling_rate:   float = DEFAULT_COOLING_RATE,
    delta_T:        float = DEFAULT_DELTA_T,
    mobility:       float = DEFAULT_MOBILITY,
    sigma:          float = DEFAULT_SIGMA,
):
    """Print a summary of all relevant time scales for a given configuration."""
    t_solid = delta_T / cooling_rate
    t_relax = relaxation_time_scale(radius_m, mobility, sigma)
    t_dimless, _, _ = steps_from_cooling(dx_m, 0.1, cooling_rate, delta_T, mobility, sigma)

    print("\n  --- Physical time scales ---")
    print(f"  Solidification time  τ_solid  = {t_solid:.2f} s  "
          f"(δT={delta_T:.0f}°C, dT/dt={cooling_rate:.1f}°C/s)")
    print(f"  Relaxation time      τ_relax  = {t_relax:.4f} s  "
          f"(R={radius_m*1e6:.1f}µm, M={mobility:.1e}, σ={sigma:.1f})")
    print(f"  Ratio τ_solid/τ_relax = {t_solid/t_relax:.2f}  "
          f"({'>> 1 → full rounding expected' if t_solid/t_relax > 5 else '~ 1 → partial rounding' if t_solid/t_relax > 0.5 else '<< 1 → minimal rounding'})")
    print(f"  Dimensionless time budget: {t_dimless:.1f}")