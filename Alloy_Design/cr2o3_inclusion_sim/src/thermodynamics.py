"""
thermodynamics.py
-----------------
Interfacial energy parameters for Cr2O3 inclusions in a Fe-28Cr-14Ni-8Mo (wt%) melt.

Physical background
-------------------
The Allen-Cahn equation drives the phase-field order parameter phi toward values
that minimise the total free energy of the system:

    F = integral[ (epsilon^2 / 2)|grad(phi)|^2  +  W * f_dw(phi) ] dV

where:
  epsilon   - gradient energy coefficient  (related to interface width)
  W         - height of the double-well potential barrier
  f_dw(phi) - double-well potential:  phi^2 * (1 - phi)^2

The solver operates in dimensionless pixel units.  The physical surface energy
sigma (J/m^2) enters only through the time scale, not the equilibrium shape.
The volume-conserving Lagrange multiplier in step() ensures the inclusion
retains its area throughout the simulation.

Literature values for Cr2O3 / Fe-based melt interfacial energy
---------------------------------------------------------------
Reported values range from ~1.0 to ~2.0 J/m^2.  We use 1.5 J/m^2 as a
central estimate.

Reference:
  Cramb & Jimbo, ISIJ International 32 (1992) 476-485.
"""

import numpy as np

SIGMA = 1.5          # Cr2O3 / melt interfacial energy  [J m^-2]
INTERFACE_WIDTH = 3  # Diffuse interface half-width in grid cells (informational)


def compute_ac_parameters(dx: float, interface_width_cells: int = INTERFACE_WIDTH):
    """
    Return dimensionless Allen-Cahn parameters for the pixel-unit solver.

    We use epsilon = W = mobility = 1.0.  With kappa = epsilon^2 = W = 1 the
    tanh interface half-width is pi/sqrt(2) ≈ 2.2 pixels, which is well
    resolved whenever the inclusion radius exceeds ~20 pixels.

    The volume-conserving Lagrange multiplier in step() makes the solver
    insensitive to the exact values of epsilon and W — what matters is that
    they are both O(1) in pixel units.

    Parameters
    ----------
    dx : float
        Physical grid spacing in metres (kept for API consistency).
    interface_width_cells : int
        Accepted for API compatibility; not used in computation.

    Returns
    -------
    epsilon, W, mobility : all 1.0
    """
    return 1.0, 1.0, 1.0


def double_well(phi: np.ndarray) -> np.ndarray:
    """f_dw(phi) = phi^2 * (1 - phi)^2"""
    return phi**2 * (1.0 - phi)**2


def double_well_derivative(phi: np.ndarray) -> np.ndarray:
    """df_dw/dphi = 2*phi*(1-phi)*(1-2*phi)"""
    return 2.0 * phi * (1.0 - phi) * (1.0 - 2.0 * phi)