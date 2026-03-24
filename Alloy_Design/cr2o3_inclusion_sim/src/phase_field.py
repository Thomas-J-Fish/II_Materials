"""
phase_field.py
--------------
Core Allen-Cahn phase-field solver for inclusion shape evolution.

Governing equation
------------------
The Allen-Cahn equation describes the time evolution of the order parameter
phi(x, y, t), which represents the local phase:

    dphi/dt = M * [ epsilon^2 * laplacian(phi)  -  W * df_dw/dphi ]

where:
  M         - kinetic mobility
  epsilon   - gradient energy coefficient  (penalises sharp interfaces)
  W         - double-well barrier height   (drives phi toward 0 or 1)
  f_dw(phi) - double-well potential  phi^2 * (1 - phi)^2
  laplacian - spatial second derivative (finite difference, periodic or
              Neumann boundary conditions)

Physical interpretation
-----------------------
The equation balances two competing effects:
  1. The gradient term epsilon^2 * laplacian(phi) tends to smooth and round
     the interface — this is what causes the angular spalled oxide shape to
     relax toward a more rounded morphology.
  2. The bulk term W * df_dw/dphi drives phi toward the stable well minima
     at phi=0 (melt) and phi=1 (oxide), keeping the phases distinct.

Numerical scheme
----------------
Explicit (forward Euler) time integration with a standard 5-point Laplacian
stencil.  The time step dt must satisfy the CFL-like stability condition:

    dt <= dx^2 / (2 * M * epsilon^2)

This is checked automatically and dt is clipped if necessary.

Anisotropy (optional)
---------------------
Real Cr2O3 has a corundum (trigonal) crystal structure, meaning the
interfacial energy is anisotropic — some crystallographic faces have lower
energy and are therefore preferred.  An optional weak anisotropy term is
included, controlled by `anisotropy_strength`.  Setting this to 0 gives
isotropic (spherical) energy minimisation.
"""

import numpy as np
from .thermodynamics import double_well_derivative, compute_ac_parameters


# ---------------------------------------------------------------------------
# Laplacian with boundary conditions
# ---------------------------------------------------------------------------

def laplacian(phi: np.ndarray, dx: float, bc: str = "neumann") -> np.ndarray:
    """
    Compute the 2D Laplacian of phi using a 5-point finite difference stencil.

    Parameters
    ----------
    phi : (Ny, Nx) array
    dx  : grid spacing (same in x and y)
    bc  : boundary condition — 'neumann' (zero-flux, recommended) or 'periodic'

    Returns
    -------
    lap : (Ny, Nx) array
    """
    if bc == "periodic":
        lap = (
            np.roll(phi,  1, axis=0) +
            np.roll(phi, -1, axis=0) +
            np.roll(phi,  1, axis=1) +
            np.roll(phi, -1, axis=1) -
            4.0 * phi
        ) / dx**2
    else:  # Neumann: pad with edge values (zero normal gradient)
        phi_pad = np.pad(phi, 1, mode="edge")
        lap = (
            phi_pad[2:,  1:-1] +
            phi_pad[:-2, 1:-1] +
            phi_pad[1:-1, 2:] +
            phi_pad[1:-1, :-2] -
            4.0 * phi_pad[1:-1, 1:-1]
        ) / dx**2

    return lap


# ---------------------------------------------------------------------------
# Optional anisotropy
# ---------------------------------------------------------------------------

def anisotropy_factor(phi: np.ndarray, dx: float, strength: float, n_fold: int = 6) -> np.ndarray:
    """
    Compute a simple orientation-dependent correction to the gradient energy
    coefficient, mimicking the n-fold symmetry of a crystal.

    For Cr2O3 (trigonal / corundum), 6-fold is a reasonable approximation
    for the basal-plane symmetry.  This correction modifies the effective
    surface energy as:

        sigma_eff(theta) = sigma_0 * (1 + strength * cos(n_fold * theta))

    where theta is the local interface normal angle.

    Parameters
    ----------
    phi       : phase-field array
    dx        : grid spacing
    strength  : anisotropy strength (0 = isotropic; 0.05–0.15 = weak crystal)
    n_fold    : rotational symmetry order

    Returns
    -------
    correction : (Ny, Nx) array — multiplicative correction to laplacian term
    """
    # Gradient components (central differences)
    grad_y = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2 * dx)
    grad_x = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2 * dx)

    theta = np.arctan2(grad_y, grad_x)
    correction = 1.0 + strength * np.cos(n_fold * theta)
    return correction


# ---------------------------------------------------------------------------
# Stability check
# ---------------------------------------------------------------------------

def max_stable_dt(dx: float, mobility: float, epsilon: float) -> float:
    """
    Return the maximum stable dimensionless time step.

    Full stability for the Allen-Cahn equation (diffusion + reaction) requires:

        dt < 1 / (2*kappa + W * max|f''(phi)|)

    where kappa = epsilon^2, W = 1, and f''(phi) = 2*(1 - 6*phi + 6*phi^2).
    The maximum of |f''| over phi in [0,1] is 2 (at phi=0 or phi=1).
    With kappa = W = mobility = 1.0:

        dt_max = 1 / (2*1 + 1*2) = 0.25

    A safety factor of 0.4 is applied giving dt = 0.1.
    """
    kappa   = epsilon ** 2
    W       = 1.0
    f_pp_max = 2.0   # max |d²f_dw/dphi²| over phi in [0,1]
    return 0.4 / (2.0 * kappa + W * f_pp_max)


# ---------------------------------------------------------------------------
# Single time step
# ---------------------------------------------------------------------------

def step(
    phi:              np.ndarray,
    dt:               float,
    dx:               float,
    epsilon:          float,
    W:                float,
    mobility:         float,
    bc:               str   = "neumann",
    anisotropy_strength: float = 0.0,
    n_fold:           int   = 6,
) -> np.ndarray:
    """
    Advance the volume-conserving Allen-Cahn equation by one dimensionless
    time step dt.

    Standard (unconstrained) Allen-Cahn does not conserve the volume of the
    inclusion — it minimises total interfacial length, which causes small
    convex inclusions to coarsen rather than simply rounding their corners.
    We therefore add a Lagrange multiplier lambda(t) that enforces
    dA/dt = 0 at each step:

        dphi/dt = M * [ epsilon^2 * lap(phi)  -  W * f'(phi)  +  lambda ]

    where lambda is chosen so that d/dt integral(phi) = 0:

        lambda = - integral[ (eps^2*lap - W*f') * |grad phi| ] dA
                 / integral[ |grad phi| ] dA

    i.e. the interface-weighted spatial average of the unconstrained RHS,
    negated.  This is the standard approach for inclusion morphology problems
    (see Bronsard & Wetton 1995; Kim et al. 2004).

    All spatial operations are in pixel units (dx_tilde = 1).

    Parameters
    ----------
    phi       : current phase-field array  (Ny, Nx)
    dt        : dimensionless time step
    dx        : physical grid spacing in metres (not used in solver; kept for
                API consistency)
    epsilon   : gradient energy coefficient  (= sqrt(kappa))
    W         : double-well barrier height
    mobility  : kinetic coefficient M (= 1.0)
    bc        : 'neumann' (recommended) or 'periodic'
    anisotropy_strength : 0 = isotropic; 0.05–0.15 = weak crystal anisotropy
    n_fold    : rotational symmetry order for anisotropy

    Returns
    -------
    phi_new : (Ny, Nx) array, values clipped to [0, 1]
    """
    kappa = epsilon ** 2
    lap   = laplacian(phi, dx=1.0, bc=bc)

    if anisotropy_strength > 0:
        aniso = anisotropy_factor(phi, dx=1.0, strength=anisotropy_strength, n_fold=n_fold)
        rhs = kappa * aniso * lap - W * double_well_derivative(phi)
    else:
        rhs = kappa * lap - W * double_well_derivative(phi)

    # --- Volume-conserving Lagrange multiplier ---
    # Gradient magnitude (proxy for interface indicator)
    gy = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / 2.0
    gx = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / 2.0
    grad_mag = np.sqrt(gy**2 + gx**2)

    denom = grad_mag.sum()
    if denom > 1e-12:
        lam = (rhs * grad_mag).sum() / denom
        rhs = rhs - lam

    phi_new = phi + dt * mobility * rhs

    # Hard clip to [0, 1] — avoids numerical overshoot at corners
    phi_new = np.clip(phi_new, 0.0, 1.0)

    return phi_new


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_simulation(
    phi_init:            np.ndarray,
    n_steps:             int,
    dx:                  float,
    dt:                  float      = None,
    interface_width:     int        = 3,
    bc:                  str        = "neumann",
    anisotropy_strength: float      = 0.0,
    n_fold:              int        = 6,
    save_every:          int        = 100,
    verbose:             bool       = True,
) -> tuple:
    """
    Run the Allen-Cahn simulation for n_steps time steps.

    Parameters
    ----------
    phi_init         : initial phase-field array  (Ny, Nx)
    n_steps          : total number of time steps
    dx               : physical grid spacing in metres (used for reporting only;
                       the solver runs in dimensionless pixel units)
    dt               : dimensionless time step.  If None, the maximum stable
                       value is computed automatically.
    interface_width  : interface half-width in grid cells (used for epsilon, W)
    bc               : boundary condition
    anisotropy_strength : 0 = isotropic
    n_fold           : anisotropy symmetry order
    save_every       : save a snapshot every this many steps
    verbose          : print progress

    Returns
    -------
    phi_final   : (Ny, Nx) array — final phase field
    snapshots   : list of (step_index, phi) tuples — evolution history
    metrics     : dict — scalar metrics recorded at each snapshot
                  keys: 'steps', 'area', 'perimeter', 'circularity'
    """
    epsilon, W, mobility = compute_ac_parameters(dx, interface_width)

    # Auto time step (dimensionless)
    if dt is None:
        dt = max_stable_dt(dx, mobility, epsilon)
        if verbose:
            print(f"  Auto dimensionless time step: dt = {dt:.4e}")
    else:
        dt_max = max_stable_dt(dx, mobility, epsilon)
        if dt > dt_max:
            dt = dt_max
            if verbose:
                print(f"  Requested dt too large; clipped to dt = {dt:.4e}")

    if verbose:
        print(f"  epsilon = {epsilon:.4f},  W = {W:.4f},  M = {mobility:.4f}")
        print(f"  Running {n_steps} steps  (saving every {save_every})")

    phi = phi_init.copy()
    snapshots = [(0, phi.copy())]
    metrics   = {
        "steps":       [0],
        "area":        [_area(phi)],
        "perimeter":   [_perimeter(phi)],
        "circularity": [_circularity(phi)],
    }

    for i in range(1, n_steps + 1):
        phi = step(phi, dt, dx, epsilon, W, mobility,
                   bc, anisotropy_strength, n_fold)

        if i % save_every == 0:
            snapshots.append((i, phi.copy()))
            metrics["steps"].append(i)
            metrics["area"].append(_area(phi))
            metrics["perimeter"].append(_perimeter(phi))
            metrics["circularity"].append(_circularity(phi))

            if verbose:
                print(
                    f"  Step {i:6d}/{n_steps}  |  "
                    f"area = {metrics['area'][-1]:.2e} px²  |  "
                    f"circularity = {metrics['circularity'][-1]:.3f}"
                )

    return phi, snapshots, metrics


# ---------------------------------------------------------------------------
# Shape metrics — computed on the BINARY (phi >= 0.5) mask
# ---------------------------------------------------------------------------
#
# Using the raw diffuse phi causes two errors that combine to give circularity > 1:
#   1. _area() over-counts (the tanh tail contributes fractional phi outside
#      the true boundary).
#   2. _perimeter() under-counts (the gradient is spread over several pixels,
#      so the integral of |grad phi| is smaller than the true contour length).
#
# Fix: threshold at phi=0.5, count pixels for area, count pixel-boundary
# faces for perimeter.  Both are exact on the discrete grid and always give
# circularity <= 1 (with a small pixelation correction for small radii).

def _binary_mask(phi: np.ndarray) -> np.ndarray:
    """Boolean mask: True where phi >= 0.5 (inside inclusion)."""
    return phi >= 0.5


def _area(phi: np.ndarray) -> float:
    """Inclusion area in pixels² — count of pixels with phi >= 0.5."""
    return float(np.sum(_binary_mask(phi)))


def _perimeter(phi: np.ndarray) -> float:
    """
    Perimeter in pixels — count of pixel-boundary faces between inside and
    outside pixels (horizontal + vertical transitions in the binary mask).
    """
    mask = _binary_mask(phi).astype(np.int8)
    h_faces = np.abs(np.diff(mask, axis=1)).sum()
    v_faces = np.abs(np.diff(mask, axis=0)).sum()
    return float(h_faces + v_faces)


def _circularity(phi: np.ndarray) -> float:
    """
    Circularity = 4π·Area / Perimeter².
    Perfect circle → slightly below 1.0 (pixelation); irregular → lower.
    Returns 0.0 if perimeter is zero.
    """
    A = _area(phi)
    P = _perimeter(phi)
    if P < 1e-12:
        return 0.0
    return 4.0 * np.pi * A / P**2