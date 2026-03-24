"""
initialisation.py
-----------------
Functions to build the initial phase-field order parameter array phi(x, y).

Convention
----------
  phi = 1  inside the inclusion  (Cr2O3)
  phi = 0  in the surrounding melt

The interface between the two phases is diffuse, with a smooth tanh profile
applied over INTERFACE_WIDTH grid cells.

Inclusion geometries available
-------------------------------
  1. irregular_polygon  – a randomised convex or non-convex polygon meant to
                          mimic the angular, spalled shape of oxide fragments.
                          This is the physically appropriate starting geometry.
  2. ellipse            – useful as a reference / sanity check (should remain
                          approximately elliptical as it relaxes).
  3. from_image         – digitise an inclusion boundary from a micrograph PNG.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.draw import polygon as sk_polygon
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu


# ---------------------------------------------------------------------------
# Smooth step: tanh interface profile
# ---------------------------------------------------------------------------

def _apply_tanh_interface(mask: np.ndarray, interface_width: float) -> np.ndarray:
    """
    Convert a binary mask (1 inside, 0 outside) into a smooth phase-field
    profile using the signed distance function and a tanh transition.

    Parameters
    ----------
    mask : np.ndarray  (Ny, Nx), dtype bool
        Binary inclusion mask.
    interface_width : float
        Half-width of the diffuse interface in pixels.

    Returns
    -------
    phi : np.ndarray  (Ny, Nx), float64, values in [0, 1]
    """
    # Signed distance: positive inside, negative outside
    dist_in  = distance_transform_edt(mask)
    dist_out = distance_transform_edt(~mask)
    signed_dist = dist_in - dist_out  # pixels

    phi = 0.5 * (1.0 + np.tanh(signed_dist / interface_width))
    return phi


# ---------------------------------------------------------------------------
# Geometry 1: irregular polygon
# ---------------------------------------------------------------------------

def irregular_polygon(
    grid_shape: tuple,
    centre: tuple = None,
    mean_radius: float = None,
    n_vertices: int = 10,
    roughness: float = 0.35,
    seed: int = 42,
    interface_width: float = 3.0,
) -> np.ndarray:
    """
    Generate an irregular (non-convex) polygon meant to approximate the angular
    shape of a spalled oxide fragment.

    Parameters
    ----------
    grid_shape : (Ny, Nx)
        Shape of the simulation grid in pixels.
    centre : (cy, cx), optional
        Centre of the inclusion in pixels.  Defaults to the grid centre.
    mean_radius : float, optional
        Mean radius of the inclusion in pixels.  Defaults to ~15% of min(Ny, Nx).
    n_vertices : int
        Number of polygon vertices.  10–16 gives a convincingly irregular shape.
    roughness : float
        Fractional standard deviation of the vertex radii.  0 = perfect circle,
        0.5 = very jagged.  0.3–0.4 mimics spalled oxide well.
    seed : int
        Random seed for reproducibility.
    interface_width : float
        Diffuse interface half-width in pixels.

    Returns
    -------
    phi : np.ndarray  (Ny, Nx), float64
    """
    Ny, Nx = grid_shape
    if centre is None:
        centre = (Ny // 2, Nx // 2)
    if mean_radius is None:
        mean_radius = 0.15 * min(Ny, Nx)

    rng = np.random.default_rng(seed)

    # Random angles and radii
    angles = np.sort(rng.uniform(0, 2 * np.pi, n_vertices))
    radii  = mean_radius * (1.0 + roughness * rng.standard_normal(n_vertices))
    radii  = np.clip(radii, 0.3 * mean_radius, 2.0 * mean_radius)

    # Vertex coordinates (row, col)
    cy, cx = centre
    rows = cy + radii * np.sin(angles)
    cols = cx + radii * np.cos(angles)

    rows = np.clip(rows, 1, Ny - 2)
    cols = np.clip(cols, 1, Nx - 2)

    # Rasterise polygon
    mask = np.zeros((Ny, Nx), dtype=bool)
    rr, cc = sk_polygon(rows, cols, shape=(Ny, Nx))
    mask[rr, cc] = True

    return _apply_tanh_interface(mask, interface_width)


# ---------------------------------------------------------------------------
# Geometry 2: ellipse (reference case)
# ---------------------------------------------------------------------------

def ellipse(
    grid_shape: tuple,
    centre: tuple = None,
    semi_axes: tuple = None,
    angle_deg: float = 0.0,
    interface_width: float = 3.0,
) -> np.ndarray:
    """
    Generate an elliptical inclusion for use as a reference / sanity-check case.

    Parameters
    ----------
    grid_shape : (Ny, Nx)
    centre : (cy, cx)
    semi_axes : (a, b) in pixels   a = semi-major, b = semi-minor
    angle_deg : float
        Rotation of the major axis from horizontal, in degrees.
    interface_width : float

    Returns
    -------
    phi : np.ndarray  (Ny, Nx), float64
    """
    Ny, Nx = grid_shape
    if centre is None:
        centre = (Ny // 2, Nx // 2)
    if semi_axes is None:
        semi_axes = (0.12 * min(Ny, Nx), 0.08 * min(Ny, Nx))

    cy, cx = centre
    a, b   = semi_axes
    theta  = np.deg2rad(angle_deg)

    yy, xx = np.mgrid[0:Ny, 0:Nx]
    dy = yy - cy
    dx = xx - cx

    # Rotated ellipse equation
    y_rot =  dy * np.cos(theta) + dx * np.sin(theta)
    x_rot = -dy * np.sin(theta) + dx * np.cos(theta)

    mask = ((x_rot / a)**2 + (y_rot / b)**2) <= 1.0

    return _apply_tanh_interface(mask, interface_width)


# ---------------------------------------------------------------------------
# Geometry 3: from micrograph image
# ---------------------------------------------------------------------------

def from_image(
    image_path: str,
    grid_shape: tuple = None,
    interface_width: float = 3.0,
    invert: bool = False,
) -> np.ndarray:
    """
    Extract an inclusion mask from a micrograph image (PNG/TIF/JPG).

    The image should contain a single dark inclusion on a lighter background
    (or vice versa — use invert=True).  An Otsu threshold is applied to
    binarise the image.

    Parameters
    ----------
    image_path : str
        Path to the micrograph image.
    grid_shape : (Ny, Nx), optional
        If given, the extracted mask is resized to this shape.
        If None, the image's native resolution is used.
    interface_width : float
    invert : bool
        Set True if the inclusion is bright on a dark background.

    Returns
    -------
    phi : np.ndarray  (Ny, Nx), float64
    """
    from skimage.transform import resize

    img  = imread(image_path)
    gray = rgb2gray(img) if img.ndim == 3 else img.astype(float)

    thresh = threshold_otsu(gray)
    mask   = gray < thresh   # dark inclusion
    if invert:
        mask = ~mask

    if grid_shape is not None and gray.shape != grid_shape:
        mask = resize(mask.astype(float), grid_shape, order=0) > 0.5

    return _apply_tanh_interface(mask, interface_width)