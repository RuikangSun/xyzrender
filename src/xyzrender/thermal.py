r"""Thermal ellipsoid (ORTEP-style) geometry helpers.

CIF files store anisotropic displacement parameters (ADPs) as six ``Uij``
values in Å².  xyzrender keeps the tensor that results from the CIF
orthogonalisation convention on each graph node (``adp``), and converts it to
a Cartesian shape matrix at render time using the current lattice.  Keeping
the tensor in the crystallographic basis means that rotating a crystal (for
example with ``--axis`` or ``--gif-rot``) rotates the ellipsoids with it.

The ellipsoid drawn for probability *p* is the surface on which

.. math::

    (x - x_0)^T U^{-1} (x - x_0) = \chi^2_3(p)

so its semi-axes are ``sqrt(chi2_3(p)) * sqrt(eigenvalues(U))``.  ORTEP calls
the leading factor ``SCAL2``; it is 1.538 for the conventional 50 % level.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "adp_to_cartesian",
    "ellipse_from_adp",
    "probability_scale",
    "support_radius",
]


def probability_scale(probability: float) -> float:
    """Return the ORTEP ``SCAL2`` factor enclosing *probability* percent.

    The factor is ``sqrt(chi2.ppf(probability / 100, 3))``, computed here with
    a short bisection on the chi-square CDF so that scipy is not required.

    Parameters
    ----------
    probability:
        Probability enclosed by the ellipsoid, in percent.  Must be strictly
        between 0 and 100.

    Returns
    -------
    float
        Multiplier applied to the RMS displacement along each principal axis.
        The 50 % value is 1.5382 (ORTEP's primer constant 1.54).
    """
    if not 0.0 < probability < 100.0:
        msg = f"thermal ellipsoid probability must be between 0 and 100 (got {probability})"
        raise ValueError(msg)

    target = probability / 100.0
    # For three degrees of freedom the CDF has a closed form; invert it by
    # bisection.  x = 100 is far beyond the 99.99 % quantile (21.1).
    lo, hi = 0.0, 100.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        cdf = math.erf(math.sqrt(mid / 2.0)) - math.sqrt(2.0 * mid / math.pi) * math.exp(-mid / 2.0)
        if cdf < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(0.5 * (lo + hi))


def adp_to_cartesian(adp: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Convert a stored CIF ADP tensor to a Cartesian shape matrix (Å²).

    Parameters
    ----------
    adp:
        ``(3, 3)`` symmetric tensor in the basis stored by the CIF parser.
    lattice:
        ``(3, 3)`` cell matrix whose rows are the lattice vectors **a**, **b**,
        **c** in Å.

    Returns
    -------
    numpy.ndarray
        ``(3, 3)`` Cartesian mean-square displacement tensor in Å².
    """
    lat = np.asarray(lattice, dtype=float)
    if lat.shape != (3, 3):
        msg = f"lattice must be a 3x3 matrix (got shape {lat.shape})"
        raise ValueError(msg)
    tensor = np.asarray(adp, dtype=float)
    if tensor.shape != (3, 3):
        msg = f"adp must be a 3x3 matrix (got shape {tensor.shape})"
        raise ValueError(msg)
    frac_to_cart = lat.T
    return frac_to_cart @ tensor @ frac_to_cart.T


def ellipse_from_adp(u_cart: np.ndarray, scale: float, pixel_scale: float) -> tuple[float, float, float]:
    """Return the projected silhouette of an ellipsoid as an ellipse.

    The orthographic shadow of an ellipsoid with Cartesian shape matrix *U* is
    an ellipse whose 2x2 covariance is simply the upper-left block ``U[:2,
    :2]`` (the z extent does not change the shadow outline).  Screen y points
    down, hence the flip matrix.

    Parameters
    ----------
    u_cart:
        ``(3, 3)`` Cartesian ADP tensor in Å².
    scale:
        Probability multiplier from :func:`probability_scale` (optionally
        combined with a user scale factor).
    pixel_scale:
        Projection scale in pixels per Å.

    Returns
    -------
    tuple of float
        ``(rx, ry, angle_deg)`` for an SVG ``<ellipse>``: semi-axes in pixels
        and a clockwise rotation angle in degrees.
    """
    tensor = np.asarray(u_cart, dtype=float)
    if tensor.shape != (3, 3):
        msg = f"u_cart must be a 3x3 matrix (got shape {tensor.shape})"
        raise ValueError(msg)

    covariance = (scale * scale) * tensor[:2, :2]
    flip = np.diag([1.0, -1.0])
    projected = flip @ covariance @ flip

    # eigh returns ascending eigenvalues; the major axis is last.
    values, vectors = np.linalg.eigh(projected)
    rx = math.sqrt(max(float(values[1]), 0.0)) * pixel_scale
    ry = math.sqrt(max(float(values[0]), 0.0)) * pixel_scale
    angle = math.degrees(math.atan2(float(vectors[1, 1]), float(vectors[0, 1])))
    return rx, ry, angle


def support_radius(u_cart: np.ndarray, direction: np.ndarray) -> float:
    """Return the ellipsoid radius along a unit *direction* (Å, before scaling).

    Used to trim a bond exactly at the ellipsoid surface rather than at a
    display-sphere radius.
    """
    tensor = np.asarray(u_cart, dtype=float)
    vec = np.asarray(direction, dtype=float)
    norm = math.sqrt(float(vec @ vec))
    if norm < 1e-12:
        return 0.0
    unit = vec / norm
    return math.sqrt(max(float(unit @ tensor @ unit), 0.0))
