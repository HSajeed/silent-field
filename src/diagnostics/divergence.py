"""Spurious dilatation in a surrogate's prediction.

The training data is incompressible, so the ground truth has zero divergence to
machine precision.  Nothing in a standard MSE objective enforces that on the
prediction, so a surrogate's output carries a dilatational component that is
pure error -- and dilatation is exactly the acoustic degree of freedom.

This is a second, independent channel of the same argument: the model is not
merely inaccurate about sound, it invents compressibility that the physics
forbids.
"""
from __future__ import annotations

import numpy as np

from ..spectral import Grid, divergence, project_solenoidal


def divergence_error(u, v, grid: Grid):
    """RMS of div(u), normalised by a velocity-gradient scale so it is
    dimensionless and comparable across datasets."""
    div = divergence(u, v, grid)
    scale = np.sqrt(np.mean(u ** 2 + v ** 2)) * (2.0 * np.pi / grid.length) + 1e-30
    return float(np.sqrt(np.mean(div ** 2)) / scale)


def helmholtz_split(u, v, grid: Grid):
    """Return (solenoidal, dilatational) parts of a vector field."""
    us, vs = project_solenoidal(u, v, grid)
    return (us, vs), (u - us, v - vs)


def dilatational_energy_fraction(u, v, grid: Grid):
    """Share of kinetic energy in the dilatational part.

    Zero for the ground truth; non-zero for any unconstrained prediction.
    """
    _, (ud, vd) = helmholtz_split(u, v, grid)
    total = np.mean(u ** 2 + v ** 2)
    if total <= 0:
        return 0.0
    return float(np.mean(ud ** 2 + vd ** 2) / total)
