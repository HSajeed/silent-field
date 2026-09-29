"""The error triple.

The rhetorical force of this project is entirely in a contrast, so these are
always reported together and never one at a time:

    field error        ~  1%      (the number papers report)
    source error       ~  100%+   (the acoustic source)
    far-field error    ~  100%+   (the radiated sound)
"""
from __future__ import annotations

import numpy as np

from ..spectral import Grid
from ..acoustics.lighthill import lighthill_source


def relative_l2(pred, truth):
    denom = np.sqrt(np.mean(truth ** 2))
    if denom <= 0:
        return np.nan
    return float(np.sqrt(np.mean((pred - truth) ** 2)) / denom)


def field_error(u_p, v_p, u_t, v_t):
    num = np.mean((u_p - u_t) ** 2 + (v_p - v_t) ** 2)
    den = np.mean(u_t ** 2 + v_t ** 2)
    return float(np.sqrt(num / den)) if den > 0 else np.nan


def source_error(u_p, v_p, u_t, v_t, grid: Grid, rho0=1.0):
    """Relative error in the Lighthill source.

    The prediction uses the 'divergence' form deliberately: the gradient form
    assumes incompressibility and would silently discard the surrogate's
    spurious dilatation, which is part of the error we want counted.
    """
    s_t = lighthill_source(u_t, v_t, grid, rho0, form="divergence")
    s_p = lighthill_source(u_p, v_p, grid, rho0, form="divergence")
    return relative_l2(s_p, s_t)


def error_triple(u_p, v_p, u_t, v_t, grid: Grid, rho0=1.0):
    return {
        "field": field_error(u_p, v_p, u_t, v_t),
        "source": source_error(u_p, v_p, u_t, v_t, grid, rho0),
    }
