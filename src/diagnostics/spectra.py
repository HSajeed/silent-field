"""Radial energy spectra and inertial-range slope fitting."""
from __future__ import annotations

import numpy as np

from ..spectral import Grid, fft2


def energy_spectrum(u, v, grid: Grid):
    """Radially binned kinetic energy spectrum E(k), with sum E(k) = <|u|^2>/2."""
    n = grid.n
    u_hat, v_hat = fft2(u) / (n * n), fft2(v) / (n * n)
    e_density = 0.5 * (np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2)

    dk = 2.0 * np.pi / grid.length
    n_bins = n // 2
    edges = (np.arange(n_bins + 1) - 0.5) * dk
    edges[0] = 0.0
    centres = np.arange(n_bins) * dk

    idx = np.digitize(grid.k_mag.ravel(), edges) - 1
    flat = e_density.ravel()
    out = np.zeros(n_bins)
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            out[b] = flat[sel].sum()
    return centres, out


def time_averaged_energy_spectrum(u, v, grid: Grid, stride: int = 16):
    """Mean radial energy spectrum over every `stride`-th snapshot.

    Returns (k, E, n_used) where E is the average of energy_spectrum(u[i], v[i], grid)
    over the selected snapshots and n_used is the number of snapshots averaged.
    """
    idxs = np.arange(0, u.shape[0], stride)
    k = None
    acc = None
    for i in idxs:
        ki, ei = energy_spectrum(u[i], v[i], grid)
        if k is None:
            k = ki
            acc = np.zeros_like(ei)
        acc += ei
    acc /= len(idxs)
    return k, acc, int(len(idxs))


def fit_slope(k, e, k_lo, k_hi):
    """Least-squares slope of log E vs log k over an inertial range."""
    m = (k >= k_lo) & (k <= k_hi) & (e > 0)
    if m.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log(k[m]), np.log(e[m]), 1)[0])


def enstrophy(u, v, grid: Grid):
    from ..spectral import vorticity_from_velocity
    om = vorticity_from_velocity(u, v, grid)
    return 0.5 * float(np.mean(om ** 2))


def kinetic_energy(u, v):
    return 0.5 * float(np.mean(u ** 2 + v ** 2))
