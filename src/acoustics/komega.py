"""Wavenumber-frequency analysis: the (k, omega) plane and the radiating cone.

This is where the argument lives.

A disturbance with wavevector k and frequency omega has phase speed omega/|k|.
It can radiate to the acoustic far field only if that phase speed is at least
the speed of sound:

    omega / |k|  >=  c0        <=>        |k|  <=  omega / c0.

That inequality carves a cone out of the (k, omega) plane -- the *radiating
cone*, or equivalently the region above the sonic line omega = c0 |k|.

Hydrodynamic turbulence does not live there.  It lives on the convective ridge
omega ~ U |k|, whose slope is smaller than the sonic line's by exactly the Mach
number.  At M = 0.1 the two regions are an order of magnitude apart in slope and
the overlap carries a vanishing fraction of the energy.

An L2 training loss weights every (k, omega) mode by its energy.  So the loss
weight assigned to the only region that makes sound is the number computed by
`radiating_fraction` below -- of order 1e-6 at M = 0.1.  That is the headline.

Note that c0 never enters the simulation.  The flow is incompressible; the Mach
number is an *analysis* parameter that tilts the sonic line.  One dataset
therefore supports a whole Mach sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import get_window

from ..spectral import Grid


# ------------------------------------------------------------------ containers
@dataclass
class KOmegaSpectrum:
    """Energy density on the (|k|, omega) plane.

    power   : (n_omega, n_k)  non-negative energy density, already radially binned
    k_bins  : (n_k,)          bin centres of |k|, angular wavenumber
    omega   : (n_omega,)      angular frequency, non-negative half only
    """

    power: np.ndarray
    k_bins: np.ndarray
    omega: np.ndarray

    def sonic_line(self, c0: float) -> np.ndarray:
        """omega = c0 |k| evaluated at the bin centres, for overlaying on a plot."""
        return c0 * self.k_bins

    def convective_ridge(self, u_conv: float) -> np.ndarray:
        """omega = U |k|, where the turbulence actually lives."""
        return u_conv * self.k_bins

    def radiating_mask(self, c0: float) -> np.ndarray:
        """Boolean (n_omega, n_k) mask of the region that can radiate."""
        kk, ww = np.meshgrid(self.k_bins, self.omega, indexing="xy")
        return ww >= c0 * kk

    def radiating_fraction(self, c0: float) -> float:
        """THE metric: fraction of total spectral energy inside the radiating cone.

        This is literally the share of an L2 objective that the acoustically
        decisive subspace receives.
        """
        total = self.power.sum()
        if total <= 0.0:
            return 0.0
        return float(self.power[self.radiating_mask(c0)].sum() / total)


# ------------------------------------------------------------------- estimation
def komega_spectrum(
    field: np.ndarray,
    grid: Grid,
    dt: float,
    window: str = "hann",
    n_k_bins: int | None = None,
    detrend_mean: bool = True,
) -> KOmegaSpectrum:
    """Estimate the (|k|, omega) energy spectrum of a space-time field.

    field : (nt, ny, nx) real, or (nt, ny, nx, n_comp) for a vector field, in
            which case component powers are summed.
    dt    : sampling interval of the time axis.

    A window in time is mandatory in practice.  Rectangular windowing leaks
    energy from the enormous convective ridge straight across the (k, omega)
    plane and into the radiating cone, which would manufacture the very signal
    this project is trying to measure.  The default Hann window is the minimum
    defensible choice.  Use `welch_komega_spectrum` for a lower-variance
    estimate.
    """
    arr = np.asarray(field)
    if arr.ndim == 3:
        arr = arr[..., None]
    if arr.ndim != 4:
        raise ValueError(f"expected (nt, ny, nx[, ncomp]), got {arr.shape}")

    nt, ny, nx, ncomp = arr.shape
    if (ny, nx) != (grid.n, grid.n):
        raise ValueError(f"field is {ny}x{nx} but grid.n is {grid.n}")

    arr = arr.astype(np.float64, copy=True)
    if detrend_mean:
        arr -= arr.mean(axis=0, keepdims=True)

    # --- window in time, and correct the amplitude for the window's power loss
    w = get_window(window, nt, fftbins=True).astype(np.float64)
    w_power = np.mean(w**2)
    arr *= w[:, None, None, None]

    # --- 3D transform: time then space
    spec = np.fft.fft(arr, axis=0)
    spec = np.fft.fft2(spec, axes=(1, 2))

    power = (np.abs(spec) ** 2).sum(axis=-1)  # (nt, ny, nx), summed over components
    # Parseval: sum |FFT|^2 = N_total * sum |x|^2, so dividing by N_total^2 makes
    # the spectrum sum to the field variance.  w_power undoes the window's
    # amplitude loss.  Getting this exponent wrong inflates every reported
    # spectral energy by N_total and is invisible in a log-scale plot.
    n_total = nt * ny * nx
    power /= (n_total**2) * w_power

    # --- keep the non-negative frequency half (real input => symmetric)
    n_pos = nt // 2 + 1
    omega_all = 2.0 * np.pi * np.fft.fftfreq(nt, d=dt)
    omega = np.abs(omega_all[:n_pos])
    omega[0] = 0.0
    power = power[:n_pos]
    if nt > 1:
        power[1 : n_pos - (nt % 2 == 0)] *= 2.0  # fold negative frequencies

    # --- radially bin in |k|
    k_mag = grid.k_mag
    dk = 2.0 * np.pi / grid.length
    if n_k_bins is None:
        n_k_bins = grid.n // 2
    # Bins are CENTRED on integer multiples of dk (edges at half-integers), not
    # starting at them.  With edges at b*dk the centres land at (b+0.5)*dk, so a
    # mode at |k| = 4 dk is reported at 4.5 dk -- a half-bin bias that tilts the
    # apparent phase speed and corrupts the radiating fraction.
    edges = (np.arange(n_k_bins + 1) - 0.5) * dk
    edges[0] = 0.0
    centres = np.arange(n_k_bins) * dk

    idx = np.digitize(k_mag.ravel(), edges) - 1
    valid = (idx >= 0) & (idx < n_k_bins)
    idx_v = idx[valid]

    flat = power.reshape(n_pos, -1)[:, valid]
    binned = np.zeros((n_pos, n_k_bins))
    for b in range(n_k_bins):
        sel = idx_v == b
        if sel.any():
            binned[:, b] = flat[:, sel].sum(axis=1)

    return KOmegaSpectrum(power=binned, k_bins=centres, omega=omega)


def welch_komega_spectrum(
    field: np.ndarray,
    grid: Grid,
    dt: float,
    n_segments: int = 4,
    overlap: float = 0.5,
    window: str = "hann",
    n_k_bins: int | None = None,
) -> KOmegaSpectrum:
    """Welch-averaged (k, omega) spectrum: lower variance, coarser in omega.

    Also the answer to a memory problem -- a full 3D FFT of a long record at
    256^2 is several GB, while segments are not.  Segmenting improves the
    estimate and fits in RAM at the same time, so prefer this for production
    figures.
    """
    arr = np.asarray(field)
    nt = arr.shape[0]
    seg = int(nt / (1 + (n_segments - 1) * (1 - overlap)))
    step = max(1, int(seg * (1 - overlap)))

    acc = None
    count = 0
    for start in range(0, nt - seg + 1, step):
        s = komega_spectrum(
            arr[start : start + seg], grid, dt, window=window, n_k_bins=n_k_bins
        )
        acc = s.power if acc is None else acc + s.power
        count += 1
        last = s

    if acc is None:
        raise ValueError("record too short for the requested segmentation")
    return KOmegaSpectrum(power=acc / count, k_bins=last.k_bins, omega=last.omega)


# ------------------------------------------------------------------- convenience
def sound_speed(u_ref: float, mach: float) -> float:
    """c0 implied by a reference velocity and a chosen Mach number.

    The simulation is incompressible and knows nothing about c0; this is purely
    the analysis knob that sets the slope of the sonic line.
    """
    if mach <= 0.0:
        raise ValueError("mach must be positive")
    return u_ref / mach


def radiating_fraction_sweep(
    spectrum: KOmegaSpectrum, u_ref: float, machs
) -> dict[float, float]:
    """Radiating energy fraction across a range of Mach numbers.

    Theory says this should fall steeply as M decreases.  Reproducing that trend
    from one dataset is the cheapest available check that the measurement is
    behaving physically rather than reflecting windowing artefacts.
    """
    return {
        float(m): spectrum.radiating_fraction(sound_speed(u_ref, m)) for m in machs
    }
