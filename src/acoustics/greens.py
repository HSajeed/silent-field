"""Far-field acoustic pressure from a Lighthill source distribution, in 2D.

The single most common error in two-dimensional computational aeroacoustics is
reaching for the three-dimensional free-space Green's function.  In 3D the
frequency-domain kernel is exp(-i k r) / (4 pi r) and the far field decays as
1/r.  In 2D it is a Hankel function,

    G(r; omega) = (-i/4) H_0^(2)(k r),      k = omega / c0,

whose far-field asymptote decays as 1/sqrt(r):

    G(r) ~ (-i/4) sqrt(2 / (pi k r)) exp(-i (k r - pi/4)).

A referee who works in aeroacoustics will check this in the first thirty
seconds, so it is worth being exactly right.

Sign convention: `source_spectrum` below builds spectra with `np.fft.rfft`,
whose implied synthesis is x(t) = sum_omega X_hat(omega) exp(+i omega t) (the
numpy/engineering convention, not the e^{-i omega t} physics convention used
in most textbooks).  Under that convention the wave equation
(1/c0^2 d_tt - grad^2) p = S becomes, per frequency, (grad^2 + k^2) p_hat =
-S_hat, and the causal, OUTGOING solution of (grad^2 + k^2) G = -delta is the
complex conjugate of the textbook e^{-i omega t} kernel above -- i.e. the
Hankel function of the SECOND kind, not the first.  Using H_0^(1) here (the
first thing you'd copy from a textbook written in the e^{-i omega t}
convention) silently produces an INCOMING wave once composed with
`np.fft.rfft`/`np.fft.irfft`; the far-field phase would then advance with r
instead of retreating from it.

Convention: pressure is reconstructed one frequency at a time,

    p'(x, omega) = integral over y of  G(x - y; omega) * S(y, omega) dy,

with S the Lighthill source already carrying its double divergence.  For noisy
data it is usually better to integrate the derivatives by parts onto the
Green's function instead; that variant is provided as `far_field_pressure_ibp`
and is the safer choice when S comes from a surrogate prediction.
"""

from __future__ import annotations

import numpy as np
from scipy.special import hankel2

from ..spectral import Grid


def greens_2d(k: float, r: np.ndarray, far_field: bool = False) -> np.ndarray:
    """Frequency-domain 2D free-space Green's function (outgoing, under the
    np.fft.rfft / e^{+i omega t} convention used throughout this module).

    k : acoustic wavenumber omega / c0
    r : distance(s) from source to observer
    """
    r = np.asarray(r, dtype=np.float64)
    kr = k * np.maximum(r, 1e-12)
    if far_field:
        return (-0.25j) * np.sqrt(2.0 / (np.pi * kr)) * np.exp(-1j * (kr - 0.25 * np.pi))
    return (-0.25j) * hankel2(0, kr)


def observer_ring(radius: float, n_obs: int = 72, centre=(0.0, 0.0)) -> np.ndarray:
    """A ring of observer positions, for directivity plots. Returns (n_obs, 2)."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_obs, endpoint=False)
    cx, cy = centre
    return np.stack([cx + radius * np.cos(theta), cy + radius * np.sin(theta)], axis=1)


def source_spectrum(source: np.ndarray, dt: float, window: str = "hann"):
    """Time-Fourier transform of a source time series.

    source : (nt, ny, nx)  ->  (n_omega, ny, nx) complex, plus omega array.
    """
    from scipy.signal import get_window

    nt = source.shape[0]
    w = get_window(window, nt, fftbins=True)
    amp = w.sum()  # coherent gain: preserves the amplitude of a pure tone
    s_hat = np.fft.rfft(source * w[:, None, None], axis=0) / amp * 2.0
    omega = 2.0 * np.pi * np.fft.rfftfreq(nt, d=dt)
    return s_hat, omega


def far_field_pressure(
    source_hat: np.ndarray,
    omega: np.ndarray,
    grid: Grid,
    observers: np.ndarray,
    c0: float,
    far_field: bool = True,
    omega_indices=None,
    max_kernel_evals: float = 5e7,
) -> np.ndarray:
    """Radiate a frequency-domain source to a set of observers.

    source_hat : (n_omega, ny, nx) complex, from `source_spectrum`
    observers  : (n_obs, 2) positions, in the same length units as the grid
    returns    : (n_omega_sel, n_obs) complex acoustic pressure

    The source is treated as a compact distribution on the grid; the quadrature
    is a simple midpoint rule with weight dx^2, which is spectrally accurate for
    a smooth periodic integrand.

    With `omega_indices=None` every frequency is radiated to every observer
    from every grid point, which is n_omega * n_obs * n_src Hankel evaluations
    -- easily hundreds of millions for a routine call.  `max_kernel_evals`
    guards against that; pass `omega_indices` to select a frequency band, or
    raise the cap if you really mean it.
    """
    X, Y = grid.meshgrid()
    src_pts = np.stack([X.ravel(), Y.ravel()], axis=1)  # (n_src, 2)
    dA = grid.dx**2

    idx = list(range(len(omega))) if omega_indices is None else list(omega_indices)
    n_evals = len(idx) * observers.shape[0] * src_pts.shape[0]
    if n_evals > max_kernel_evals:
        raise ValueError(
            f"far_field_pressure would perform {n_evals:.3e} kernel evaluations "
            f"(n_freq={len(idx)} x n_obs={observers.shape[0]} x n_src={src_pts.shape[0]}), "
            f"exceeding max_kernel_evals={max_kernel_evals:.3e}. Pass `omega_indices` to "
            "select a frequency band, or raise `max_kernel_evals` if this is intentional."
        )
    out = np.zeros((len(idx), observers.shape[0]), dtype=np.complex128)

    for row, i in enumerate(idx):
        if omega[i] <= 0.0:
            continue
        k = omega[i] / c0
        s_flat = source_hat[i].ravel()
        for j, obs in enumerate(observers):
            r = np.hypot(obs[0] - src_pts[:, 0], obs[1] - src_pts[:, 1])
            out[row, j] = np.sum(greens_2d(k, r, far_field) * s_flat) * dA
    return out


def far_field_pressure_ibp(
    stress_hat: dict[str, np.ndarray],
    omega: np.ndarray,
    grid: Grid,
    observers: np.ndarray,
    c0: float,
    omega_indices=None,
    max_kernel_evals: float = 5e7,
) -> np.ndarray:
    """Far-field pressure with the double divergence moved onto the Green's
    function by parts, i.e.

        p'(x) = integral of  d2 G / dx_i dx_j  *  T_ij(y)  dy.

    Preferred when T_ij comes from a neural surrogate: differentiating a noisy
    predicted field twice amplifies its error by k^2 and confounds "the surrogate
    got the acoustics wrong" with "we differentiated noise".  Using this variant
    and still seeing a large error closes that objection.

    stress_hat : dict with keys 'uu', 'uv', 'vv', each (n_omega, ny, nx) complex,
                 already multiplied by rho0.

    See `far_field_pressure` for the meaning of `max_kernel_evals`.
    """
    X, Y = grid.meshgrid()
    sx, sy = X.ravel(), Y.ravel()
    dA = grid.dx**2

    idx = list(range(len(omega))) if omega_indices is None else list(omega_indices)
    n_evals = len(idx) * observers.shape[0] * sx.shape[0]
    if n_evals > max_kernel_evals:
        raise ValueError(
            f"far_field_pressure_ibp would perform {n_evals:.3e} kernel evaluations "
            f"(n_freq={len(idx)} x n_obs={observers.shape[0]} x n_src={sx.shape[0]}), "
            f"exceeding max_kernel_evals={max_kernel_evals:.3e}. Pass `omega_indices` to "
            "select a frequency band, or raise `max_kernel_evals` if this is intentional."
        )
    out = np.zeros((len(idx), observers.shape[0]), dtype=np.complex128)

    for row, i in enumerate(idx):
        if omega[i] <= 0.0:
            continue
        k = omega[i] / c0
        uu = stress_hat["uu"][i].ravel()
        uv = stress_hat["uv"][i].ravel()
        vv = stress_hat["vv"][i].ravel()

        for j, obs in enumerate(observers):
            dx = obs[0] - sx
            dy = obs[1] - sy
            r = np.maximum(np.hypot(dx, dy), 1e-12)
            kr = k * r

            # Far-field second derivatives of G: the leading term differentiates
            # the exp(-i k r) phase twice, giving (-ik)^2 n_i n_j G = -k^2 n_i n_j
            # G, with n the unit vector from source to observer -- the sign is
            # unchanged by the outgoing-vs-incoming convention since it enters
            # squared.  Curvature terms are O(1/(kr)) smaller.
            g = greens_2d(k, r, far_field=True)
            nx_, ny_ = dx / r, dy / r
            d2g_xx = -(k**2) * nx_ * nx_ * g
            d2g_xy = -(k**2) * nx_ * ny_ * g
            d2g_yy = -(k**2) * ny_ * ny_ * g

            out[row, j] = np.sum(d2g_xx * uu + 2.0 * d2g_xy * uv + d2g_yy * vv) * dA
    return out


def directivity(pressure: np.ndarray) -> np.ndarray:
    """|p'| around an observer ring, normalised to unit maximum."""
    mag = np.abs(pressure)
    peak = mag.max()
    return mag / peak if peak > 0 else mag
