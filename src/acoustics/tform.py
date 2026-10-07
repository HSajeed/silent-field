"""Radiating content of the Lighthill stress T_ij = u_i u_j, without the periodic-box artifact.

`komega` measures the cone share of S = d_i d_j T_ij on the box's own wavenumber grid
(R_S).  Two things are missing from that number in a periodic box:

  * nothing below the first box mode dk = 2 pi / L.  The box integral of T_ij (its
    k = 0 content) is not zero, and it is the compact term that dominates the far
    field at low frequency (review_2026-10-01.md section 1.3);
  * the cone is sampled only at grid modes, so at low Mach it holds a handful of them.

Sarkar & Hussaini (1993, ICASE 93-74, Eq. 2) treat the box contents as a finite volume
of turbulence embedded in a medium at rest.  In wavenumber space that is the continuous
Fourier transform of the box-truncated stress, T~_ij(k), which coincides with the DFT at
grid modes and is smooth between them.  Its double divergence is
S~(k) = -k_i k_j T~_ij(k), and the far field at frequency omega, direction n, depends on
S~ on the sonic circle |k| = omega / c0 only:  p ~ k^2 G(kr) n_i n_j T~_ij(k n).

`cone_share_refined` evaluates S~ on a grid P times finer than the box's (zero-padding by
P), restricted to the cone, normalised as a density so that P = 1 (untapered) is exactly
R_S.

A hard-edged box is itself an artifact here: truncating T_ij at the box edge leaks the
energetic convective-ridge content into small k through the box's sinc.  Measured on
data/turb2d_L4pi.npz at M = 0.1, 2 pi windows cut from the same flow gave 1.36x the
full-box share untapered and 1.11x with a Tukey(0.5) taper (inside the 1.4x
window-to-window scatter); the full-box value is 4.77e-9 untapered and 3.27 / 3.16 /
3.04e-9 for Tukey alpha = 0.25 / 0.5 / 1.  So the stress (and S, for the denominator) is
tapered in space by default, `taper_alpha = 0.5`.
`radiated_power` integrates |n_i n_j T~_ij(k n, omega)|^2 over the sonic circle -- the
far-field acoustic power, up to a constant -- for the Mach-scaling comparison.

Both use the temporal treatment of `komega.welch_komega_spectrum` (per-segment mean
removed, Hann window, negative frequencies folded), so their numbers sit on the same
footing as R and R_S.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import get_window
from scipy.signal.windows import tukey

from ..spectral import Grid, dealias


def _segments(nt: int, n_segments: int, overlap: float = 0.5):
    """Same segmentation as `komega.welch_komega_spectrum`."""
    if n_segments <= 1:
        return [(0, nt)]
    seg = int(nt / (1 + (n_segments - 1) * (1 - overlap)))
    step = max(1, int(seg * (1 - overlap)))
    return [(s, s + seg) for s in range(0, nt - seg + 1, step)]


def _stress_spectrum(u, v, grid: Grid, window: str = "hann", taper_alpha: float = 0.0):
    """Time transform of the dealiased stress of one segment.

    Returns (T_hat dict of (n_pos, n, n) complex, omega, fold, norm, s_total): `fold`
    doubles the interior positive frequencies, `norm` is komega's
    1 / (N_total^2 * <w^2>), so that |DFT|^2 * fold * norm is komega's power, and
    s_total is the total power of S = d_i d_j T_ij under the same treatment (Parseval).
    """
    nt = u.shape[0]
    w = get_window(window, nt, fftbins=True)
    n_pos = nt // 2 + 1
    tap = np.outer(tukey(grid.n, taper_alpha, sym=False), tukey(grid.n, taper_alpha, sym=False))
    kx, ky = grid.wavenumbers()
    kk = {"uu": kx * kx, "uv": 2.0 * kx * ky, "vv": ky * ky}
    out = {}
    s_hat = 0.0
    for key, (a, b) in {"uu": (u, u), "uv": (u, v), "vv": (v, v)}.items():
        t = np.stack([dealias(a[i].astype(np.float64) * b[i], grid) for i in range(nt)])
        t -= t.mean(axis=0, keepdims=True)
        s_hat = s_hat - kk[key] * np.fft.fft2(t)
        t *= w[:, None, None] * tap
        out[key] = np.fft.rfft(t, axis=0)
        del t
    s = np.real(np.fft.ifft2(s_hat)) * w[:, None, None] * tap
    del s_hat
    s_total = float((s**2).sum() / (nt * grid.n * grid.n * np.mean(w**2)))
    del s
    omega = np.abs(2.0 * np.pi * np.fft.fftfreq(nt, d=1.0)[:n_pos])  # rescaled by caller
    fold = np.ones(n_pos)
    fold[1 : n_pos - (nt % 2 == 0)] = 2.0
    norm = 1.0 / ((nt * grid.n * grid.n) ** 2 * np.mean(w**2))
    return out, omega, fold, norm, s_total


def _dft_matrix(k: np.ndarray, grid: Grid) -> np.ndarray:
    """Rows e^{-i k x_j} on the grid points, so (E @ f) is the unnormalised DFT at k."""
    return np.exp(-1j * np.outer(k, grid.x))


def cone_share_refined(
    u: np.ndarray,
    v: np.ndarray,
    grid: Grid,
    dt: float,
    c0s,
    pad: int = 4,
    n_segments: int = 4,
    taper_alpha: float = 0.5,
) -> dict:
    """Cone share of S~ = -k_i k_j T~_ij on a `pad`-times refined wavenumber grid.

    The denominator is the total power of S under the same window, segmentation and
    taper -- with taper_alpha = 0 that is `welch_komega_spectrum(S).power_resolved.sum()`,
    R_S's denominator.

    Returns {"share": [per c0], "share_sub_dk": [per c0]}: the second is the part with
    0 < |k| < dk, which the periodic grid cannot hold at all.  With pad = 1 the share is
    R_S to rounding (taper_alpha = 0), because the refined grid then *is* the box grid and
    S~ = S-hat there.
    """
    c0s = np.atleast_1d(np.asarray(c0s, dtype=float))
    dk = 2.0 * np.pi / grid.length
    dkf = dk / pad
    segs = _segments(u.shape[0], n_segments)
    num = np.zeros(c0s.size)
    sub = np.zeros(c0s.size)
    s_total = 0.0
    for lo, hi in segs:
        T, om_idx, fold, norm, s_seg = _stress_spectrum(u[lo:hi], v[lo:hi], grid,
                                                        taper_alpha=taper_alpha)
        s_total += s_seg
        omega = om_idx / dt
        # cone edge, clipped to the box's resolved band [-n/2, n/2) per axis
        m_cut = min(int(np.floor(omega.max() / c0s.min() / dkf)), grid.n * pad // 2)
        m = np.arange(-m_cut, min(m_cut + 1, grid.n * pad // 2))
        kf = m * dkf
        E = _dft_matrix(kf, grid)                              # (nk, n)
        KY, KX = np.meshgrid(kf, kf, indexing="ij")
        kmag = np.hypot(KX, KY)
        for i in range(1, omega.size):                        # omega = 0 never radiates
            if omega[i] < c0s.min() * dkf:
                continue
            # T~(ky, kx) = sum_y sum_x e^{-i ky y} T(y, x) e^{-i kx x}
            s = -(KX**2 * (E @ T["uu"][i] @ E.T)
                  + 2.0 * KX * KY * (E @ T["uv"][i] @ E.T)
                  + KY**2 * (E @ T["vv"][i] @ E.T))
            p = np.abs(s) ** 2 * fold[i] * norm / pad**2
            for j, c0 in enumerate(c0s):
                inside = (kmag > 0) & (kmag <= omega[i] / c0)
                num[j] += p[inside].sum()
                sub[j] += p[inside & (kmag < dk * (1 - 1e-9))].sum()
        del T
    return {"share": (num / s_total).tolist(),
            "share_sub_dk": (sub / s_total).tolist(),
            "pad": pad, "taper_alpha": taper_alpha}


def radiated_power(
    u: np.ndarray,
    v: np.ndarray,
    grid: Grid,
    dt: float,
    c0s,
    n_theta: int = 64,
    n_segments: int = 4,
    taper_alpha: float = 0.5,
) -> dict:
    """Far-field acoustic power of the stress, per c0, up to a c0-independent constant.

    P(c0) = sum_omega k^3 / c0 * mean_theta |n_i n_j T~_ij(k n, omega)|^2,  k = omega / c0.

    From the T-form far field p = -k^2 G(kr) n_i n_j T~_ij(k n) with
    |G|^2 = 1 / (8 pi k r) (2D Hankel asymptote) and intensity |p|^2 / (rho0 c0) through a
    circle of radius r.  Same window, segmentation and folding as the cone shares.
    Also returns the compact limit (T~ replaced by its k = 0 value, the box integral),
    which scales as c0^-4 exactly at fixed flow.  The stress is tapered as in
    `cone_share_refined`, so the compact term is the taper-weighted box integral.
    """
    c0s = np.atleast_1d(np.asarray(c0s, dtype=float))
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    nx, ny = np.cos(theta), np.sin(theta)
    segs = _segments(u.shape[0], n_segments)
    full = np.zeros(c0s.size)
    compact = np.zeros(c0s.size)
    x = grid.x
    for lo, hi in segs:
        T, om_idx, fold, norm, _ = _stress_spectrum(u[lo:hi], v[lo:hi], grid,
                                                    taper_alpha=taper_alpha)
        omega = om_idx / dt
        # box integrals (k = 0), shape (n_pos,)
        T0 = {key: a.sum(axis=(1, 2)) for key, a in T.items()}
        for j, c0 in enumerate(c0s):
            for i in range(1, omega.size):
                k = omega[i] / c0
                ey = np.exp(-1j * k * np.outer(ny, x))       # (n_theta, n)
                ex = np.exp(-1j * k * np.outer(nx, x))
                tt = {key: ((ey @ T[key][i]) * ex).sum(axis=1) for key in T}
                q = nx * nx * tt["uu"] + 2 * nx * ny * tt["uv"] + ny * ny * tt["vv"]
                q0 = nx * nx * T0["uu"][i] + 2 * nx * ny * T0["uv"][i] + ny * ny * T0["vv"][i]
                pref = k**3 / c0 * fold[i] * norm
                full[j] += pref * np.mean(np.abs(q) ** 2)
                compact[j] += pref * np.mean(np.abs(q0) ** 2)
        del T
    count = len(segs)
    return {"power": (full / count).tolist(), "compact": (compact / count).tolist()}
