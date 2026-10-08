"""Training objectives, including the Sobolev variant used in Stage 4.

The Stage-4 argument in code: `sobolev_loss` up-weights high wavenumbers by
(1 + k^2)^order.  It measurably improves small-scale fidelity.  It cannot help
the acoustics, because the radiating cone sits at LOW k where this weighting is
approximately 1 -- the same weight plain MSE already gave it.

If you later want to test a fix rather than diagnose the failure,
`cone_weighted_loss` is the obvious candidate and the natural continuation of
the project: weight by proximity to the sonic line instead of by |k|.  It needs
a time window to define omega, so it is a multi-step objective, not a one-step
one. `cone_window_loss` below is Experiment 5's implementation of exactly that.
"""
from __future__ import annotations

import torch


def mse_loss(pred, target):
    return torch.mean((pred - target) ** 2)


def _k_grid(h, w, device, length: float = 2 * 3.141592653589793):
    """Squared angular wavenumber k^2 = kx^2 + ky^2 on the rfft2 half-plane.

    `fftfreq`/`rfftfreq` return cycles-per-sample (mode indices) when d=1;
    converting to angular wavenumber requires k = 2*pi*m/length, matching the
    convention already used in `divergence_penalty`. Using the mode index
    directly (as if length were always 2*pi) silently mis-scales the weighting
    for any other domain size.
    """
    ky = torch.fft.fftfreq(h, d=length / h, device=device)[:, None] * 2 * 3.141592653589793
    kx = torch.fft.rfftfreq(w, d=length / w, device=device)[None, :] * 2 * 3.141592653589793
    return kx ** 2 + ky ** 2


def sobolev_loss(pred, target, order: int = 1, weight: float = 1.0,
                  length: float = 2 * 3.141592653589793):
    """MSE + H^order penalty, applied spectrally.

    The spectral weight (1 + k^2)^order is normalised by its own mean over the
    rfft2 half-plane before being applied. Without this, the weight grows with
    the grid resolution (k reaches ~pi*n/length at Nyquist), so at weight=1.0
    the penalty term can be ~1e3x the MSE term on a typical grid, making
    "MSE + H^order" effectively pure H^order and `weight` not a meaningful,
    resolution-independent knob. Normalising by the mean weight makes
    weight=1.0 give the penalty roughly the same overall scale as the MSE for
    spectrally white error (mean(|rfft2(diff, norm="ortho")|^2) ~= mean(diff^2)
    is itself Parseval-consistent, verified numerically), while preserving the
    *relative* up-weighting of high k relative to low k that is the whole
    point of the Stage-4 comparison.
    """
    diff = pred - target
    base = torch.mean(diff ** 2)
    d_ft = torch.fft.rfft2(diff, norm="ortho")
    k_sq = _k_grid(diff.shape[-2], diff.shape[-1], diff.device, length)
    w = (1.0 + k_sq) ** order
    w = w / w.mean()
    high = torch.mean(w * torch.abs(d_ft) ** 2)
    return base + weight * high


def h1_loss(pred, target, length: float = 2 * 3.141592653589793):
    """The standard relative H^1 loss, as practitioners use it (neuraloperator `H1Loss`):
    ||e||_H1 / ||target||_H1 per sample, averaged over the batch, with
    ||f||_H1^2 = sum_k (1 + |k|^2) |f_hat(k)|^2 -- i.e. ||f||^2 + ||grad f||^2 in Fourier space.

    Multiplicative (1 + k^2) weighting, unlike `sobolev_loss` (MSE + a small mean-normalised
    term; review_2026-10-01.md section 2.3).  Uses the full-plane fft2, so every mode is
    weighted once and there is no rfft2 kx = 0 column anisotropy.
    """
    h, w = pred.shape[-2:]
    ky = torch.fft.fftfreq(h, d=length / h, device=pred.device)[:, None] * 2 * 3.141592653589793
    kx = torch.fft.fftfreq(w, d=length / w, device=pred.device)[None, :] * 2 * 3.141592653589793
    weight = 1.0 + kx ** 2 + ky ** 2
    e = torch.fft.fft2(pred - target, norm="ortho")
    t = torch.fft.fft2(target, norm="ortho")
    dims = tuple(range(1, pred.ndim))
    num = (weight * torch.abs(e) ** 2).sum(dim=dims)
    den = (weight * torch.abs(t) ** 2).sum(dim=dims)
    return torch.mean(torch.sqrt(num / den))


def cone_window_loss(err, dt: float, length: float, c0: float):
    """Energy of `err` (B, W, 2, n, n, normalised units) inside the radiating cone
    omega >= c0*|k|, as a fraction of the total MSE scale.

    Hann-windowed (rescaled to unit mean square, so a stationary signal's energy is
    preserved) along the time axis, then a complex FFT over time and a full 2-D FFT
    over space, both `norm="ortho"`.  The cone mask is evaluated per exact (omega, k)
    mode -- same convention as `_k_grid`, but on the full fft2 plane since the cone
    is not kx=0-column symmetric the way `_k_grid`'s rfft2 half-plane is.  The k=0
    column is excluded (same reasoning as `radiating_mask_resolved` in
    src/acoustics/komega.py): omega >= c0*0 is trivially true and would count the
    mean mode as radiating.

    Returns sum(power * mask) / err.numel(), so a mask of all ones reproduces the
    plain MSE on `err` (up to windowing), making the term directly comparable to
    (and addable to) `mse_loss`.
    """
    b, w, c, h, wi = err.shape
    window = torch.hann_window(w, periodic=False, dtype=err.dtype, device=err.device)
    window = window / torch.sqrt(torch.mean(window ** 2))
    e = err * window.view(1, w, 1, 1, 1)
    e_ft = torch.fft.fft(e, dim=1, norm="ortho")
    e_ft = torch.fft.fft2(e_ft, dim=(-2, -1), norm="ortho")
    power = (e_ft.abs() ** 2).sum(dim=2)                     # (B, W, h, wi)

    omega = torch.fft.fftfreq(w, d=dt, device=err.device) * 2 * 3.141592653589793
    omega = omega.abs()
    ky = torch.fft.fftfreq(h, d=length / h, device=err.device)[:, None] * 2 * 3.141592653589793
    kx = torch.fft.fftfreq(wi, d=length / wi, device=err.device)[None, :] * 2 * 3.141592653589793
    k = torch.sqrt(kx ** 2 + ky ** 2)

    mask = (omega[:, None, None] >= c0 * k[None, :, :]) & (k[None, :, :] > 0.0)
    return (power * mask.unsqueeze(0)).sum() / err.numel()


def divergence_penalty(pred, length: float = 2 * 3.141592653589793):
    """Optional: penalise spurious dilatation directly.

    Not used in the main experiment -- the point is that the STANDARD recipe
    fails -- but useful as a control showing the failure is fixable in
    principle, which strengthens rather than weakens the diagnosis.
    """
    b, c, h, w = pred.shape
    assert c == 2, "expected 2 velocity channels"
    ky = torch.fft.fftfreq(h, d=length / h, device=pred.device)[:, None] * 2 * 3.141592653589793
    kx = torch.fft.rfftfreq(w, d=length / w, device=pred.device)[None, :] * 2 * 3.141592653589793
    u_ft = torch.fft.rfft2(pred[:, 0], norm="ortho")
    v_ft = torch.fft.rfft2(pred[:, 1], norm="ortho")
    div = 1j * (kx * u_ft + ky * v_ft)
    return torch.mean(torch.abs(div) ** 2)
