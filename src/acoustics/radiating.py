"""The headline metric, and the loss-weighting argument made explicit.

`radiating_fraction` answers: of all the energy an L2 loss is summing over, what
share sits in the region of (k, omega) space that can radiate sound?

`spectral_loss_weight` makes the comparison between candidate objectives concrete,
and it has to answer a question that is easy to get wrong.  There are two different
things called "the Sobolev weighting", and they are not the same objective:

`convention="multiplicative"`  --  w = (1 + k^2)^order
    A pure reweighting of the same energy.  This is the natural way to *analyse* an
    objective, and it is what this module computed before 2026-09-30.

`convention="additive"`  --  w_eff = 1 + lambda * c * (1 + k^2)^order / <.>,  c = N_cells / N_half
    on the kx = 0 / Nyquist columns, N_cells / (2 N_half) elsewhere

    What `src/models/losses.py::sobolev_loss` actually minimises.  It is the sum of
    two separately normalised terms, `mean(diff**2)` over all cells plus
    `lambda * mean(w * |rfft2(diff)|**2)` over the rfft2 half-plane, so the per-mode
    weight is one *plus* a multiple of the Sobolev weight, with the multiple fixed by
    the ratio of the two cell counts (N_cells / N_half = n / (n//2 + 1)) and by whether
    the rfft2 coefficient stands for one full-plane mode or a conjugate pair.

The distinction is not pedantry.  The pure multiplicative reading is 0.053 of the L2
cone share (~19x below); the additive reading is >= 1 everywhere (w_eff >= 1) and ~1 in
the cone, so Sobolev weighting can never raise the cone share.  At the default
lam = 1e-2 the Sobolev term is a negligible part of the whole objective (~0.01%: the
cone share moves by 1.3e-4 on data/turb2d.npz at M = 0.1) -- it is idle everywhere,
not specifically in the cone.  Where lambda is large enough to
matter it lowers the share, to 0.075 of L2 (~13x) at lam = 1e6 (the pure-H1 value, 19x,
is the unreachable limit of the shell-weighted form).  See
`scripts/06_sobolev_argument.py` for the sweep: the cone is starved by the *geometry*
of the weighting, not by its exponent.  94% of (1 + k^2)^order's mass sits at high k,
where the cone has no area at all.

One honest caveat, stated rather than hidden.  The absolute prefactor relating the two
terms depends on how the (k, omega) plane is normalised -- the analysis folds the
frequency axis, the training loss does not have one.  The *shape* of the answer above
is unaffected (the additive weight is within a factor of two of the pure
multiplicative one for any plausible prefactor, and the cone sits where w ~ 1e-3 of its
own mean either way), which is exactly why the result is published as a lambda sweep
rather than a single ratio.
"""
from __future__ import annotations

import numpy as np

from ..spectral import Grid
from .komega import KOmegaSpectrum, sound_speed


def radiating_fraction(spectrum: KOmegaSpectrum, c0: float) -> float:
    return spectrum.radiating_fraction(c0)


def weighted_cone_share(
    power: np.ndarray, weight: np.ndarray, mask: np.ndarray
) -> float:
    """Share of a weighted objective that the radiating cone receives.

    Deliberately c0-free and shape-agnostic so the arithmetic can be tested against
    `src/models/losses.py` directly, without a spectrum in the way.  The share is
    invariant to any overall rescaling of `weight`, so the only normalisation that
    matters is the *relative* one -- which for the additive convention is the whole
    point.
    """
    power = np.asarray(power, dtype=float)
    weight = np.asarray(weight, dtype=float)
    weighted = power * weight
    total = weighted.sum()
    if total <= 0:
        return 0.0
    return float(weighted[mask].sum() / total)


def spectral_loss_weight(k: np.ndarray, kind: str = "l2", order: int = 1) -> np.ndarray:
    """Pure mode weighting implied by a training objective.

    'l2'      : uniform            -> weights by energy alone
    'h1'      : (1 + k^2)          -> the common special case
    'sobolev' : (1 + k^2)^order    -> up-weights high k, ignores low-k acoustics
    """
    k = np.asarray(k, dtype=float)
    if kind == "l2":
        return np.ones_like(k)
    if kind == "h1":
        return 1.0 + k**2
    if kind == "sobolev":
        return (1.0 + k**2) ** order
    raise ValueError(f"unknown weighting {kind!r}")


def sobolev_halfplane_mean_weight(grid: Grid, order: int = 1) -> float:
    """Mean of (1 + k^2)^order over the rfft2 half-plane.

    Reproduces `losses._k_grid(...).mean()` in closed form, so the analysis does not
    import torch and cannot silently drift from it.  Note that `_k_grid` builds its
    wavenumbers with torch's default float32, so the two agree to float32 precision
    (~1e-7 relative) rather than exactly; `tests/test_radiating.py` asserts that bound.
    """
    n, length = grid.n, grid.length
    kx = 2.0 * np.pi * np.fft.rfftfreq(n, d=length / n)
    ky = 2.0 * np.pi * np.fft.fftfreq(n, d=length / n)
    k_sq = ky[:, None] ** 2 + kx[None, :] ** 2
    return float(((1.0 + k_sq) ** order).mean())


def shell_axis_fraction(k: np.ndarray, grid: Grid) -> np.ndarray:
    """Fraction of full-grid modes at each |k| lying on the rfft2 kx = 0 / Nyquist columns.

    Keyed on the integer shell m^2 = rint((k / dk)^2), dk = 2 pi / L, counted over the
    whole n x n FFT grid.  Independent of how the caller groups |k| values.
    """
    n = grid.n
    dk = 2.0 * np.pi / grid.length
    idx = np.fft.fftfreq(n, d=1.0 / n).astype(int)
    m2 = idx[:, None] ** 2 + idx[None, :] ** 2
    on_axis = np.broadcast_to(np.isin(idx, (0, -n // 2))[None, :], m2.shape)
    total = np.bincount(m2.ravel())
    axis = np.bincount(m2.ravel(), weights=on_axis.ravel())
    frac = np.divide(axis, total, out=np.zeros(total.size), where=total > 0)
    shell = np.rint((np.asarray(k, dtype=float) / dk) ** 2).astype(int)
    return frac[np.clip(shell, 0, total.size - 1)]


def additive_sobolev_weight(
    k: np.ndarray, grid: Grid, lam: float, order: int = 1, axis_fraction=0.0
) -> np.ndarray:
    """Per-full-plane-mode weight of the objective `losses.sobolev_loss` minimises.

    `1 + lam * c * (1 + k^2)^order / <weight>` with the two normalisations taken from
    that function:

      * the Sobolev weight is divided by its mean over the rfft2 half-plane;
      * the term is added to `mean(diff**2)`, a mean over all N_cells = n^2 cells,
        while the Sobolev term is a mean over N_half = n (n//2 + 1) coefficients.

    A half-plane coefficient on the kx = 0 or kx = n/2 columns is one full-plane mode
    and gets c = N_cells / N_half; every other one stands for the pair (+k, -k), so
    per mode c = N_cells / (2 N_half).  On a radial shell,
    c = N_cells / (2 N_half) * (1 + axis_fraction), with `axis_fraction` from
    `shell_axis_fraction`; shell-averaging is exact for the share when energy is
    isotropic within a shell.

    The leading 1 is the point.  A pure Sobolev weight is 1e-3 of its own mean
    everywhere inside the radiating cone, so at any usable lambda the sum is the 1.
    """
    k = np.asarray(k, dtype=float)
    ratio = grid.n / (grid.n // 2 + 1)
    w = (1.0 + k**2) ** order
    w = w / sobolev_halfplane_mean_weight(grid, order)
    return 1.0 + lam * 0.5 * ratio * (1.0 + axis_fraction) * w


def effective_weight_in_cone(
    spectrum: KOmegaSpectrum,
    c0: float,
    kind: str = "l2",
    order: int = 1,
    lam: float = 1e-2,
    convention: str = "multiplicative",
    grid: Grid | None = None,
) -> float:
    """Share of a weighted objective that the radiating cone receives.

    Evaluated on the spectrum's *resolved* radial grid with the exact cone mask, so
    the result is the metric and not a binning artifact.  `spectrum.power` /
    `radiating_mask` remain the display path.

    The archived analyses under `runs/*/analysis.py` contain inline arithmetic on the
    binned grid that predates this fix; they will not reproduce these numbers, and are
    kept as historical records rather than re-run.

    Run this for kind='l2' and convention='additive'.  Both numbers being tiny is the
    quantitative core of the Stage-4 argument.
    """
    if not spectrum.has_resolved():
        raise ValueError(
            "spectrum has no resolved radial grid; rebuild it with komega_spectrum"
        )
    k = spectrum.k_resolved
    if convention == "multiplicative":
        w = spectral_loss_weight(k, kind, order)[None, :]
    elif convention == "additive":
        if grid is None:
            raise ValueError("the additive convention needs the Grid used to build the spectrum")
        # kind still selects the base objective: 'l2' has no Sobolev term at all, so
        # in the additive convention its weight is flat and the result is the L2 share.
        base = (
            np.ones_like(k) if kind == "l2"
            else additive_sobolev_weight(k, grid, lam, order, shell_axis_fraction(k, grid))
        )
        w = base[None, :]
    else:
        raise ValueError(f"unknown convention {convention!r}")
    return weighted_cone_share(spectrum.power_resolved, w, spectrum.radiating_mask_resolved(c0))


def sobolev_lambda_sweep(
    spectrum: KOmegaSpectrum,
    c0: float,
    grid: Grid,
    order: int = 1,
    lambdas=(1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e6),
) -> list[dict]:
    """Cone share of the trained Sobolev objective as a function of lambda.

    The corrected Stage-4 result: the share is flat at its L2 value across the whole
    range of lambda that Sobolev training is actually used at, and only bends toward
    the pure H1 weighting five orders of magnitude above the default `lam = 1e-2`.
    Reporting the sweep rather than one number is deliberate -- see the module
    docstring on prefactors.
    """
    k = spectrum.k_resolved
    axis_frac = shell_axis_fraction(k, grid)
    mask = spectrum.radiating_mask_resolved(c0)
    l2 = effective_weight_in_cone(spectrum, c0, "l2", convention="additive", grid=grid)
    h1 = effective_weight_in_cone(
        spectrum, c0, "h1", order=order, convention="multiplicative"
    )
    rows = []
    for lam in lambdas:
        w = additive_sobolev_weight(k, grid, lam, order, axis_frac)[None, :]
        share = weighted_cone_share(spectrum.power_resolved, w, mask)
        rows.append({
            "lambda": float(lam),
            "cone_share": share,
            "ratio_to_l2": float(share / l2) if l2 > 0 else float("inf"),
        })
    return [{"lambda": None, "cone_share": l2, "ratio_to_l2": 1.0, "note": "l2 baseline"},
            {"lambda": None, "cone_share": h1, "ratio_to_l2": float(h1 / l2) if l2 > 0 else float("inf"),
             "note": "h1 multiplicative, not the trained objective"},
            *rows]


def mach_sweep(spectrum: KOmegaSpectrum, u_ref: float, machs, kinds=("l2", "h1"),
               grid: Grid | None = None):
    """Table of cone weight share vs Mach number for several objectives."""
    rows = []
    for m in machs:
        c0 = sound_speed(u_ref, m)
        row = {"mach": float(m), "c0": c0}
        for kind in kinds:
            row[kind] = effective_weight_in_cone(
                spectrum, c0, kind, grid=grid, convention="multiplicative"
            )
        rows.append(row)
    return rows
