"""The headline metric, and the loss-weighting argument made explicit.

`radiating_fraction` answers: of all the energy an L2 loss is summing over, what
share sits in the region of (k, omega) space that can radiate sound?

`spectral_loss_weight` makes the comparison between candidate objectives concrete,
and it has to answer a question that is easy to get wrong.  There are two different
things called "the Sobolev weighting", and they are not the same objective:

`convention="multiplicative"`  --  w = (1 + k^2)^order
    A pure reweighting of the same energy.  This is the natural way to *analyse* an
    objective, and it is what this module computed before 2026-09-30.

`convention="additive"`  --  w_eff = 1 + lambda * (N_cells / N_half) * (1 + k^2)^order
    What `src/models/losses.py::sobolev_loss` actually minimises.  It is the sum of
    two separately normalised terms, `mean(diff**2)` over all cells plus
    `lambda * mean(w * |rfft2(diff)|**2)` over the rfft2 half-plane, so the per-mode
    weight is one *plus* a multiple of the Sobolev weight, with the multiple fixed by
    the ratio of the two cell counts.  N_cells / N_half = n / (n//2 + 1).

The distinction is not pedantry.  The multiplicative reading is 16x below the L2 cone
share; the additive reading, at the weight the code actually trains with, is
1.00x below it -- Sobolev training at its default strength changes the share of the
objective reaching the radiating cone by 0.02%.  See `scripts/06_sobolev_argument.py`
for the sweep, which is the honest form of the claim: the cone is starved by the
*geometry* of the weighting, not by its exponent.  94% of (1 + k^2)^order's mass sits
at high k, where the cone has no area at all.

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


def additive_sobolev_weight(
    k: np.ndarray, grid: Grid, lam: float, order: int = 1
) -> np.ndarray:
    """Per-mode weight of the objective `losses.sobolev_loss` actually minimises.

    `1 + lam * (N_cells / N_half) * (1 + k^2)^order / <weight>`, with the two factors
    after lam coming straight out of that function:

      * `losses.sobolev_loss` divides the Sobolev weight by its own mean over the
        rfft2 half-plane, so the high-k up-weighting is scale-free;
      * it then adds that term to `mean(diff**2)`, which is a mean over *all* cells.
        The two terms therefore carry different per-mode mass, and the ratio is
        N_cells / N_half = n / (n//2 + 1).

    The leading 1 is the point.  A pure Sobolev weight is 1e-3 of its own mean
    everywhere inside the radiating cone, so at any usable lambda the sum is the 1.
    """
    k = np.asarray(k, dtype=float)
    ratio = grid.n / (grid.n // 2 + 1)
    w = (1.0 + k**2) ** order
    w = w / sobolev_halfplane_mean_weight(grid, order)
    return 1.0 + lam * ratio * w


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
        base = np.ones_like(k) if kind == "l2" else additive_sobolev_weight(k, grid, lam, order)
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
    mask = spectrum.radiating_mask_resolved(c0)
    l2 = effective_weight_in_cone(spectrum, c0, "l2", convention="additive", grid=grid)
    h1 = effective_weight_in_cone(
        spectrum, c0, "h1", order=order, convention="multiplicative"
    )
    rows = []
    for lam in lambdas:
        w = additive_sobolev_weight(k, grid, lam, order)[None, :]
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
