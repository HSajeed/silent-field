"""The headline metric, and the loss-weighting argument made explicit.

`radiating_fraction` answers: of all the energy an L2 loss is summing over, what
share sits in the region of (k, omega) space that can radiate sound?

`spectral_loss_weight` makes the comparison between candidate objectives
concrete.  An L2 loss weights every mode equally (weight 1).  Sobolev/H^n
training weights by (1 + k^2)^n -- which *increases* the weight at high k and
therefore still assigns almost nothing to the low-k radiating cone.  That is the
Stage-4 counterexample in one function: the standard remedy for spectral bias
does not and cannot fix this.
"""
from __future__ import annotations

import numpy as np

from .komega import KOmegaSpectrum, sound_speed


def radiating_fraction(spectrum: KOmegaSpectrum, c0: float) -> float:
    return spectrum.radiating_fraction(c0)


def spectral_loss_weight(k: np.ndarray, kind: str = "l2", order: int = 1) -> np.ndarray:
    """Mode weighting implied by a training objective.

    'l2'      : uniform            -> weights by energy alone
    'sobolev' : (1 + k^2)^order    -> up-weights high k, ignores low-k acoustics
    'h1'      : (1 + k^2)          -> the common special case
    """
    k = np.asarray(k, dtype=float)
    if kind == "l2":
        return np.ones_like(k)
    if kind == "h1":
        return 1.0 + k**2
    if kind == "sobolev":
        return (1.0 + k**2) ** order
    raise ValueError(f"unknown weighting {kind!r}")


def effective_weight_in_cone(spectrum: KOmegaSpectrum, c0: float,
                             kind: str = "l2", order: int = 1) -> float:
    """Share of a weighted objective that the radiating cone receives.

    Run this for kind='l2' and kind='sobolev'. Both numbers being tiny is the
    quantitative core of the Stage-4 argument.
    """
    w = spectral_loss_weight(spectrum.k_bins, kind, order)[None, :]
    weighted = spectrum.power * w
    total = weighted.sum()
    if total <= 0:
        return 0.0
    return float(weighted[spectrum.radiating_mask(c0)].sum() / total)


def mach_sweep(spectrum: KOmegaSpectrum, u_ref: float, machs, kinds=("l2", "h1")):
    """Table of cone weight share vs Mach number for several objectives."""
    rows = []
    for m in machs:
        c0 = sound_speed(u_ref, m)
        row = {"mach": float(m), "c0": c0}
        for kind in kinds:
            row[kind] = effective_weight_in_cone(spectrum, c0, kind)
        rows.append(row)
    return rows
