"""The Sobolev convention, pinned to the objective the code actually trains.

There are two things called "the Sobolev weighting".  `losses.sobolev_loss`
minimises a *sum* of two separately normalised terms; analysing it as a pure
multiplicative (1 + k^2)^order reweighting measures a different objective.  These
tests hold the two files together and encode the corrected Stage-4 claim.

The fixture is a synthetic turbulent-like spectrum, not white noise, because the claim
is about a regime where the cone is starved: energy on a convective ridge with a
steep k^-4 tail, and the cone holding ~1e-8 of it.  White noise puts ~45% of its
energy in the cone and would not exercise the argument at all.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.spectral import Grid
from src.acoustics.komega import KOmegaSpectrum, komega_spectrum
from src.acoustics.radiating import (
    additive_sobolev_weight,
    effective_weight_in_cone,
    sobolev_halfplane_mean_weight,
    sobolev_lambda_sweep,
    weighted_cone_share,
)
from src.models.losses import _k_grid
from src.models.train import TrainConfig

GRID = Grid(n=64)
C0 = 14.0
U_CONV = 1.0


def _turbulent_spectrum(n_omega=400, omega_max=78.0, slope=4.0, width=2.4):
    """E(|k|, omega) peaked on the convective ridge omega = U|k|, tail ~ k^-slope.

    Built directly rather than by FFT so the fixture is fast and its energy in the
    cone is a controlled quantity.  The |k| axis comes from a real spectrum though,
    because the argument depends on the *range* of |k| -- the mean of (1 + k^2) is set
    by the corner modes near 90, and a fixture whose k axis stops at n/2 would
    understate how far the cone sits below the weight.

    `width` is the ridge's omega-broadening, and it is what decides how much the cone
    is starved: the cone is reached only where the ridge leaks upward by (c0 - U)k, so
    a narrow ridge misses it entirely.  At 2.4 the cone holds 3.1e-8 of the energy,
    the regime the real record is in (7.6e-8).

    What the multiplicative reading does to that number is NOT asserted here: it
    depends on the spectrum's k-distribution rather than on the code, and this
    fixture's is not the real one (it gives ~2.3x where the real record gives ~16x).
    That contrast is measured in `scripts/06_sobolev_argument.py`.
    """
    rng = np.random.default_rng(0)
    template = komega_spectrum(rng.standard_normal((128, GRID.n, GRID.n)), GRID, dt=0.05)
    k = template.k_resolved
    omega = np.linspace(0.0, omega_max, n_omega)
    kk, ww = np.meshgrid(k, omega, indexing="xy")
    ridge = np.exp(-0.5 * ((ww - U_CONV * kk) / width) ** 2)
    power = np.zeros_like(ridge)
    power[:, 1:] = ridge[:, 1:] * kk[:, 1:] ** (-slope)
    return KOmegaSpectrum(
        power=power,
        k_bins=template.k_bins,
        omega=omega,
        k_resolved=k,
        power_resolved=power,
    )


def _noise_spectrum(seed=0, nt=512):
    rng = np.random.default_rng(seed)
    return komega_spectrum(rng.standard_normal((nt, 64, 64)), GRID, dt=0.05)


# ------------------------------------------------------------ weight vs losses.py
def test_halfplane_mean_weight_matches_losses_k_grid():
    """The mean the analysis divides by is the mean the training loss divides by.

    `_k_grid` builds its wavenumbers in torch's default float32, so exact agreement
    is not available; float32 relative precision is the honest bound.
    """
    for order in (1, 2):
        expected = float(((1.0 + _k_grid(64, 64, "cpu", GRID.length)) ** order).mean())
        assert abs(sobolev_halfplane_mean_weight(GRID, order) / expected - 1.0) < 1e-6


def test_additive_weight_is_one_plus_scaled_sobolev_weight():
    """Elementwise against the torch weight `sobolev_loss` builds, and against the
    N_cells / N_half ratio its two normalisations imply.  This is the test that stops
    `radiating.py` and `losses.py` drifting apart."""
    order, lam = 1, 1e-2
    k_sq = _k_grid(GRID.n, GRID.n, "cpu", GRID.length).numpy()
    k_half = np.sqrt(k_sq)                       # every rfft2 half-plane mode
    w_torch = (1.0 + k_sq) ** order
    w_torch = w_torch / w_torch.mean()           # losses.sobolev_loss normalises
    ratio = GRID.n / (GRID.n // 2 + 1)           # N_cells / N_half

    # compare mode by mode, at the |k| values the half-plane actually contains
    ours = additive_sobolev_weight(k_half, GRID, lam, order).ravel()
    expected = 1.0 + lam * ratio * w_torch.ravel()
    assert np.allclose(ours, expected, rtol=1e-5)
    assert abs(ours.min() - 1.0) < 1e-3, "weight must be ~1 at low k, not ~0"


# ------------------------------------------------------------------- the two limits
def test_additive_tends_to_l2_as_lambda_vanishes():
    s = _turbulent_spectrum()
    l2 = effective_weight_in_cone(s, C0, "l2", grid=GRID, convention="additive")
    tiny = effective_weight_in_cone(
        s, C0, "sobolev", lam=1e-9, grid=GRID, convention="additive"
    )
    assert abs(tiny / l2 - 1.0) < 1e-3


def test_additive_tends_to_multiplicative_as_lambda_grows():
    s = _turbulent_spectrum()
    huge = effective_weight_in_cone(
        s, C0, "sobolev", lam=1e12, grid=GRID, convention="additive"
    )
    h1 = effective_weight_in_cone(s, C0, "h1", convention="multiplicative")
    assert abs(huge / h1 - 1.0) < 1e-3


# ------------------------------------------------------------------ the paper claim
def test_cone_is_starved_of_energy_in_the_fixture():
    """Guard on the fixture itself: if this stops holding, the claim test below is
    measuring nothing."""
    s = _turbulent_spectrum()
    assert 1e-9 < s.radiating_fraction(C0) < 1e-6


def test_default_lambda_leaves_cone_share_unchanged():
    """The corrected Stage-4 claim, as an assertion: at the weight the code actually
    trains with, Sobolev training does not change how much of the objective reaches
    the radiating cone.  Analysing the same trained objective as a pure
    multiplicative (1 + k^2) reweighting instead is what produced the old "ten times
    worse" number, and it is an artifact of that reading, not a property of training."""
    s = _turbulent_spectrum()
    l2 = effective_weight_in_cone(s, C0, "l2", grid=GRID, convention="additive")
    trained = effective_weight_in_cone(
        s, C0, "sobolev", lam=TrainConfig().sobolev_weight, grid=GRID,
        convention="additive",
    )
    assert abs(trained / l2 - 1.0) < 0.01

    # and the weighting is inert for the stated reason: the cone sits where the
    # Sobolev term is far below the leading 1, so the sum is the L2 weight
    w = additive_sobolev_weight(s.k_resolved, GRID, TrainConfig().sobolev_weight)
    mask = s.radiating_mask_resolved(C0)
    in_cone = np.broadcast_to(w, mask.shape)[mask]
    assert np.abs(in_cone - 1.0).max() < 0.01, "the leading 1 is the whole point"


def test_sweep_is_monotone_and_reports_both_asymptotes():
    s = _turbulent_spectrum()
    rows = sobolev_lambda_sweep(s, C0, GRID, lambdas=(1e-3, 1e-1, 1e1, 1e3))
    assert rows[0]["note"] == "l2 baseline"
    assert "not the trained objective" in rows[1]["note"]
    shares = [r["cone_share"] for r in rows[2:]]
    assert all(a >= b for a, b in zip(shares, shares[1:])), shares
    assert all(r["cone_share"] < 1.0 for r in rows)


# --------------------------------------------------------------------- primitives
def test_weighted_cone_share_is_invariant_to_rescaling():
    s = _turbulent_spectrum()
    mask = s.radiating_mask_resolved(C0)
    w = additive_sobolev_weight(s.k_resolved, GRID, 1e-2)
    a = weighted_cone_share(s.power_resolved, w, mask)
    b = weighted_cone_share(s.power_resolved, 1e6 * w, mask)
    assert abs(a / b - 1.0) < 1e-12


def test_multiplicative_share_is_invariant_to_bin_grid():
    """The k-4 dependent answer that the pre-fix binned mask produced must not be
    reachable any more: the resolved grid and the binned grid differ in a way that
    matters, so both conventions go through the resolved one."""
    s = _noise_spectrum()
    assert s.has_resolved()
    binned = float(s.power[s.radiating_mask(C0)].sum() / s.power.sum())
    resolved = effective_weight_in_cone(s, C0, "l2", convention="multiplicative")
    assert binned > resolved
    assert abs(resolved - s.radiating_fraction(C0)) < 1e-15


def test_effective_weight_requires_grid_for_additive():
    s = _turbulent_spectrum()
    try:
        effective_weight_in_cone(s, C0, "sobolev", convention="additive")
    except ValueError as e:
        assert "needs the Grid" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_script_04_lambda_default_matches_train_config():
    """`scripts/04` must stay torch-free, so it spells the default lambda out instead
    of importing it.  This is the test that stops the copy going stale."""
    import re
    from pathlib import Path as _P

    src = (_P(__file__).resolve().parents[1] / "scripts" / "04_komega_analysis.py").read_text()
    m = re.search(r"^DEFAULT_SOBOLEV_LAMBDA\s*=\s*([0-9eE.+-]+)", src, re.M)
    assert m, "04 no longer defines DEFAULT_SOBOLEV_LAMBDA"
    assert float(m.group(1)) == TrainConfig().sobolev_weight
