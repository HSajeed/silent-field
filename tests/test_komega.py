"""A planted travelling wave must land in the right (k, omega) bin, and the
spectrum must obey Parseval. Both were bugs in the first draft.

Plus the cone mask: it must be evaluated at the true |k| of every mode, not at the
bin centre it happens to have been binned into, and it must never count the k = 0
column as radiating."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.spectral import Grid
from src.acoustics.komega import komega_spectrum, sound_speed, KOmegaSpectrum

G = Grid(n=32)

def _wave(kx0, cphase, nt=512, dt=0.05):
    X, _ = G.meshgrid()
    return np.stack([np.cos(kx0*X - kx0*cphase*t) for t in np.arange(nt)*dt]), dt

def test_peak_lands_in_correct_bin():
    fld, dt = _wave(4.0, 1.3)
    s = komega_spectrum(fld, G, dt)
    i_om, i_k = np.unravel_index(np.argmax(s.power), s.power.shape)
    assert abs(s.k_bins[i_k] - 4.0) < 1e-9, "half-bin offset in radial binning"
    assert abs(s.omega[i_om] - 5.2) < 0.5

def test_parseval():
    fld, dt = _wave(4.0, 1.3)
    s = komega_spectrum(fld, G, dt)
    assert abs(s.power.sum() - fld.var()) / fld.var() < 0.01

def test_sonic_mask():
    """Phase speed 1.3: radiates when c0 < 1.3, silent when c0 > 1.3."""
    fld, dt = _wave(4.0, 1.3)
    s = komega_spectrum(fld, G, dt)
    assert s.radiating_fraction(1.0) > 0.9
    assert s.radiating_fraction(1.5) < 0.05

def test_sound_speed():
    assert abs(sound_speed(2.0, 0.1) - 20.0) < 1e-12


# ---------------------------------------------------------------- cone mask
def _two_modes_in_one_bin(power_resolved):
    """One radial bin (centre |k| = 1) holding two distinct resolved |k| values.

    On a square periodic grid a bin does collect several |k| values -- sqrt(21) and 5
    and sqrt(29) all land in the bin centred on 5 -- so this is the real situation, not
    a contrivance.  It is also the one that broke: the binned mask tests both modes
    against |k| = 1 and calls them both radiating.
    """
    return KOmegaSpectrum(
        power=np.array([[0.0, 0.0], [0.0, 1.0]]),
        k_bins=np.array([0.0, 1.0]),
        omega=np.array([0.0, 1.0]),
        k_resolved=np.array([0.0, 0.9, 1.1]),
        power_resolved=power_resolved,
    )


def test_mask_uses_true_k_not_bin_centre():
    """omega = 1.0, c0 = 1.0.  Mode |k| = 0.9 radiates, mode |k| = 1.1 does not."""
    s = _two_modes_in_one_bin(np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0]]))
    assert s.radiating_fraction(1.0) == 0.5
    # the pre-fix path, kept only so the regression is visible
    binned = float(s.power[s.radiating_mask(1.0)].sum() / s.power.sum())
    assert binned == 1.0


def test_mask_excludes_k_zero_column():
    """At |k| = 0 the condition omega >= 0 holds everywhere, which would make the
    whole column radiate.  Its energy still belongs in the denominator."""
    only_mean = _two_modes_in_one_bin(np.array([[7.0, 0.0, 0.0], [7.0, 0.0, 0.0]]))
    assert only_mean.radiating_fraction(1.0) == 0.0
    assert not only_mean.radiating_mask_resolved(1.0)[:, 0].any()

    with_mean = _two_modes_in_one_bin(np.array([[7.0, 0.0, 0.0], [7.0, 1.0, 1.0]]))
    assert with_mean.radiating_fraction(1.0) == 1.0 / 16.0


def test_resolved_fraction_matches_mask_on_the_mode_grid():
    """Regression against an independent recomputation: sum the spectrum over each
    resolved |k| and apply the cone condition there.  This is what the metric is
    supposed to be, stated without reference to the implementation."""
    rng = np.random.default_rng(0)
    fld = rng.standard_normal((256, 32, 32))
    s = komega_spectrum(fld, G, dt=0.05)

    c0 = 1.3
    by_k = s.power_resolved.T                               # (n_k_res, n_omega)
    kk = s.k_resolved[:, None]
    ww = s.omega[None, :]
    expected = by_k[(ww >= c0 * kk) & (kk > 0)].sum() / by_k.sum()
    assert abs(s.radiating_fraction(c0) / expected - 1.0) < 1e-12

    # the resolved grid keeps the corner modes that the binned grid drops
    assert s.k_resolved.size > s.k_bins.size
    assert s.k_resolved.max() > s.k_bins.max()
