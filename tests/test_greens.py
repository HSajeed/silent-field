"""2D Green's function must be the Hankel kernel, not the 3D 1/r one."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.special import hankel2
from src.acoustics.greens import greens_2d

def test_matches_hankel():
    """Under np.fft.rfft's e^{+i omega t} synthesis, the outgoing kernel is
    the complex conjugate of the textbook e^{-i omega t} one: (-i/4) H_0^(2)."""
    r = np.linspace(0.5, 20, 50)
    assert np.abs(greens_2d(1.3, r) - (-0.25j)*hankel2(0, 1.3*r)).max() < 1e-12

def test_far_field_asymptote():
    """Exact and asymptotic forms must converge as kr grows."""
    r = np.array([200.0, 2000.0])
    err = np.abs(greens_2d(1.0, r, False) - greens_2d(1.0, r, True)) / np.abs(greens_2d(1.0, r, False))
    assert err[1] < err[0] and err[1] < 1e-3

def test_decays_as_inverse_sqrt_r():
    """2D far field falls off as 1/sqrt(r). Getting 1/r here means the 3D
    kernel crept in."""
    r = np.array([100.0, 400.0])
    g = np.abs(greens_2d(1.0, r, True))
    assert abs(g[0]/g[1] - 2.0) < 0.01     # sqrt(400/100) = 2

def test_is_outgoing():
    """Under the np.fft.rfft convention, phase must RETREAT with distance:
    angle(G(r+delta)/G(r)) ~ -k*delta, for both the exact and far-field forms."""
    k = 1.3
    r = np.array([5.0, 50.0])
    delta = 1e-4
    for far_field in (False, True):
        g0 = greens_2d(k, r, far_field)
        g1 = greens_2d(k, r + delta, far_field)
        dphi = np.angle(g1 / g0)
        assert np.abs(dphi - (-k * delta)).max() < 1e-6


# ------------------------------------------------------------- radiating band
def _omega(nt=1024, dt=0.04):
    return 2.0 * np.pi * np.fft.rfftfreq(nt, d=dt)


def test_band_only_contains_radiating_frequencies():
    """Every selected frequency must satisfy the radiation condition.  The original
    bug took om[:24] by array index, which at M = 0.1 lands entirely below c0 * dk --
    0 of 24 frequencies able to radiate."""
    from src.acoustics.greens import radiating_band
    from src.spectral import Grid

    om = _omega()
    grid = Grid(n=128)
    c0 = 11.308
    idx, meta = radiating_band(om, c0, grid, n_band=24, mode="k")

    assert len(idx) == 24
    assert (om[idx] >= c0 * (2 * np.pi / grid.length)).all()
    assert meta["n_radiating_available"] > len(idx)
    # the pre-fix selection, for contrast
    old = [i for i in range(1, len(om)) if om[i] > 0][:24]
    assert (om[old] >= c0 * (2 * np.pi / grid.length)).sum() == 0


def test_band_is_independent_of_record_length():
    """Doubling the record must not move the band in physical frequency.  The old
    index slice scaled with record length, so the 'far-field error' it produced was
    not even a stable quantity.  The grid mode each band frequency sits on is the
    record-length invariant; the realised omega only has to land in the same bin."""
    from src.acoustics.greens import radiating_band
    from src.spectral import Grid

    grid, c0 = Grid(n=128), 11.308
    dk = 2 * np.pi / grid.length
    for nt in (512, 1024, 2048):
        om = _omega(nt)
        idx, _ = radiating_band(om, c0, grid, mode="k")
        k = om[idx] / c0 / dk
        assert abs(k[0] - 1.0) < 0.05, k[0]
        assert abs(k[-1] - om.max() / c0 / dk) < 0.05 * k[-1]


def test_top_band_is_a_subset_of_radiating_and_finds_the_peak():
    from src.acoustics.greens import radiating_band
    from src.spectral import Grid

    om = _omega()
    grid, c0 = Grid(n=128), 11.308
    rng = np.random.default_rng(0)
    power = rng.random(len(om))
    # plant a strong peak on a radiating frequency
    rad = np.nonzero(om >= c0 * (2 * np.pi / grid.length))[0]
    peak = int(rad[5])
    power[peak] = 100.0

    idx, meta = radiating_band(om, c0, grid, n_band=8, mode="top", power=power)
    assert peak in idx
    assert set(idx) <= set(rad.tolist())
    assert meta["band_share_of_radiating_power"] > 0.0


def test_band_reports_kr_large_enough_for_the_far_field_asymptote():
    from src.acoustics.greens import radiating_band
    from src.spectral import Grid

    om = _omega()
    grid = Grid(n=128)
    radius = 30 * grid.length
    _, meta = radiating_band(om, 11.308, grid, mode="k", radius=radius)
    assert meta["kr_range"][0] > 100.0


def test_band_reports_empty_when_the_record_cannot_reach_the_cone():
    from src.acoustics.greens import radiating_band
    from src.spectral import Grid

    om = _omega(nt=64, dt=0.04)      # omega_max = 78.5
    idx, meta = radiating_band(om, 200.0, Grid(n=128), mode="k")
    assert idx == [] and meta["n_radiating_available"] == 0


def test_auto_band_keeps_k_when_it_sees_enough_power():
    from src.acoustics.greens import select_band
    from src.spectral import Grid

    om = _omega()
    grid, c0 = Grid(n=128), 11.308
    power = np.ones(len(om))
    rad = np.nonzero(om >= c0 * (2 * np.pi / grid.length))[0]
    power[:rad[0]] = 0.0
    power[rad] = 1.0
    idx, meta = select_band(om, c0, grid, n_band=24, mode="auto", power=power)
    assert meta["mode"] == "k"
    assert meta["fallback_reason"] is None


def test_auto_band_falls_back_to_top_when_k_under_covers():
    """24 log-spaced samples cannot cover 439 radiating bins, so a threshold asking
    for a large share must reject the 'k' band.  This is the decision logic itself,
    which is what has to be tested -- `scripts/03` needs a trained checkpoint to
    reach this point at all."""
    from src.acoustics.greens import select_band
    from src.spectral import Grid

    om = _omega()
    grid, c0 = Grid(n=128), 11.308
    rad = np.nonzero(om >= c0 * (2 * np.pi / grid.length))[0]
    power = np.zeros(len(om))
    power[rad] = 1.0                     # uniform over everything that can radiate

    idx, meta = select_band(om, c0, grid, n_band=24, mode="auto", power=power,
                            min_fraction=0.5)
    assert meta["mode"] == "top"
    assert "under the" in meta["fallback_reason"]
    assert meta["band_mode_requested"] == "auto"
    assert set(idx) <= set(rad.tolist())

    # the same fixture at a realistic threshold keeps the physical band
    _, kept = select_band(om, c0, grid, n_band=24, mode="auto", power=power,
                          min_fraction=0.01)
    assert kept["mode"] == "k"
