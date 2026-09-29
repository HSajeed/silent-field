"""A planted travelling wave must land in the right (k, omega) bin, and the
spectrum must obey Parseval. Both were bugs in the first draft."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.spectral import Grid
from src.acoustics.komega import komega_spectrum, sound_speed

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
