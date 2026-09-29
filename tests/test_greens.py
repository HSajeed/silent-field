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
