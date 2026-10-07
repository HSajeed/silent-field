"""The T-form cone share must reduce to R_S on the box grid, and the T-form radiated
power must agree with the far field `far_field_pressure_ibp` actually computes."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from src.spectral import Grid, project_solenoidal
from src.acoustics.komega import welch_komega_spectrum
from src.acoustics.lighthill import source_timeseries
from src.acoustics.tform import cone_share_refined


def _field(n=16, nt=96, seed=0):
    """A smooth, solenoidal, time-correlated random field."""
    rng = np.random.default_rng(seed)
    grid = Grid(n=n)
    kx, ky = grid.wavenumbers()
    filt = np.exp(-(kx**2 + ky**2) / 8.0)
    u = np.empty((nt, n, n)); v = np.empty_like(u)
    a = rng.standard_normal((2, n, n)); b = rng.standard_normal((2, n, n))
    for t in range(nt):
        ph = 2 * np.pi * t / nt
        f = [np.real(np.fft.ifft2(np.fft.fft2(np.cos(3 * ph) * a[c] + np.sin(5 * ph) * b[c]) * filt))
             for c in range(2)]
        u[t], v[t] = project_solenoidal(f[0], f[1], grid)
    return u, v, grid


def test_pad_one_is_exactly_R_S():
    u, v, grid = _field()
    dt = 0.05
    spec = welch_komega_spectrum(source_timeseries(u, v, grid), grid, dt, n_segments=4)
    c0s = [5.0, 10.0, 20.0]
    out = cone_share_refined(u, v, grid, dt, c0s, pad=1, n_segments=4, taper_alpha=0.0)
    r_s = [spec.radiating_fraction(c0) for c0 in c0s]
    np.testing.assert_allclose(out["share"], r_s, rtol=1e-9)
    assert max(out["share_sub_dk"]) == 0.0           # the box grid has nothing below dk


def test_refined_grid_adds_sub_dk_content_and_keeps_grid_modes():
    u, v, grid = _field()
    dt = 0.05
    r1 = cone_share_refined(u, v, grid, dt, [10.0], pad=1)
    r2 = cone_share_refined(u, v, grid, dt, [10.0], pad=2)
    assert r2["share_sub_dk"][0] > 0.0
    assert np.isfinite(r2["share"][0]) and r2["share"][0] > 0.0
    assert r1["share"][0] > 0.0


def test_radiated_power_tends_to_compact_limit_which_scales_as_c0_to_minus_4():
    from src.acoustics.tform import radiated_power
    u, v, grid = _field()
    c0s = [1e3, 2e3, 4e3]
    out = radiated_power(u, v, grid, 0.05, c0s, n_theta=16, n_segments=1)
    np.testing.assert_allclose(out["power"], out["compact"], rtol=1e-3)
    c = np.array(out["compact"])
    np.testing.assert_allclose(c[:-1] / c[1:], 16.0, rtol=1e-12)


def test_radiated_power_matches_the_far_field_solver():
    """k^3/c0 |n n T~(k n)|^2 must be what `far_field_pressure_ibp` radiates:
    |p|^2 = k^3 |q|^2 dA^2 / (8 pi r) on a distant ring (Fraunhofer + Hankel asymptote)."""
    from src.acoustics.tform import radiated_power, _stress_spectrum
    from src.acoustics.greens import far_field_pressure_ibp, observer_ring
    u, v, grid = _field(nt=32)
    dt, c0, n_theta, r = 0.05, 3.0, 16, 2e4
    T, om_idx, fold, norm, _ = _stress_spectrum(u, v, grid, taper_alpha=0.5)
    omega = om_idx / dt
    ring = observer_ring(r, n_theta, (0.0, 0.0))
    idx = list(range(1, omega.size))
    p = far_field_pressure_ibp(T, omega, grid, ring, c0, idx, far_field=False)
    k = omega[idx] / c0
    from_solver = (fold[idx] * norm * 8 * np.pi * r / c0 * np.mean(np.abs(p) ** 2, axis=1)
                   / grid.dx**4).sum()
    direct = radiated_power(u, v, grid, dt, [c0], n_theta=n_theta, n_segments=1)["power"][0]
    np.testing.assert_allclose(from_solver, direct, rtol=1e-3)


def test_stress_difference_with_itself_is_zero_and_with_zero_is_unchanged():
    u, v, grid = _field()
    dt = 0.05
    plain = cone_share_refined(u, v, grid, dt, [10.0], pad=2)
    vs_zero = cone_share_refined(u, v, grid, dt, [10.0], pad=2,
                                 minus=(np.zeros_like(u), np.zeros_like(v)))
    np.testing.assert_allclose(vs_zero["share"], plain["share"], rtol=1e-12)
    from src.acoustics.tform import radiated_power
    p = radiated_power(u, v, grid, dt, [10.0], n_theta=8, n_segments=1, minus=(u, v))
    assert p["power"][0] == 0.0 and p["compact"][0] == 0.0
