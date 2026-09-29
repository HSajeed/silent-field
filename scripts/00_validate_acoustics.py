#!/usr/bin/env python3
"""STAGE 1 GATE -- validate the acoustic pipeline against a known answer.

Co-rotating vortex pair. The checks performed are:
  1. the radiated tone is at TWICE the orbital frequency (rotating quadrupole)
  2. the far-field pressure winds with azimuthal mode m = 2 (a spinning
     quadrupole is nearly axisymmetric in MAGNITUDE -- "four lobes" is the
     wrong thing to look for; the structure is in the phase)
  3. the far-field kernel is the 2D one: mean |p'| on a ring falls off as
     1/sqrt(r), i.e. the ratio between r = 40L and r = 160L is 2.00, not the
     4.00 a 3D 1/r kernel would give
  4. the kernel is OUTGOING: dphi/dr at the tone must be ~= -k, under the
     np.fft.rfft (e^{+i omega t}) convention used throughout this pipeline

Also runs a resolution-convergence check on the Lighthill source, which is what
forecloses the objection that a large source error is really discretisation
error.

    python scripts/00_validate_acoustics.py
"""
import argparse, sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.spectral import Grid, divergence
from src.solvers.vortex_pair import VortexPair
from src.acoustics.lighthill import lighthill_source, source_timeseries
from src.acoustics.greens import source_spectrum, far_field_pressure, observer_ring, directivity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--periods", type=int, default=8)
    ap.add_argument("--ppp", type=int, default=64, help="samples per orbit")
    ap.add_argument("--mach", type=float, default=0.1)
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    grid = Grid(n=args.n)
    vp = VortexPair(grid=grid, circulation=1.0, radius=0.6, core=0.18)
    nt, dt = vp.suggest_sampling(args.periods, args.ppp)

    print(f"Omega_orbit = {vp.omega_orbit:.5f}")
    print(f"expected acoustic omega = {vp.omega_acoustic:.5f}")
    print(f"record: {nt} samples, dt = {dt:.5f}")

    u, v = vp.timeseries(nt, dt)
    u_rms = float(np.sqrt(np.mean(u**2 + v**2)))
    c0 = u_rms / args.mach
    print(f"u_rms = {u_rms:.4f}   c0 = {c0:.4f} (M = {args.mach})")

    div = np.abs(divergence(u[0], v[0], grid)).max()
    print(f"[check] max |div u| = {div:.3e}   (want ~1e-14)")
    assert div < 1e-8, "velocity field is not solenoidal"

    # ---- CHECK 1: two independent Lighthill formulations must agree.
    # They are algebraically identical for a solenoidal field, so any
    # disagreement is numerical -- and it turns out to be a sensitive and free
    # UNDER-RESOLUTION DETECTOR. With a Gaussian core of size sigma, the two
    # forms dealias at different stages (products vs final field), so they
    # diverge exactly when sigma stops being resolved. Empirically for
    # core=0.18: n=96 -> 1e-8, n=128 -> 1e-14. Treat a value above ~1e-10 as
    # "refine the grid", not as "the code is wrong".
    sg = lighthill_source(u[0], v[0], grid, form="gradient")
    sd = lighthill_source(u[0], v[0], grid, form="divergence")
    rel = np.abs(sg - sd).max() / np.abs(sd).max()
    print(f"[check] gradient vs divergence form: {rel:.3e}")
    if rel > 1e-10:
        print(f"        WARNING: core size {vp.core} is under-resolved at n={args.n}.")
        print(f"        Increase --n (or the core) before trusting source amplitudes.")
    assert rel < 1e-5, "formulations disagree far beyond a resolution effect"

    # ---- CHECK 2: convergence of the source under refinement
    print("[check] source convergence under grid refinement:")
    prev = None
    for n in (args.n // 2, args.n, args.n * 2):
        g2 = Grid(n=n)
        vp2 = VortexPair(grid=g2, circulation=1.0, radius=0.6, core=0.18)
        uu, vv = vp2.velocity(0.0)
        s = lighthill_source(uu, vv, g2, form="divergence")
        norm = float(np.sqrt(np.mean(s**2)))
        msg = "" if prev is None else f"   change {abs(norm-prev)/prev:.2e}"
        print(f"    n={n:4d}  rms(S) = {norm:.6f}{msg}")
        prev = norm

    # ---- CHECK 1: the tone is at 2*Omega
    S = source_timeseries(u, v, grid)
    probe = S[:, grid.n // 2, grid.n // 2 + grid.n // 6]
    spec = np.abs(np.fft.rfft(probe * np.hanning(nt)))
    omega = 2 * np.pi * np.fft.rfftfreq(nt, d=dt)
    peak = omega[np.argmax(spec[1:]) + 1]
    ratio = peak / vp.omega_orbit
    print(f"[CHECK 1] source tone omega = {peak:.5f}, ratio to Omega = {ratio:.3f}  (want 2.000)")
    assert abs(ratio - 2.0) < 0.1, "tone is not at twice the orbital frequency"

    # ---- CHECK 2: four-lobe quadrupole directivity
    s_hat, om_axis = source_spectrum(S, dt)
    i_tone = int(np.argmin(np.abs(om_axis - vp.omega_acoustic)))
    obs = observer_ring(radius=40.0 * grid.length, n_obs=180,
                        centre=(grid.length / 2, grid.length / 2))
    p = far_field_pressure(s_hat, om_axis, grid, obs, c0,
                           far_field=True, omega_indices=[i_tone])[0]
    d = directivity(p)
    theta = np.linspace(0, 2 * np.pi, len(d), endpoint=False)

    # A co-rotating pair radiates a SPINNING quadrupole: p' ~ cos(2 theta - 2
    # Omega t). At the tone frequency the magnitude |p'| is therefore nearly
    # axisymmetric -- the azimuthal structure lives in the PHASE, which must
    # advance as 2 theta (azimuthal mode number m = 2). Testing the magnitude
    # for four lobes is the natural-looking check and it is wrong; it returns a
    # near-null for a perfectly correct answer.
    phase = np.unwrap(np.angle(p))
    winding = (phase[-1] - phase[0]) / (2 * np.pi) * len(d) / (len(d) - 1)
    m_fit = float(np.polyfit(theta, phase, 1)[0])
    print(f"[CHECK 2] azimuthal phase winding m = {m_fit:.3f}  (want |m| = 2)")
    print(f"          magnitude anisotropy = {d.std()/d.mean():.4f}  "
          f"(small is CORRECT: the pattern spins)")
    assert abs(abs(m_fit) - 2.0) < 0.25, f"expected azimuthal mode 2, got {m_fit:.3f}"

    # ---- CHECK 3 (2D vs 3D kernel): mean |p'| on an observer ring must fall
    # off as 1/sqrt(r) (the 2D free-space kernel), not the 1/r of the 3D one.
    centre = (grid.length / 2, grid.length / 2)
    r_near, r_far = 40.0 * grid.length, 160.0 * grid.length
    obs_near = observer_ring(radius=r_near, n_obs=72, centre=centre)
    obs_far = observer_ring(radius=r_far, n_obs=72, centre=centre)
    p_near = far_field_pressure(s_hat, om_axis, grid, obs_near, c0,
                                far_field=True, omega_indices=[i_tone])[0]
    p_far = far_field_pressure(s_hat, om_axis, grid, obs_far, c0,
                               far_field=True, omega_indices=[i_tone])[0]
    ratio_r = float(np.abs(p_near).mean() / np.abs(p_far).mean())
    print(f"[CHECK 3] mean|p'|(r=40L) / mean|p'|(r=160L) = {ratio_r:.4f}  "
          f"(want 2.00 = sqrt(160/40); a 3D 1/r kernel would give 4.00)")
    assert abs(ratio_r - 2.0) < 0.05, \
        f"far-field magnitude does not decay as 1/sqrt(r) (got ratio {ratio_r:.3f}) -- wrong kernel?"

    # ---- CHECK 4 (outgoing): under the np.fft.rfft (e^{+i omega t})
    # convention used by source_spectrum, the phase must RETREAT with
    # distance: dphi/dr ~= -k. Averaged over several angles to avoid picking
    # up an azimuthal-only artefact.
    k_tone = om_axis[i_tone] / c0
    dr = 1.0
    angles = np.linspace(0.0, 2 * np.pi, 8, endpoint=False)
    obs0 = np.stack([centre[0] + r_near * np.cos(angles),
                      centre[1] + r_near * np.sin(angles)], axis=1)
    obs1 = np.stack([centre[0] + (r_near + dr) * np.cos(angles),
                      centre[1] + (r_near + dr) * np.sin(angles)], axis=1)
    p0 = far_field_pressure(s_hat, om_axis, grid, obs0, c0, far_field=True, omega_indices=[i_tone])[0]
    p1 = far_field_pressure(s_hat, om_axis, grid, obs1, c0, far_field=True, omega_indices=[i_tone])[0]
    dphi_dr = float(np.mean(np.angle(p1 / p0)) / dr)
    print(f"[CHECK 4] dphi/dr = {dphi_dr:.5f}   -k = {-k_tone:.5f}  (want dphi/dr ~= -k: outgoing wave)")
    assert abs(dphi_dr - (-k_tone)) < 0.05 * abs(k_tone), \
        f"kernel is not outgoing -- got dphi/dr = {dphi_dr:.5f}, expected ~ {-k_tone:.5f}"

    out = Path(args.outdir); out.mkdir(exist_ok=True, parents=True)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        ax[0].imshow(vp.vorticity(0.0), origin="lower"); ax[0].set_title("vorticity")
        ax[1].semilogy(omega, spec); ax[1].axvline(vp.omega_acoustic, color="r", ls="--")
        ax[1].set_xlim(0, 6 * vp.omega_orbit); ax[1].set_title("source spectrum (red = 2$\\Omega$)")
        ax[1].set_xlabel("$\\omega$")
        axp = plt.subplot(1, 3, 3, projection="polar"); axp.plot(theta, d)
        axp.set_title("far-field directivity")
        fig.tight_layout(); fig.savefig(out / "00_validation.png", dpi=140)
        print(f"wrote {out/'00_validation.png'}")
    except Exception as e:
        print(f"(plot skipped: {e})")

    print("\nGATE PASSED -- acoustic pipeline reproduces the known answer.")


if __name__ == "__main__":
    main()
