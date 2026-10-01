# %% [markdown]
# # 01 — The validation gate (Stage 1)
#
# **What you will learn:** how the acoustic pipeline (Lighthill source → 2D Green's
# function → far-field pressure) is checked against a flow whose answer is known, what
# each check can and cannot catch, and the one real bug the checks found.
#
# **Inputs:** none (the vortex pair is generated analytically). **Runtime:** ~3 min.
#
# **Old / new / future:** the gate as it stands today; a *reconstruction* of the Green's
# function as it was before 2026-09-15; and one future check (S-form vs T-form) that runs now.

# %% [bootstrap]

# %% [markdown]
# ## 1. The test flow: a co-rotating vortex pair
#
# Two equal vortices orbit their common centre at angular speed Ω. Viewed from far away
# the pair is a **spinning quadrupole**: the source repeats every half orbit, so it radiates
# a single tone at **2Ω**. That is a known answer the pipeline must reproduce.
#
# The parameters below are the ones `scripts/00_validate_acoustics.py` uses.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hankel1

from src.spectral import Grid, divergence
from src.solvers.vortex_pair import VortexPair
from src.acoustics.lighthill import lighthill_source, source_timeseries
from src.acoustics.greens import (source_spectrum, far_field_pressure, far_field_pressure_ibp,
                                  observer_ring, greens_2d)

grid = Grid(n=128)
vp = VortexPair(grid=grid, circulation=1.0, radius=0.6, core=0.18)
nt, dt = vp.suggest_sampling(8, 64)          # 8 orbits, 64 samples per orbit
u, v = vp.timeseries(nt, dt)
u_rms = float(np.sqrt(np.mean(u**2 + v**2)))
c0 = u_rms / 0.1                             # analysis Mach number 0.1
print(f"orbit frequency Omega = {vp.omega_orbit:.5f};  expected tone 2*Omega = {vp.omega_acoustic:.5f}")
print(f"record: {nt} samples, dt = {dt:.5f};  u_rms = {u_rms:.4f};  c0 = {c0:.4f}")
print(f"max |div u| = {np.abs(divergence(u[0], v[0], grid)).max():.2e}   (incompressible to round-off)")

fig, ax = plt.subplots(1, 2, figsize=(9, 4))
ax[0].imshow(vp.vorticity(0.0), origin="lower", cmap="RdBu_r"); ax[0].set_title("vorticity, t = 0")
ax[1].imshow(vp.vorticity(0.25 * vp.period), origin="lower", cmap="RdBu_r"); ax[1].set_title("a quarter orbit later")
for a in ax: a.set_xticks([]); a.set_yticks([])
plt.show()

# %% [markdown]
# ## 2. The Lighthill source, two ways
#
# For an incompressible field, $S=\partial_i\partial_j(u_iu_j)$ (**divergence form**) and
# $S=(\partial_j u_i)(\partial_i u_j)$ (**gradient form**) are algebraically identical. The
# code computes both. Agreement to machine precision is a free correctness check, and the
# point where they start to disagree flags an under-resolved vortex core.
#
# The divergence form is the one used for surrogate predictions, because a surrogate's
# output is not exactly divergence-free and the gradient form would hide that error.

# %%
sg = lighthill_source(u[0], v[0], grid, form="gradient")
sd = lighthill_source(u[0], v[0], grid, form="divergence")
print(f"gradient vs divergence form, max relative difference: {np.abs(sg - sd).max() / np.abs(sd).max():.2e}")

print("\nsource rms under grid refinement (converged = the source is not a discretisation artifact):")
prev = None
for n in (64, 128, 256):
    g2 = Grid(n=n)
    uu, vv = VortexPair(grid=g2, circulation=1.0, radius=0.6, core=0.18).velocity(0.0)
    s = float(np.sqrt(np.mean(lighthill_source(uu, vv, g2, form="divergence") ** 2)))
    print(f"  n = {n:4d}   rms(S) = {s:.6f}" + ("" if prev is None else f"   change {abs(s - prev) / prev:.1e}"))
    prev = s

# %% [markdown]
# ## 3. CHECK 1 — the tone is at 2Ω
#
# A probe of the source spectrum must peak at twice the orbital frequency.
#
# *Audit note (2026-09-15):* this check can hardly fail. Any quadratic quantity of a
# symmetric orbiting pair repeats every half orbit, whether or not the acoustics is right.
# It confirms the source is computed from the right flow, not that sound is radiated
# correctly. The later checks do that.

# %%
S = source_timeseries(u, v, grid)
probe = S[:, grid.n // 2, grid.n // 2 + grid.n // 6]
spec = np.abs(np.fft.rfft(probe * np.hanning(nt)))
omega = 2 * np.pi * np.fft.rfftfreq(nt, d=dt)
peak = omega[np.argmax(spec[1:]) + 1]
print(f"CHECK 1: tone at {peak:.5f} = {peak / vp.omega_orbit:.3f} x Omega   (want 2.000)")
plt.semilogy(omega, spec); plt.axvline(vp.omega_acoustic, color=style.OLD, ls="--", label=r"$2\Omega$")
plt.xlim(0, 6 * vp.omega_orbit); plt.xlabel(r"$\omega$"); plt.title("source spectrum at one probe"); plt.legend()
plt.show()

# %% [markdown]
# ## 4. The Green's function, and the bug the gate missed at first
#
# To radiate, the code transforms the source to frequency, $\hat S(\mathbf y,\omega)$, and
# integrates it against the Green's function:
# $\hat p(\mathbf x,\omega)=\int G(|\mathbf x-\mathbf y|)\,\hat S(\mathbf y,\omega)\,d\mathbf y$.
#
# Which Green's function is correct depends on the sign convention of the time transform.
# The code uses numpy's `rfft`, i.e. a signal is synthesised as $e^{+i\omega t}$. A wave
# travelling *outward* is then $e^{i(\omega t-kr)}$: its phase must **decrease** with
# distance, $d\varphi/dr=-k$. The outgoing 2D Green's function for that convention is
# $G=(-i/4)H_0^{(2)}(kr)$.
#
# **History.** Until 2026-09-15 the code used $H_0^{(1)}$, the textbook kernel for the
# *opposite* convention ($e^{-i\omega t}$). Under numpy's convention that is an
# **incoming** wave. The original gate (CHECKs 1–2) passed anyway. The independent audit
# found it by measuring $d\varphi/dr=+k$.
#
# The cell below **reconstructs** that situation. The old kernel is not in git, so it is
# rebuilt here as $(+i/4)H_0^{(1)}$, alongside a deliberately wrong 3D kernel. All three
# are run through the gate's checks.

# %%
s_hat, om_axis = source_spectrum(S, dt)
i_tone = int(np.argmin(np.abs(om_axis - vp.omega_acoustic)))
k_tone = om_axis[i_tone] / c0
X, Y = grid.meshgrid()
src_xy = np.stack([X.ravel(), Y.ravel()], axis=1)
s_tone = s_hat[i_tone].ravel()
dA = grid.dx ** 2
centre = (grid.length / 2, grid.length / 2)

KERNELS = {
    "2D outgoing (-i/4)H0(2)   [current]": lambda k, r: greens_2d(k, r, far_field=False),
    "2D incoming (+i/4)H0(1)   [before 09-15, reconstruction]": lambda k, r: 0.25j * hankel1(0, k * r),
    "3D outgoing e^{-ikr}/4pi r [wrong dimension]": lambda k, r: np.exp(-1j * k * r) / (4 * np.pi * r),
}

def radiate(kernel, observers):
    out = np.empty(len(observers), dtype=complex)
    for j, (ox, oy) in enumerate(observers):
        r = np.hypot(ox - src_xy[:, 0], oy - src_xy[:, 1])
        out[j] = np.sum(kernel(k_tone, r) * s_tone) * dA
    return out

r_near, r_far = 40 * grid.length, 160 * grid.length
theta = np.linspace(0, 2 * np.pi, 180, endpoint=False)
rows = []
for name, ker in KERNELS.items():
    ring = observer_ring(r_near, 180, centre)
    p = radiate(ker, ring)
    m_fit = np.polyfit(theta, np.unwrap(np.angle(p)), 1)[0]                      # CHECK 2
    ratio = np.abs(p[::5]).mean() / np.abs(radiate(ker, observer_ring(r_far, 36, centre))).mean()  # CHECK 3
    ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    p0 = radiate(ker, np.stack([centre[0] + r_near * np.cos(ang), centre[1] + r_near * np.sin(ang)], 1))
    p1 = radiate(ker, np.stack([centre[0] + (r_near + 1) * np.cos(ang), centre[1] + (r_near + 1) * np.sin(ang)], 1))
    dphi = float(np.mean(np.angle(p1 / p0)))                                         # CHECK 4
    rows.append((name, m_fit, ratio, dphi / k_tone))

md("| kernel | CHECK 2: phase winding m (want ±2) | CHECK 3: \\|p\\|(40L)/\\|p\\|(160L) (want 2.00) | "
   "CHECK 4: (dφ/dr)/k (want −1) |\n|---|---|---|---|\n" +
   "".join(f"| {n} | {m:+.3f} | {r:.3f} | {d:+.3f} |\n" for n, m, r, d in rows))

# %% [markdown]
# **Read the table row by row:**
# - **CHECK 2 passes for all three kernels.** A wrong kernel still produces the right
#   spinning pattern, so CHECK 2 cannot detect a kernel error.
# - **CHECK 3** (added 09-15) catches the wrong *dimension*. In 2D, |p| falls as $r^{-1/2}$,
#   so going 4× farther halves it (ratio 2.00). The 3D kernel gives 4.00.
# - **CHECK 4** (added 09-15) catches the wrong *direction*. Only the current kernel has
#   dφ/dr = −k.
#
# Both checks are now part of the gate.

# %% [markdown]
# ## 5. Run the gate as it stands today

# %%
out = subprocess.run([sys.executable, "scripts/00_validate_acoustics.py", "--n", "128",
                      "--outdir", str(ENV.cache / "gate")], capture_output=True, text=True)
print(out.stdout)
assert out.returncode == 0 and "GATE PASSED" in out.stdout
from IPython.display import Image, display
display(Image(filename=str(ENV.cache / "gate" / "00_validation.png")))

# %% [markdown]
# **Still open from the gate:** the build plan's original pass/fail criterion was the
# *absolute amplitude* of the radiated sound (Müller & Obermeier 1967). That comparison was
# never implemented. The review recommends Rucz et al. 2024 (J. Sound Vib. 587, 118496) as
# a modern, open-access replacement with error bounds (chapter 07).

# %% [markdown]
# ## 6. Future step, runnable now: two forms of the far field
#
# `review_2026-10-01.md` (chapter 05) finds that, for turbulence in a periodic box, the
# far field computed from S (the **S-form**) contains a large artifact. The fix is to move
# both derivatives onto the Green's function and integrate against $T_{ij}=u_iu_j$ directly
# (the **T-form**, `far_field_pressure_ibp`). This is the time-derivative form Sarkar &
# Hussaini (1993) recommend for periodic boxes.
#
# Before trusting the T-form on turbulence, it must reproduce the S-form on a flow where
# the S-form is trustworthy: this vortex pair, at its tone.

# %%
stress_hat = {key: source_spectrum(a, dt)[0] for key, a in [("uu", u * u), ("uv", u * v), ("vv", v * v)]}
ring = observer_ring(r_near, 72, centre)
rows = []
for label, w in [("Omega (no physical tone)", vp.omega_orbit), ("2 Omega (the tone)", vp.omega_acoustic),
                 ("4 Omega", 2 * vp.omega_acoustic)]:
    j = int(np.argmin(np.abs(om_axis - w)))
    pS = far_field_pressure(s_hat, om_axis, grid, ring, c0, True, [j])[0]
    pT = far_field_pressure_ibp(stress_hat, om_axis, grid, ring, c0, [j])[0]
    rows.append((label, om_axis[j], np.abs(pS).mean(), np.abs(pT).mean(),
                 np.abs(pS).mean() / np.abs(pT).mean(), np.linalg.norm(pS - pT) / np.linalg.norm(pT)))
md("| frequency | ω | mean\\|p\\| S-form | mean\\|p\\| T-form | S/T | relative complex difference |\n"
   "|---|---|---|---|---|---|\n" +
   "".join(f"| {a} | {b:.4f} | {c:.3e} | {d:.3e} | {e:.3f} | {f:.3f} |\n" for a, b, c, d, e, f in rows))
tone_row = rows[1]
print(f"at the tone the two forms agree to {abs(tone_row[4] - 1):.1%} in magnitude, "
      f"{tone_row[5]:.1%} as complex fields")

# %% [markdown]
# **What this shows** *(computed 2026-10-01; first time this check has been run)*:
# - **At the tone, where the pair genuinely radiates, the two forms agree:** within ~2% in
#   magnitude and ~6% as complex fields. The T-form implementation reproduces the known
#   answer to that accuracy.
# - **At Ω, ignore the ratio.** Both pressures are ~1e-18 or below, i.e. numerical zero:
#   the pair's source has no content at the orbital frequency, only at even multiples. A
#   ratio of two round-off numbers carries no information.
# - **At 4Ω (the weak second harmonic)** the magnitudes agree to ~10%, but the complex
#   fields differ by ~70%. The forms do not agree in detail once the signal is weak.
# - **Proposal (needs author approval):** make "S-form = T-form at the tone, within a set
#   tolerance" a new gate check before switching `scripts/03` to the T-form. Set the
#   tolerance only after understanding why the tone agrees to ~6% rather than ~0%, and why
#   4Ω differs in phase. The far-field asymptotic of ∂²G and the steady 1/r velocity of the
#   pair's net circulation (which does not vanish at the box edge) are two candidates.

# %% [markdown]
# ## What this chapter established
# - The pipeline reproduces the tone and the spinning pattern of a known flow.
# - CHECK 1 and CHECK 2 alone could not catch a wrong Green's function; CHECKs 3 and 4
#   (added 2026-09-15) catch the wrong dimension and the wrong direction.
# - The kernel is now outgoing, $(-i/4)H_0^{(2)}$, for numpy's convention.
# - The T-form far field agrees with the S-form on the vortex pair at its tone, to a few
#   percent. That is the prerequisite for using it on turbulence; turning it into a
#   pass/fail gate check needs a tolerance chosen with care (open).
