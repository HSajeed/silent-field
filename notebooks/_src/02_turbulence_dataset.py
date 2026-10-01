# %% [markdown]
# # 02 — The turbulence dataset
#
# **What you will learn:** what the forced 2D turbulence simulation solves and how; why
# the time sampling was chosen the way it was; how stationary the record is; and why its
# energy spectrum is steeper than the textbook k⁻³ (and why the obvious fix is risky).
#
# **Inputs:** `turb2d.npz` (required); the manifests of the other two datasets.
# **Runtime:** ~5 min, plus an optional ~1 min toy simulation.
#
# **Old / new / future:** all analysis here is current. The "spin-up" correction and the
# slope refit are history from the 2026-09-15 audit and follow-up.

# %% [bootstrap]

# %% [markdown]
# ## 1. What is simulated
#
# The incompressible 2D Navier–Stokes equations in vorticity form, on a doubly periodic
# box of side L:
#
# $$\partial_t\omega + \mathbf u\cdot\nabla\omega = \nu\nabla^2\omega - \mu\,\omega + f,
# \qquad \omega=-\nabla^2\psi,\ \ u=\partial_y\psi,\ v=-\partial_x\psi.$$
#
# - **Forcing** $f=A k_f\sin(k_f y)$: Kolmogorov forcing at $k_f=8$, deterministic and
#   reproducible.
# - **Linear drag** $\mu\omega$: not optional in 2D. Energy cascades to *large* scales
#   (the inverse cascade), and without a large-scale sink it piles up in the box-size mode.
#   Section 5 comes back to this.
# - **Numerics:** pseudospectral (every derivative exact on the periodic grid), 2/3-rule
#   dealiasing, RK4 with an exact integrating factor for the viscous and drag terms.
#
# Owning a ~120-line solver keeps one guarantee visible: no source error can be blamed on
# finite-difference derivatives.

# %%
import json
import numpy as np
import matplotlib.pyplot as plt

from src.spectral import Grid, vorticity_from_velocity
from src.solvers.io import load_dataset
from src.solvers.turb2d import Turb2D, Turb2DConfig, acoustic_sampling_interval
from src.diagnostics.spectra import time_averaged_energy_spectrum, fit_slope

info = nbenv.verify_dataset(ENV, "turb2d")
u, v, meta = load_dataset(ENV.dataset("turb2d"))
grid = Grid(n=u.shape[-1], length=meta["grid_length"])
dt_save = meta["dt"]
u_rms = float(np.sqrt(np.mean(u**2 + v**2)))
md("**Configuration (from the tracked manifest `data/turb2d.manifest.json`):**\n\n"
   "| parameter | value |\n|---|---|\n" +
   "".join(f"| {k} | {v} |\n" for k, v in info["config"].items()) +
   f"| spin-up steps | 5000 (from the rerun log; not recorded in the manifest, an open item) |\n"
   f"| saved every | 20 steps, so dt_save = {dt_save} |\n| snapshots | {u.shape[0]} (record length {u.shape[0] * dt_save:.2f}) |\n"
   f"| u_rms | {u_rms:.4f} |\n")

# %% [markdown]
# ## 2. Why the sampling interval matters
#
# The radiating cone needs ω ≥ c₀|k|. The smallest non-zero wavenumber in the box is
# $k_{min}=2\pi/L$, so the record must resolve frequencies up to at least
# $c_0k_{min}=(U/M)(2\pi/L)$. Nyquist ($\pi/\Delta t$) above that gives
# $\Delta t < M L/(2U)$. The code adds a 4× safety factor.

# %%
dt_max = acoustic_sampling_interval(u_rms, grid, mach=0.1)
print(f"bound at M = 0.1 (with 4x safety): dt < {dt_max:.5f};  dataset dt_save = {dt_save}  "
      f"-> oversampled {dt_max / dt_save:.2f}x")
print(f"Nyquist omega = {np.pi / dt_save:.2f};  c0 * k_min at M = 0.1 = {u_rms / 0.1 * 2 * np.pi / grid.length:.2f}")

# %% [markdown]
# ## 3. What the flow looks like

# %%
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for a, i in zip(ax, (0, u.shape[0] // 2, u.shape[0] - 1)):
    w = vorticity_from_velocity(u[i], v[i], grid)
    a.imshow(w, origin="lower", cmap="RdBu_r", vmin=-np.abs(w).max(), vmax=np.abs(w).max())
    a.set_title(f"vorticity, t = {i * dt_save:.1f}"); a.set_xticks([]); a.set_yticks([])
plt.show()

# %% [markdown]
# ## 4. Is the record statistically stationary?
#
# Every (k, ω) estimate in chapter 03 assumes it is.
#
# **History.** The 2026-09-15 rerun printed u_rms values that dipped during "spinning
# up..." and flagged a possibly unfinished spin-up. The independent audit found that those
# prints come from the *recording* loop, not the spin-up loop (which prints nothing), so
# they describe the saved record itself. It then measured stationarity on the record
# directly. The cell below repeats that measurement.

# %%
ke = 0.5 * np.mean(u**2 + v**2, axis=(1, 2))
t = np.arange(len(ke)) * dt_save
cv = ke.std() / ke.mean()
slope_fit = np.polyfit(t, ke, 1)[0]
drift = abs(slope_fit) * t[-1] / ke.mean()
x = ke - ke.mean()
ac = np.correlate(x, x, "full")[len(x) - 1:] / (x.var() * len(x))
t_corr = t[np.argmax(ac < np.exp(-1))]
print(f"kinetic energy: mean {ke.mean():.3f}, min {ke.min():.3f}, max {ke.max():.3f}")
print(f"coefficient of variation {cv:.1%}   (audit: 4.4%)")
print(f"linear drift over the record {drift:.1%} of the mean   (audit: ~6%)")
print(f"autocorrelation e-folding time {t_corr:.2f}   (audit: ~6.44) -> "
      f"the record spans ~{t[-1] / t_corr:.0f} correlation times")
plt.plot(t, ke); plt.xlabel("t"); plt.ylabel("kinetic energy"); plt.title("kinetic energy over the record")
plt.show()

# %% [markdown]
# **Reading:** stationary by the CV measure, but only ~6 independent samples long. That is
# why chapter 03 reports every number as a band (full record vs each half), and why the
# band is wide (×2–4).
#
# A separate, valid concern from the audit stands: the spin-up (5000 steps = 10 time units)
# is shorter than the drag time 1/μ = 12.5. Spin-up length is not recorded in the manifest
# (HANDOFF open item 2).

# %% [markdown]
# ## 5. The energy spectrum: −3.98, not −3
#
# 2D turbulence theory (Kraichnan) predicts $E(k)\propto k^{-3}$ in the *enstrophy cascade*
# range, between the forcing scale and the dissipation scale.
#
# **History.**
# - The 09-15 rerun fitted a single snapshot over k ∈ [4, 32] and got −3.70.
# - The audit noted that range includes the forcing wavenumber (k = 8), mixing injection
#   with cascade.
# - The follow-up refitted a time average (64 snapshots) over k ∈ [12, 36], excluding
#   forcing: **−3.98**. No fit range reaches −3.

# %%
k, e, n_used = time_averaged_energy_spectrum(u, v, grid, stride=16)
slope = fit_slope(k, e, 12, 36)
Comparison("energy-spectrum slope").add("slope on k in [12, 36], time-averaged", new=slope,
                                         logged=-3.981, check="new", rtol=2e-3,
                                         source="rerun log B3, scripts/05").show()
print("local slope on [k, 2k]:", ", ".join(f"k={kk}: {fit_slope(k, e, kk, 2 * kk):.2f}" for kk in range(10, 21, 2)))

good = (k > 0) & (e > 0)
plt.loglog(k[good], e[good], color=style.NEW, label="E(k), time-averaged")
kk = k[(k >= 12) & (k <= 36)]
c = np.exp(np.polyfit(np.log(kk), np.log(e[(k >= 12) & (k <= 36)]), 1)[1])
plt.loglog(kk, c * kk**slope, color=style.OLD, lw=2, label=f"fit {slope:.2f}")
plt.loglog(k[good], e[good][0] * k[good][0]**3 * k[good]**-3.0, "k--", lw=0.8, label=r"$k^{-3}$")
plt.axvline(8, color="g", ls=":", label="forcing k = 8")
plt.xlabel("|k|"); plt.ylabel("E(k)"); plt.legend(); plt.show()

# %% [markdown]
# **Why steeper than −3** *(cited, see `review_2026-10-01.md` §4.4)*. Linear friction
# steepens the enstrophy-cascade spectrum. Boffetta & Ecke (Annu. Rev. Fluid Mech. 2012,
# §4.3): with friction, "E(k) is predicted to become steeper than −3 with a correction
# proportional to the friction coefficient α" (Lilly 1972; Nam et al. 2000). The available
# scale separation is also short: forcing at k = 8, dealiasing cutoff at k ≈ 42.7, about 5×.
# Borue (1993), as cited in the same review, finds k⁻³ is approached only as resolution grows.
#
# **Why not just lower the drag** (`branch_analysis.md` §6G proposed 0.08 → 0.01):
# - In a finite box with too little large-scale dissipation, the inverse cascade piles
#   energy into the lowest mode, forming a *condensate* (Boffetta & Ecke 2012 §4.2;
#   Chertkov et al. 2007, PRL 99, 084501).
# - The radiating-cone energy of this dataset lives almost entirely in that lowest mode
#   (chapter 03), so a condensate would change the headline number drastically.
# - It would also mean regenerating data.
#
# The review's recommendation: keep −3.98 and explain it with the friction result.

# %% [markdown]
# ## 6. Integral length scale
#
# How big are the energetic eddies compared with the box? This matters in chapter 05:
# Sarkar & Hussaini (1993) require the periodic box to be much larger than the flow's
# correlation length before its sound can be computed.

# %%
uh = np.fft.fft2(u[::32]); vh = np.fft.fft2(v[::32])
P = (np.abs(uh)**2 + np.abs(vh)**2).mean(0)
kmag = np.sqrt(grid.k_sq); m = kmag > 0
L_int = np.pi / 2 * (P[m] / kmag[m]).sum() / P[m].sum()
Comparison("integral scale").add("integral length scale", new=L_int, logged=0.512, check="new", rtol=2e-3,
                                 source="review §1.3").show()
print(f"box / integral scale = {grid.length / L_int:.1f}")

# %% [markdown]
# ## 7. Optional: watch the solver run (toy size)
#
# A 64² run from a random initial condition, kept in memory only. The real datasets are
# **not** regenerated (project rule): every logged number refers to the files on Drive.

# %%
RUN_TOY = True
if RUN_TOY:
    toy = Turb2D(Turb2DConfig(n=64, viscosity=5e-4, drag=0.08, forcing_k=8.0, dt=2e-3))
    tu, tv, _ = toy.run(n_snapshots=6, save_every=250, spinup_steps=500, progress=False)
    fig, ax = plt.subplots(1, 6, figsize=(14, 2.6))
    for i, a in enumerate(ax):
        a.imshow(vorticity_from_velocity(tu[i], tv[i], toy.grid), origin="lower", cmap="RdBu_r")
        a.set_title(f"t = {1.0 + (i + 1) * 0.5:.1f}", fontsize=8); a.set_xticks([]); a.set_yticks([])
    plt.suptitle("toy run, 64², from a random start: the k = 8 forcing organises the flow")
    plt.show()

# %% [markdown]
# ## 8. The two other datasets
#
# Both were generated on 2026-09-15 to test whether the headline number is a property of
# the flow or of the set-up (chapter 03).

# %%
for name, purpose in [("turb2d_kf16", "same box, forcing at k = 16: does the forcing scale matter?"),
                      ("turb2d_L4pi", "box twice as large (n = 256, same dx): does the box size matter?")]:
    man = json.loads(ENV.manifest(name).read_text())
    present = "present" if ENV.has(name) else "not uploaded (optional)"
    cfg = man["config"]
    print(f"{name}: {purpose}\n   L = {cfg['length']:.4f}, n = {cfg['n']}, forcing k = {cfg['forcing_k']}, "
          f"drag = {cfg['drag']}, manifest {man['hash']}, u_rms = {man['extra']['u_rms']:.4f}  [{present}]")
print("\nNote: the L4pi spin-up length cannot be verified from its artifacts (rerun log B7).")

# %% [markdown]
# ## What this chapter established
# - The dataset is forced 2D turbulence at 128², sampled fast enough for the M = 0.1 cone.
# - It is stationary to ~4%, but only ~6 correlation times long, so every number from it
#   needs a band.
# - Its spectrum is steeper than k⁻³, for a known reason (friction). Lowering the drag
#   risks a box-scale condensate exactly where the cone energy lives.
# - Energetic eddies are ~12× smaller than the box.
