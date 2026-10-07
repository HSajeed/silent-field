# %% [markdown]
# # 04 — Tier-0 fixes, old vs new
#
# **What you will learn:** the three correctness bugs fixed on 2026-09-30 ("Tier 0"), plus
# two fixed earlier (2026-09-15). For each one:
# - what the old code did;
# - how the bug was found;
# - the old number, computed by **running the old code** and checked against the log
#   written at the time;
# - the new number;
# - why the new one is right.
#
# **Inputs:** `turb2d.npz`; spectra cached by chapter 03 if available. **Runtime:** ~10 min.
#
# **How "old" works:** `nbtools.versions.old_src()` extracts `src/` from the initial commit
# `2535bb4` (2026-09-29, after the 09-15 follow-up, before Tier 0). It imports that copy as
# the package `src_old`, so both versions run side by side in this kernel. The two 09-15
# fixes predate that commit; they are shown as labelled *reconstructions*.

# %% [bootstrap]

# %%
import inspect
import json
import numpy as np
import matplotlib.pyplot as plt

from nbtools import spectra
from nbtools.versions import old_src, git_show, diff, INITIAL_COMMIT

old_src()
from src_old.acoustics.radiating import effective_weight_in_cone as cone_share_old
from src.acoustics.radiating import effective_weight_in_cone as cone_share_new, sobolev_lambda_sweep
from src.acoustics.komega import sound_speed

OLD_LOG = {r["mach"]: r for r in json.loads(git_show("figures/komega_metrics.json"))["rows"]}
NEW_LOG = {r["mach"]: r for r in json.loads((ENV.repo / "figures/komega_metrics.json").read_text())["rows"]}

nbenv.verify_dataset(ENV, "turb2d")
old = spectra.spectra_for(ENV, "turb2d", old=True)
new = spectra.spectra_for(ENV, "turb2d", halves=True)
U, grid = new["U"], new["grid"]
c0 = sound_speed(U, 0.1)
print(f"old code: {INITIAL_COMMIT};  new code: working tree;  u_rms = {U:.4f};  c0(M=0.1) = {c0:.3f}")

# %% [markdown]
# ## Fix 1 — the cone mask was evaluated on bin centres
#
# **What the old code did.** To draw the (|k|, ω) picture, each 2D mode is put into a
# radial bin centred on an integer multiple of 2π/L. The old metric then tested
# ω ≥ c₀ × (bin centre) instead of ω ≥ c₀ × (the mode's own |k|).
#
# **Why that is wrong.** On a square grid, a bin holds several different |k|. The bin centred
# on 5 (edges 4.5 and 5.5) contains 5, √26 and √29. A mode at |k| = √29 = 5.39 needs
# ω ≥ 5.39 c₀ to radiate, but the old mask accepted it from ω ≥ 5 c₀. Near the cone edge
# that admits a slab of non-radiating energy.
#
# *Note:* HANDOFF.md and README.md illustrate this with "√21, 5, √29". There is no grid
# mode at |k| = √21, because 21 is not a sum of two squares. The point stands; the example
# needs √26 instead of √21.
#
# **How it was found** (`TIER0_PLAN.md`, 2026-09-30): by re-deriving the metric from the
# mode grid instead of trusting the binned arrays.

# %%
kx = np.fft.fftfreq(grid.n, d=1 / grid.n)
KX, KY = np.meshgrid(kx, kx)
kmag = np.hypot(KX, KY)
in_bin = np.unique(np.round(kmag[(kmag >= 4.5) & (kmag < 5.5)] ** 2).astype(int))
md("| \\|k\\| in the bin centred on 5 | old test (bin centre) | correct test | ω window wrongly counted as radiating (M = 0.1) |\n|---|---|---|---|\n" +
   "".join(f"| √{q} = {np.sqrt(q):.3f} | ω ≥ {5 * c0:.1f} | ω ≥ {np.sqrt(q) * c0:.1f} | "
           f"{'[%.1f, %.1f)' % (5 * c0, np.sqrt(q) * c0) if np.sqrt(q) > 5 else 'none'} |\n" for q in in_bin))
below = np.unique(np.round(kmag[(kmag >= 5.5) & (kmag < 6.0)] ** 2).astype(int))
print("in other bins the error also goes the other way: the bin centred on 6 holds "
      + ", ".join(f"√{q} = {np.sqrt(q):.3f}" for q in below)
      + " below its centre, which the old mask wrongly *excluded* until ω ≥ 6 c0")

# %%
tab = Comparison("Fix 1: cone share at M = 0.1, headline estimator (Welch, 4 segments)")
r_old = cone_share_old(old["vel_full"], c0, "l2"); r_new = cone_share_new(new["vel_full"], c0, "l2")
s_old = cone_share_old(old["src_full"], c0, "l2"); s_new = cone_share_new(new["src_full"], c0, "l2")
tab.add("R (velocity)", old=r_old, new=r_new, logged=OLD_LOG[0.1]["l2"], check="old", rtol=1e-9,
        source=f"komega_metrics.json @ {INITIAL_COMMIT}")
tab.add("R (velocity), new", old=r_old, new=r_new, logged=NEW_LOG[0.1]["l2"], check="new", rtol=1e-9,
        source="komega_metrics.json now")
tab.add("R_S (source)", old=s_old, new=s_new, logged=OLD_LOG[0.1]["src_l2"], check="old", rtol=1e-9,
        source=f"komega_metrics.json @ {INITIAL_COMMIT}")
tab.add("R_S (source), new", old=s_old, new=s_new, logged=NEW_LOG[0.1]["src_l2"], check="new", rtol=1e-9,
        source="komega_metrics.json now")
tab.show()
print(f"inflation removed: velocity {r_old / r_new:.2f}x, source {s_old / s_new:.2f}x")

# %% [markdown]
# **About the size of the inflation.** `TIER0_PLAN.md` and HANDOFF quote **2.84×** and
# **4.79×**. Those were measured on a *single full-record Hann* estimate
# (`TIER0_PLAN.md:34`: "velocity 1.47e-7 vs 5.18e-8 exact"). On the headline estimator
# (Welch, 4 segments) the same fix moves R by 2.71× and R_S by 4.52×. Both are correct for
# their estimator; quote the one that matches the number you are correcting.
#
# **Two smaller parts of the same fix.** Neither changes the numbers on this data, as the
# next cell shows.
# - A `k > 0` guard: the condition ω ≥ c₀·0 is true for the whole k = 0 column, so without
#   the guard the mean mode would count as "radiating".
# - The **corner modes** (|k| up to √2·64 ≈ 90.5) are now inside the total; the old binning
#   stopped at |k| = 63.5.

# %%
spec = new["vel_full"]
k_res, P = spec.k_resolved, spec.power_resolved
mask = spec.radiating_mask_resolved(c0)
inside_bins = k_res < spec.k_bins.size - 0.5
r_trunc = (P * mask * inside_bins).sum() / (P * inside_bins).sum()
print(f"energy in the k = 0 column (what the guard excludes): {P[:, k_res == 0].sum() / P.sum():.1e} of the total")
print(f"energy in the corner modes |k| > 63.5:                 {P[:, ~inside_bins].sum() / P.sum():.1e} of the total")
print(f"split of the change: mask {r_old / r_trunc:.4f}x  x  corner/guard {r_trunc / r_new:.4f}x")

# %%
print(diff("src/acoustics/komega.py", grep="radiating_mask")[:2500])

# %% [markdown]
# ## Fix 2 — the Sobolev objective was analysed in the wrong form
#
# **Background.** "Sobolev training" adds a penalty on derivatives, i.e. it up-weights high
# wavenumbers. It is the standard remedy for spectral bias. The project's Stage-4 claim was
# that it cannot rescue the cone.
#
# **What the old code did.** It scored the Sobolev objective as a *pure reweighting*
# w(k) = (1 + k²): `effective_weight_in_cone(..., "h1")` multiplied the energy by that
# weight. That gave a cone share 19× below L2, reported as "Sobolev makes it about ten
# times worse".
#
# **What training actually does.** `src/models/losses.py::sobolev_loss`:

# %%
from src.models import losses
print(inspect.getsource(losses.sobolev_loss).split('"""')[0] + "    ...\n" +
      "\n".join(inspect.getsource(losses.sobolev_loss).split("\n")[-9:]))

# %% [markdown]
# It returns `base + weight * high`: the plain MSE **plus** λ times a normalised
# (1 + k²)-weighted term. Per mode, the weight is therefore 1 + λ·(…)·(1 + k²)/⟨1 + k²⟩,
# not (1 + k²). With λ = 1e-2 (the training default), inside the cone the second term is
# tiny and the weight is ≈ 1, i.e. L2.
#
# **How it was found:** `branch_analysis.md` §2.2 (2026-09-29) compared the analysis code
# with the training code.

# %%
tab = Comparison("Fix 2: cone share of the Sobolev objective at M = 0.1")
h1_old = cone_share_old(old["vel_full"], c0, "h1")
h1_mult = cone_share_new(new["vel_full"], c0, "h1", convention="multiplicative")
h1_add = cone_share_new(new["vel_full"], c0, "sobolev", lam=1e-2, convention="additive", grid=grid)
tab.add("H1 as pure reweighting (old code, old mask)", old=h1_old, logged=OLD_LOG[0.1]["h1"], check="old",
        rtol=1e-9, source=f"komega_metrics.json @ {INITIAL_COMMIT}")
tab.add("H1 as pure reweighting (new mask)", new=h1_mult, logged=NEW_LOG[0.1]["h1_mult"], check="new",
        rtol=1e-9, source="komega_metrics.json now")
tab.add("Sobolev as trained, λ = 1e-2 (new)", new=h1_add, logged=NEW_LOG[0.1]["h1_add"], check="new",
        rtol=1e-9, source="komega_metrics.json now")
tab.show()
print(f"old reading: Sobolev cone share = {h1_old / r_old:.3f} x L2  ('about ten times worse')")
print(f"as trained:  Sobolev cone share = {h1_add / r_new:.5f} x L2")

# %%
sweep = sobolev_lambda_sweep(new["vel_full"], c0, grid=grid, lambdas=(1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6))
lam = [r["lambda"] for r in sweep if r["lambda"] is not None]
ratio = [r["ratio_to_l2"] for r in sweep if r["lambda"] is not None]
logged = json.loads((ENV.repo / "figures/06_sobolev_argument.json").read_text())
logged_m01 = [r for r in logged["rows"] if r["mach"] == 0.1][0]
logged_ratio = [r["ratio_to_l2"] for r in logged_m01["sweep"] if r["lambda"] is not None]
assert np.allclose(ratio, logged_ratio, rtol=1e-9), "sweep does not reproduce figures/06_sobolev_argument.json"
plt.semilogx(lam, ratio, "o-", color=style.NEW, label="Sobolev as trained (additive)")
plt.axhline(h1_mult / r_new, color=style.OLD, ls="--", label="pure (1+k²) reweighting")
plt.axvline(1e-2, color=style.REF, ls=":", label="training default λ = 1e-2")
plt.xlabel("λ"); plt.ylabel("cone share ÷ L2 cone share"); plt.legend(); plt.title("λ sweep, M = 0.1 (reproduces scripts/06)")
plt.show()

# %% [markdown]
# **What Tier 0 concluded:** "Sobolev is inert, not harmful": at the default λ the cone
# share equals the L2 share.
#
# **What the 2026-10-01 review adds (chapter 05):**
# - at λ = 1e-2 the Sobolev term is only a tiny fraction of the whole objective (~0.03% at
#   the review, before the 2026-10-07 factor-2 fix; now ~0.01%, chapter 05 §2.1), so "inert" mostly
#   reflects a negligible λ;
# - wherever the term matters, the cone share drops (the curve above);
# - the analysis weight was 2× too large for most modes (fixed 2026-10-07; the λ sweep above
#   is the corrected one, so its values differ from the old log).
#
# The Tier-0 fix to the *convention* is right. The *conclusion* drawn from it is revised in
# chapter 05.

# %% [markdown]
# ## Fix 3 — the far-field band was chosen by array index
#
# **What the old code did** (`scripts/03_compute_sources.py`, Stage 3, never run): to
# compute the far-field error it radiated "the first 24 positive frequencies", i.e. the
# 24 *lowest*.
#
# **Why that is wrong.** Radiation in the box needs ω ≥ c₀·k for some box mode, i.e.
# ω ≥ c₀·(2π/L) = 11.3 at M = 0.1. The 24 lowest frequencies are all far below that.
#
# **How it was found:** `branch_analysis.md` §2.1, by checking the selected frequencies
# against the radiation condition.

# %%
print(diff("scripts/03_compute_sources.py", grep="band")[:2000])

# %%
from src.acoustics.lighthill import source_timeseries
from src.acoustics.greens import source_spectrum, select_band, far_field_pressure, observer_ring

u, v, _, dt = spectra.load(ENV, "turb2d")
S = source_timeseries(u, v, grid); del u, v
s_hat, om = source_spectrum(S, dt); del S
dk = 2 * np.pi / grid.length
old_band = [i for i in range(1, len(om)) if om[i] > 0][:24]
radius = 30 * grid.length
truth_power = (np.abs(s_hat) ** 2).sum(axis=(1, 2))
new_band, meta = select_band(om, c0, grid, 24, mode="auto", power=truth_power, radius=radius)
n_rad_old = sum(om[i] >= c0 * dk for i in old_band)
n_rad_new = sum(om[i] >= c0 * dk for i in new_band)
below = truth_power[(om > 0) & (om < c0 * dk)].sum() / truth_power[om > 0].sum()
print(f"lowest frequency that can radiate in this box: c0*dk = {c0 * dk:.2f}")
print(f"old band: omega in [{om[old_band[0]]:.2f}, {om[old_band[-1]]:.2f}], {n_rad_old} of 24 can radiate")
print(f"new band: omega in [{om[new_band[0]]:.2f}, {om[new_band[-1]]:.2f}], {n_rad_new} of 24 can radiate; "
      f"kr from {meta['kr_range'][0]:.0f} to {meta['kr_range'][1]:.0f}")
print(f"{below:.0%} of the source's omega > 0 power lies below c0*dk")

obs = observer_ring(radius, 36, (grid.length / 2, grid.length / 2))
p_old = far_field_pressure(s_hat, om, grid, obs, c0, True, old_band)
p_new = far_field_pressure(s_hat, om, grid, obs, c0, True, new_band)
print(f"mean |p| over the band: old {np.abs(p_old).mean():.3e}, new {np.abs(p_new).mean():.3e} "
      f"(old/new = {np.abs(p_old).mean() / np.abs(p_new).mean():.1f}x)")

# %% [markdown]
# **Reading:**
# - The old band measured "radiated sound" error entirely at frequencies that cannot
#   radiate in this box. It also produced a much larger far-field pressure: 43× by the
#   measure above (mean |p| over band and observers). `TIER0_PLAN.md` reports 11× from its
#   smoke test, whose exact measure was not recorded.
# - Its size also scaled with record length, since longer records give lower frequencies.
# - The new band is chosen by the radiation condition, log-spaced in |k| from the first box
#   mode to the Nyquist limit. That makes it independent of record length, with kr in the
#   hundreds, where the far-field formula is accurate.
#
# **Revised again on 2026-10-01 (chapter 05).** The "needs a box mode" logic belongs to the
# S-form of the source. In the T-form, frequencies below c₀·dk radiate physically through a
# term the S-form cannot represent, so the band choice is to be revisited together with
# the switch to the T-form.

# %% [markdown]
# ## Earlier fixes (2026-09-15) — reconstructions
#
# These predate the first commit, so the old code cannot be run from git. Each cell rebuilds
# the old behaviour from the rerun log's description.
#
# **(a) The Green's function was incoming.** Shown in chapter 01, §4.
#
# **(b) The loss's wavenumber grid ignored the box size.** `losses._k_grid` used mode
# indices m directly as wavenumbers. Angular wavenumber is k = 2πm/L, so the two agree only
# for L = 2π. For the 4π box every k² was 4× too large.

# %%
import torch
from src.models.losses import _k_grid

n = 8
for L in (2 * np.pi, 4 * np.pi):
    k_now = _k_grid(n, n, "cpu", length=L)
    ky = torch.fft.fftfreq(n)[:, None] * n; kx = torch.fft.rfftfreq(n)[None, :] * n   # reconstruction: mode index
    k_then = kx**2 + ky**2
    print(f"L = {L / np.pi:.0f}π: k² at mode (1, 1): then {k_then[1, 1].item():.3f}, now {k_now[1, 1].item():.3f}  "
          f"-> ratio {k_then[1, 1].item() / k_now[1, 1].item():.2f}")

# %% [markdown]
# **(c) The Sobolev penalty was not normalised.** The weight (1 + k²) reaches ~2·64² at
# n = 128, so at `weight = 1` the penalty was ~10³ × the MSE and the objective was
# effectively pure H¹. The follow-up divided the weight by its mean over the rfft2 half-plane.
# The rerun log records penalty/MSE ≈ 2728 before and ≈ 0.9955 after, for white noise at 128².
# The cell draws its own white noise, so it agrees to within 1%, not exactly.

# %%
torch.manual_seed(0)
x = torch.randn(1, 2, 128, 128, dtype=torch.float64)
d_ft = torch.fft.rfft2(x, norm="ortho")
w = 1.0 + _k_grid(128, 128, "cpu")
before = (torch.mean(w * torch.abs(d_ft) ** 2) / torch.mean(x**2)).item()          # reconstruction
after = (torch.mean(w / w.mean() * torch.abs(d_ft) ** 2) / torch.mean(x**2)).item()  # current code path
Comparison("Sobolev normalisation (white noise, 128², weight = 1)").add(
    "penalty / MSE, before", old=before, logged=2728, check="old", rtol=0.01, source="rerun log B1").add(
    "penalty / MSE, after", new=after, logged=0.9955, check="new", rtol=0.01, source="rerun log B1").show()

# %% [markdown]
# ## Summary of the fixes
#
# | fix | when | found by | old → new (M = 0.1) | status |
# |---|---|---|---|---|
# | Green's function outgoing | 09-15 | independent audit | dφ/dr = +k → −k | done; gate CHECK 4 |
# | 2D vs 3D kernel guard | 09-15 | independent audit | (no number changed) | done; gate CHECK 3 |
# | loss wavenumbers in angular units | 09-15 | follow-up | k² 4× off at L = 4π → correct | done |
# | Sobolev penalty normalised | 09-15 | follow-up | penalty/MSE 2728 → ~1 | done |
# | cone mask on the true \|k\| | 09-30 | Tier 0 | R 8.69e-08 → 3.20e-08; R_S 1.03e-08 → 2.27e-09 | done |
# | Sobolev analysed as trained | 09-30 | branch_analysis §2.2 | 0.053× L2 → 1.000× L2 | done; conclusion revised (ch. 05) |
# | far-field band by radiation condition | 09-30 | branch_analysis §2.1 | 0/24 → 24/24 radiating | done; to revisit (ch. 05) |
#
# Every Tier-0 change moved the headline **down**, i.e. made the cone look even more
# starved. The qualitative claim got stronger, not weaker.
