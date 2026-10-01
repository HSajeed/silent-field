# %% [markdown]
# # 03 — The k–ω headline (Stage 2)
#
# **What you will learn:** how the (|k|, ω) energy spectrum is estimated; how the
# radiating-cone share R is computed from it; how uncertain R is; where in the plane the
# cone energy actually sits; and whether R is a property of the flow or of the box.
#
# **Inputs:** `turb2d.npz`; optionally `turb2d_kf16.npz` and `turb2d_L4pi.npz`.
# **Runtime:** ~10 min the first time (spectra are cached for chapters 04–05).
#
# **Old / new / future:**
# - every number computed here uses today's code (post-Tier-0);
# - pre-Tier-0 numbers appear only as quoted history, clearly labelled;
# - section 6 produces the first post-fix numbers for the larger box (HANDOFF open item 1).

# %% [bootstrap]

# %% [markdown]
# ## 1. Estimating the spectrum
#
# `src/acoustics/komega.py::komega_spectrum` turns a space–time record $u(t,x,y)$ into
# energy density on the (|k|, ω) plane:
#
# 1. **Remove the time mean**, then apply a **Hann window** in time. Without a window, the
#    huge convective ridge leaks across the plane into the cone and manufactures the very
#    signal being measured.
# 2. **3D FFT** (time, then both space directions).
# 3. **Normalise** with Parseval, $\sum|\hat u|^2/N^2 = \overline{u^2}$, corrected for the
#    window's power loss, so the spectrum sums to the variance.
# 4. **Fold** negative frequencies onto positive ones.
# 5. **Collapse 2D wavenumber to |k|** in two ways:
#    - radial *bins* centred on integer multiples of $2\pi/L$, for plotting;
#    - a *resolved* grid of every distinct |k| present, for arithmetic (the Tier-0 fix,
#      chapter 04).
#
# `welch_komega_spectrum` averages this over overlapping segments (4 segments, 50% overlap
# for the full record). That lowers the variance, at the cost of frequency resolution.

# %%
import json
import resource
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from src.spectral import Grid
from src.acoustics.komega import sound_speed
from src.acoustics.radiating import effective_weight_in_cone
from nbtools import spectra

MACHS = (0.05, 0.1, 0.2, 0.3)
LOGGED = json.loads((ENV.repo / "figures" / "komega_metrics.json").read_text())
LOG_ROWS = {r["mach"]: r for r in LOGGED["rows"]}


nbenv.verify_dataset(ENV, "turb2d")
base = spectra.spectra_for(ENV, "turb2d", halves=True)
U, grid = base["U"], base["grid"]
spec = base["vel_full"]
print(f"u_rms = {U:.4f};  spectrum {spec.power.shape} (omega x |k| bins), "
      f"{spec.k_resolved.size} resolved |k| values;  omega up to {spec.omega.max():.2f}")

# %% [markdown]
# ## 2. The picture
#
# Energy density on the (|k|, ω) plane, with the M = 0.1 sonic line (red) and the
# convective ridge ω = U|k| (dashed). The cone is the thin wedge above the red line.

# %%
m_plot = 0.1
c0 = sound_speed(U, m_plot)
P = np.maximum(spec.power, spec.power.max() * 1e-12)
fig, ax = plt.subplots(figsize=(7.5, 5.5))
pc = ax.pcolormesh(spec.k_bins, spec.omega, P, norm=LogNorm(), shading="auto")
ax.plot(spec.k_bins, spec.sonic_line(c0), color="red", lw=2, label=rf"sonic line $\omega=c_0|k|$ (M={m_plot})")
ax.plot(spec.k_bins, spec.convective_ridge(U), "w--", lw=2, label=r"convective ridge $\omega=U|k|$")
ax.set_xlim(spec.k_bins[1], spec.k_bins.max()); ax.set_ylim(spec.omega[1], spec.omega.max())
ax.set_xlabel("|k|"); ax.set_ylabel(r"$\omega$"); ax.legend(loc="upper right", fontsize=8)
fig.colorbar(pc, ax=ax, label=r"E(|k|, $\omega$)")
ax.set_title("Energy sits along the ridge; the cone is the sliver above the red line")
plt.show()

# %% [markdown]
# ## 3. The number: R, and R_S for the Lighthill source
#
# $$R(M)=\frac{\sum_{\text{cone}}E(|k|,\omega)}{\sum_{\text{all}}E(|k|,\omega)},\qquad
# \text{cone}=\{\omega\ge c_0|k|,\ |k|>0\},\quad c_0=U/M.$$
#
# - The flow is incompressible, so c₀ never enters the simulation. M is an analysis knob
#   that tilts the sonic line, and one dataset supports a whole Mach sweep.
# - **R_S** is the same share computed on the Lighthill source S rather than on velocity.
#   S is what radiates, so the audit argued R_S is the more direct number.
# - **Band:** each number is also computed on each half of the record. The envelope over
#   {full, first half, second half} is reported with its spread factor F = max/min.
#
# Every value is checked against `figures/komega_metrics.json`, written by
# `scripts/04_komega_analysis.py`.

# %%
tab = Comparison("baseline (L = 2π, forcing k = 8): cone share, post-Tier-0")
rows = {}
for M in MACHS:
    c0 = sound_speed(U, M)
    r = {p: effective_weight_in_cone(base[f"vel_{p}"], c0, "l2") for p in ("full", "half1", "half2")}
    s = {p: effective_weight_in_cone(base[f"src_{p}"], c0, "l2") for p in ("full", "half1", "half2")}
    rows[M] = (r, s)
    tab.add(f"R, M = {M}", new=r["full"], logged=LOG_ROWS[M]["l2"], check="new", rtol=1e-6,
            source="komega_metrics.json")
    tab.add(f"R_S, M = {M}", new=s["full"], logged=LOG_ROWS[M]["src_l2"], check="new", rtol=1e-6,
            source="komega_metrics.json")
tab.show()

md("| M | R | R band (F) | R_S | R_S band (F) |\n|---|---|---|---|---|\n" + "".join(
    f"| {M} | {r['full']:.2e} | [{min(r.values()):.1e}, {max(r.values()):.1e}] ({max(r.values()) / min(r.values()):.1f}) "
    f"| {s['full']:.2e} | [{min(s.values()):.1e}, {max(s.values()):.1e}] ({max(s.values()) / min(s.values()):.1f}) |\n"
    for M, (r, s) in rows.items()))

# %% [markdown]
# **Reading:**
# - At M = 0.1 about 3e-8 of the velocity energy, and 2e-9 of the source energy, lies in
#   the cone.
# - The band is wide (F ≈ 2–4). With only ~6 correlation times of record (chapter 02),
#   one realisation cannot pin R down better than a factor of a few.
# - The band covers finite-record sensitivity only, not realisation-to-realisation or
#   box-size uncertainty.

# %% [markdown]
# ## 4. How R depends on Mach
#
# A clean power law would have constant local exponents $d\ln R/d\ln M$. It does not.
#
# **History.** The 09-15 rerun first described this as "~M⁶–M⁷". The audit corrected it to
# "local exponents 10.7 / 7.1 / 5.7" (pre-Tier-0 numbers). A Gaussian random-sweeping model
# predicting erfc(1/M) was checked and failed by many orders of magnitude.
#
# **Correction (review 2026-10-01, chapter 05):**
# - the decision log's reason for retiring the M⁸ comparison ("compactness breaks at low
#   Mach") is backwards;
# - the 2D benchmark for radiated power is M⁷, not M⁸;
# - R is an energy share, not a radiated power, so it should not be compared with either
#   law directly.

# %%
def local_exponents(vals):
    ms = list(vals)
    return [np.log(vals[b] / vals[a]) / np.log(b / a) for a, b in zip(ms[:-1], ms[1:])]

eu = local_exponents({M: rows[M][0]["full"] for M in MACHS})
es = local_exponents({M: rows[M][1]["full"] for M in MACHS})
print("local exponents (0.05->0.1, 0.1->0.2, 0.2->0.3)")
print("  R   now:", ", ".join(f"{x:.2f}" for x in eu), "   (pre-Tier-0: 10.66, 7.10, 5.68)")
print("  R_S now:", ", ".join(f"{x:.2f}" for x in es), "   (pre-Tier-0: 8.65, 6.19, 5.65)")
plt.loglog(MACHS, [rows[M][0]["full"] for M in MACHS], "o-", color=style.NEW, label="R (velocity)")
plt.loglog(MACHS, [rows[M][1]["full"] for M in MACHS], "s-", color=style.OLD, label="R_S (source)")
plt.xlabel("M"); plt.ylabel("cone share"); plt.legend(); plt.title("cone share vs Mach (baseline)")
plt.show()

# %% [markdown]
# ## 5. Where the cone energy sits
#
# The audit's most consequential observation (pre-Tier-0): essentially **all** of the cone
# energy is in the lowest wavenumber shell the box allows. Recomputed with today's exact mask:

# %%
k = spec.k_resolved
lines = []
for M in MACHS:
    c0 = sound_speed(U, M)
    cone = (spec.power_resolved * spec.radiating_mask_resolved(c0)).sum(0)
    tot = cone.sum()
    top = np.argsort(cone)[::-1][:3]
    lines.append(f"| {M} | " + " | ".join(f"\\|k\\| = {k[i]:.2f}: {cone[i] / tot:.0%}" for i in top) + " |\n")
md("| M | largest | 2nd | 3rd |\n|---|---|---|---|\n" + "".join(lines))
print(f"the |k| = 1 shell holds {spec.power_resolved[:, np.isclose(k, 1)].sum() / spec.power_resolved.sum():.1%} "
      "of all the energy, but nearly all of the cone's")

# %% [markdown]
# **Why this matters:**
# - At M = 0.1 the cone is essentially "the high-frequency tail of the four |k| = 1 modes".
# - The acoustic wavelength at the energetic frequencies is longer than the box (L = 2π).
#   Most of the radiation the physics would produce sits at wavenumbers *between* 0 and the
#   first box mode, where the periodic grid has no modes at all.
# - So R depends on the box. Section 6 tests that directly.
# - Chapter 05 shows the deeper cause: the S-based source formulation drops the k = 0 term
#   that carries this low-wavenumber radiation.

# %% [markdown]
# ## 6. Is R a property of the flow or of the box?
#
# **History (rerun log B6–B7, 2026-09-15, all pre-Tier-0):**
#
# | M = 0.1 | baseline (L = 2π) | forcing k = 16 | L = 4π |
# |---|---|---|---|
# | R, all k | 8.69e-08 | 3.53e-07 | 7.53e-07 |
# | R, k ≥ 2 | 3.82e-10 | 5.68e-09 | 1.32e-10 |
# | R_S | 1.03e-08 | 7.01e-09 | 7.35e-09 |
#
# The conclusion at the time:
# - R(all k) is box-dependent (8.7× higher in the 4π box);
# - R over k ≥ 2 is roughly box-independent but depends on the forcing scale;
# - R_S was "the most robust number".
#
# After Tier 0 these numbers were not comparable to the corrected ones. HANDOFF listed
# regenerating them as blocked, because `scripts/04` stacks both velocity components and
# needs ~3.5 GB for n = 256. `nbtools.spectra` transforms one velocity component at a time.
# On the author's machine that peaks at ~2.7 GB for the 4π box, so the post-fix numbers can
# be computed here.

# %%
RUN_OTHER_BOXES = True
datasets = [n for n in ("turb2d", "turb2d_kf16", "turb2d_L4pi") if ENV.has(n)]
results = {"turb2d": base}
if RUN_OTHER_BOXES:
    for name in datasets[1:]:
        nbenv.verify_dataset(ENV, name)
        results[name] = spectra.spectra_for(ENV, name)
        print(f"{name}: done; peak memory so far {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB")


def cone_share(spec_, c0, k_min=0.0):
    mask = spec_.radiating_mask_resolved(c0) & (spec_.k_resolved[None, :] >= k_min)
    return float((spec_.power_resolved * mask).sum() / spec_.power_resolved.sum())


post = {}
for name, res in results.items():
    c0 = sound_speed(res["U"], 0.1)
    post[name] = {"R": cone_share(res["vel_full"], c0), "R_k2": cone_share(res["vel_full"], c0, 2.0),
                  "R_S": cone_share(res["src_full"], c0)}
cols = list(post)
md("**Post-Tier-0, M = 0.1** (exact cone mask; Welch 4 segments; c₀ = U/M with each dataset's own U)\n\n"
   "| measure | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n" +
   "".join(f"| {lab} | " + " | ".join(f"{post[c][key]:.2e}" for c in cols) + " |\n"
           for lab, key in [("R, all k", "R"), ("R, \\|k\\| ≥ 2", "R_k2"), ("R_S", "R_S")]))
if "turb2d_kf16" in post:
    Comparison("check against HANDOFF's post-Tier-0 kf16 column (rounded there to 2 s.f.)").add(
        "R, kf16, M = 0.1", new=post["turb2d_kf16"]["R"], logged=1.4e-07, check="new", rtol=0.05,
        source="HANDOFF.md table").add(
        "R_S, kf16, M = 0.1", new=post["turb2d_kf16"]["R_S"], logged=2.6e-09, check="new", rtol=0.05,
        source="HANDOFF.md table").show()
(ENV.cache / "post_tier0_boxes.json").write_text(json.dumps(post, indent=2))
if "turb2d_L4pi" in post:
    for key, lab in [("R", "R, all k"), ("R_k2", "R, |k| >= 2"), ("R_S", "R_S")]:
        vals = [post[c][key] for c in cols]
        print(f"{lab:<12} L4pi / baseline = {post['turb2d_L4pi'][key] / post['turb2d'][key]:.2f}x;  "
              f"spread over all set-ups (max/min) = {max(vals) / min(vals):.2f}x")

# %% [markdown]
# **Reading** *(computed 2026-10-01; the first post-fix numbers for the 4π box)*:
# - **R(all k) is still box-dependent after the fix.** It is ~7× higher in the 4π box at
#   M = 0.1 (8.7× pre-fix). Do not quote R(all k) as a flow property.
# - **R over |k| ≥ 2 differs by ~3× between the boxes,** and by more between forcing scales.
# - **R_S agrees to within ~1.4× across all three set-ups** (box size and forcing scale).
#   The pre-fix observation that R_S is the most robust number survives the Tier-0 fix.
# - **Chapter 05 explains the box dependence of R** by the source formulation, not the box
#   size as such: the box is ~12× the eddy size (chapter 02). It also shows that R_S, being
#   built on S, misses a term too, so "robust" is not the same as "physically complete".
#
# **One definition to keep in mind:** "|k| ≥ 2" is an absolute wavenumber. In the 4π box
# it removes the shells |k| = 0.5, 1, 1.5; in the 2π box only |k| = 1 and √2.

# %% [markdown]
# ## 7. Robustness checks done on 2026-09-15 (pre-Tier-0, quoted)
#
# These were run on the binned grid before Tier 0 and have not been re-run. They are kept
# because they rule out several artifacts.
#
# - **Window choice** (Hann / Blackman–Harris / DPSS, 4 segments): R varies < 2×. Halving
#   the Nyquist frequency barely moves it, so R is not an aliasing artifact.
# - **Noise floor:** float32 rounding noise sits ~13 orders of magnitude below the spectrum
#   at the M = 0.1 threshold.
# - **Method pitfalls found along the way:**
#   - per-bin power from estimators with different segment lengths is not comparable;
#     compare densities (P/dω);
#   - single-bin comparisons against a full-record periodogram (2 degrees of freedom) are
#     noise, so integrate instead;
#   - "still decaying" can be window leakage (Hann beyond ω ≈ 35).
# - **Still open:** re-running the noise-floor check post-Tier-0 (HANDOFF open item 3).

# %% [markdown]
# ## What this chapter established
# - **Headline at M = 0.1, post-Tier-0:** R ≈ 3.2e-8 and R_S ≈ 2.3e-9 of the energy lies in
#   the radiating cone, each with a band of a factor 2–4.
# - **The Mach dependence is not a power law**, and R should not be compared with
#   radiated-power laws.
# - **Nearly all of the cone energy is in the lowest box mode,** and R(all k) changes with
#   box size. That holds before and after the Tier-0 fix.
# - **Post-fix numbers for the kf16 and L = 4π boxes now exist.** HANDOFF open item 1 no
#   longer needs the chunking refactor; one component at a time is enough.
