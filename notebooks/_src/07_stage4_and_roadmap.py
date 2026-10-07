# %% [markdown]
# # 07 — Stage 4 and the roadmap (future; partly runnable)
#
# **What this chapter is for:** what remains, in what order, and how each step will be
# judged done.
#
# | part | status |
# |---|---|
# | Stage 4 λ sweep with the factor-2 correction (analysis) | runnable |
# | trained Sobolev runs, standard H¹ loss | **future**; one opt-in cell runs the repo's loss on GPU |
# | causal test (multi-step / temporal loss) | **future: placeholder** |
# | Rucz et al. 2024 absolute-amplitude gate | **future: placeholder** |
# | Mach as a continuous knob on a trained model | runnable once chapter 06 has a model |
# | a second, non-acoustic quantity of interest | **future: placeholder** |
# | the plan as a status table | below |

# %% [bootstrap]

# %%
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

from nbtools import spectra
from src.acoustics.komega import sound_speed
from src.acoustics.radiating import sobolev_halfplane_mean_weight, weighted_cone_share, effective_weight_in_cone

nbenv.verify_dataset(ENV, "turb2d")
base = spectra.spectra_for(ENV, "turb2d", source=False)
spec, grid, U = base["vel_full"], base["grid"], base["U"]
c0 = sound_speed(U, 0.1)

# %% [markdown]
# ## 1. Stage 4 analysis with the factor-2 correction
#
# Chapter 05 §2.2 showed that `additive_sobolev_weight` was 2× too large for every mode with
# kx ≠ 0 (fixed in the code on 2026-10-07). The cell below redoes the λ sweep with the per-mode factor n/(n+2) instead of
# n/(n/2+1).
#
# **One approximation, stated:** the resolved spectrum sums power over all modes with the
# same |k|, so kx = 0 modes (whose factor really is n/(n/2+1)) cannot be separated. They
# are a small minority of modes, so applying n/(n+2) to all is accurate to a few percent in λ.

# %%
k = spec.k_resolved
mask = spec.radiating_mask_resolved(c0)
w_norm = (1 + k**2) / sobolev_halfplane_mean_weight(grid)
lams = np.logspace(-3, 6, 28)
l2 = weighted_cone_share(spec.power_resolved, np.ones_like(k)[None, :], mask)
as_coded = [weighted_cone_share(spec.power_resolved, (1 + l * grid.n / (grid.n // 2 + 1) * w_norm)[None, :], mask) / l2 for l in lams]
corrected = [weighted_cone_share(spec.power_resolved, (1 + l * grid.n / (grid.n + 2) * w_norm)[None, :], mask) / l2 for l in lams]
mult = effective_weight_in_cone(spec, c0, "h1", convention="multiplicative") / l2
plt.semilogx(lams, as_coded, color=style.OLD, label="additive, as coded at the review (2× too strong)")
plt.semilogx(lams, corrected, color=style.NEW, label="additive, corrected (= current code)")
plt.axhline(mult, color=style.REF, ls="--", label=f"multiplicative (1+k²): {mult:.3f}× (neuraloperator H1Loss form)")
plt.axvline(1e-2, color=style.REF, ls=":", label="repo default λ = 1e-2")
plt.xlabel("λ"); plt.ylabel("cone share ÷ L2 cone share"); plt.legend(fontsize=8); plt.title("Stage 4 at M = 0.1")
plt.show()
print("The curve can only go down: the Sobolev weight is >= 1 everywhere and ~1 in the cone.")

# %% [markdown]
# **The Stage-4 claim, as the review restates it:** Sobolev weighting can never raise the
# cone share, and wherever it is strong enough to change training it lowers it, by up to
# ~19× in the standard (multiplicative) form.

# %% [markdown]
# ## 2. Trained Sobolev runs (future; one opt-in cell)
#
# What remains:
# 1. Add the **standard H¹ loss** (‖e‖² + ‖∇e‖², as in neuraloperator's `H1Loss`) to
#    `src/models/losses.py`. It is not in the repo yet. **[future]**
# 2. Train FNO and UNet with it and with MSE; compare the cone share of their *errors* (B,
#    chapter 06). **[future]**
#
# The opt-in cell below trains with the repo's existing additive `sobolev_loss` at a chosen
# λ, so the machinery can be tried now. It is off by default (about an hour per run on a T4).

# %%
RUN_SOBOLEV_TRAINING = False
SOBOLEV_LAMBDA = 1.0
if RUN_SOBOLEV_TRAINING and torch.cuda.is_available():
    from src.models.fno import FNO2d
    from src.models.train import TrainConfig, make_pairs, train
    u, v, grid_, dt = spectra.load(ENV, "turb2d")
    x, y, stats = make_pairs(u, v)
    model, hist = train(FNO2d(modes=min(24, grid_.n // 4), width=48), x, y,
                        TrainConfig(loss="sobolev", sobolev_weight=SOBOLEV_LAMBDA, domain_length=grid_.length))
    torch.save({"state_dict": model.state_dict(), "arch": "fno", "loss": f"sobolev_{SOBOLEV_LAMBDA:g}",
                "stats": stats, "history": hist, "n": grid_.n}, ENV.cache / f"model_fno_sobolev_{SOBOLEV_LAMBDA:g}.pt")
else:
    print("not run (set RUN_SOBOLEV_TRAINING = True on a GPU runtime)")

# %% [markdown]
# ## 3. The causal test (future)
#
# **Question:** if the failure in the cone is caused by the loss, does changing the loss
# change it?
#
# The review's suggestion is the cheapest intervention that targets the mechanism: the
# one-step loss has no time axis (chapter 05 §4.2), so give it one.
# - **Multi-step loss:** train on short rollouts (2–4 steps) so step-to-step coherence is
#   penalised; or
# - **Temporal-derivative loss:** add ‖∂ₜu_pred − ∂ₜu_true‖² estimated across consecutive
#   predictions.
#
# **Pass criterion:** B and B_S (chapter 06) drop relative to the MSE model, while the field
# error stays comparable. That would confirm the mechanism. It is deliberately not the open
# question at the end of the README (a cone-weighted loss), which remains future work.

# %%
def temporal_loss(*args, **kwargs):
    """FUTURE: multi-step / temporal-derivative objective for the causal test."""
    raise NotImplementedError

print("FUTURE: causal test not implemented")

# %% [markdown]
# ## 4. Absolute-amplitude gate: Rucz et al. 2024 (future)
#
# The gate (chapter 01) checks the shape of the radiated field, not its amplitude. Rucz,
# Ulveczki, Heinz & Schoder (J. Sound Vib. 587, 118496, 2024; open access) derive the
# co-rotating vortex pair's far-field pressure *ab initio*, with error bounds against Mach
# number, core radius and source truncation. Their companion data are on Zenodo ("VORTEX 01")
# *[not yet checked]*. The plan: reproduce their amplitude for this repo's vortex pair
# within their stated bounds, as gate CHECK 5.

# %%
print("FUTURE: absolute-amplitude gate check not implemented")

# %% [markdown]
# ## 5. Mach as a continuous knob (runnable once a model exists)
#
# c₀ never enters the simulation, so one trained model can be examined at every Mach
# number. If chapter 06 has saved B for the MSE model, plot it against M.

# %%
b_path = ENV.cache / "B_fno_teacher_forced.json"
if b_path.exists():
    rows = json.loads(b_path.read_text())
    plt.loglog([r[0] for r in rows], [r[3] for r in rows], "o-", color=style.NEW, label="B (velocity)")
    plt.loglog([r[0] for r in rows], [r[6] for r in rows], "s-", color=style.OLD, label="B_S (source, S-form)")
    plt.axhline(1, color=style.REF, ls=":"); plt.xlabel("M"); plt.ylabel("blindness ratio"); plt.legend()
    plt.title("preliminary: teacher-forced, before the controls"); plt.show()
else:
    print("no B results yet -- run chapter 06 on a GPU first")

# %% [markdown]
# ## 6. A second quantity of interest (future)
#
# The structure "a quantity that matters but carries almost no energy" is not specific to
# acoustics: trace species in combustion, eigenmode amplitudes, wall-pressure fluctuations.
# Applying B to one non-acoustic case would turn a niche result into a general one. Planned
# for after Stages 3–4.

# %% [markdown]
# ## 7. The plan, as a status table
#
# From `review_2026-10-01.md` §5, updated with what these notebooks have already done.

# %%
post = ENV.cache / "post_tier0_boxes.json"
PLAN = [
    ("1a", "Gate check: S-form = T-form on the vortex pair at its tone",
     "agreement measured (ch. 01 §6: ~2% magnitude, ~6% complex); tolerance + gate code pending", "in progress"),
    ("1b", "Switch scripts/03 to far_field_pressure_ibp; revisit the band", "T-form error used in ch. 06", "done 2026-10-07 (scripts/03 uses the T-form); band revisit pending"),
    ("1c", "Define the radiating content on T_ij (replaces R_S)", "derivation + test", "todo"),
    ("1d", "Post-fix numbers for kf16 and L = 4π (HANDOFF open item 1)",
     "done in ch. 03 §6 without chunking" if post.exists() else "ch. 03 §6", "done" if post.exists() else "todo"),
    ("2", "Fix the factor 2 in additive_sobolev_weight; non-circular test; restate Stage 4",
     "test calls sobolev_loss on single modes", "done 2026-10-07 (weight fixed); non-circular test + restated Stage 4 pending"),
    ("3", "Mach scaling: drop the compactness rationale; compare T-form radiated power with M⁷", "—", "todo"),
    ("4", "Stage 3: train FNO + UNet; error triple; B and B_S with both controls", "ch. 06", "partly runnable"),
    ("5", "Causal test: multi-step / temporal loss", "B and B_S drop", "todo"),
    ("6", "Rucz gate; Mach-knob figure; noise-floor re-run; manifest records spin-up", "—", "todo"),
]
md("| step | what | acceptance / evidence | status |\n|---|---|---|---|\n" +
   "".join(f"| {a} | {b} | {c} | {d} |\n" for a, b, c, d in PLAN))

# %% [markdown]
# ## Where to go from here
# - The audit trail: `rerun_log_2026-09-15.md` → `branch_analysis.md` → `TIER0_PLAN.md` →
#   `review_2026-10-01.md`.
# - The next code change is step 1a/1b. Its result decides how chapter 06's far-field numbers
#   should be read.
# - As each future step lands, its placeholder here or in chapter 06 is replaced by a
#   runnable cell and this table is updated.
