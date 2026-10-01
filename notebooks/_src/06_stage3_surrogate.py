# %% [markdown]
# # 06 — Stage 3: the surrogate demonstration (future; partly runnable)
#
# **What this chapter is for:** Stage 3 trains a deliberately standard surrogate and
# measures what it gets wrong. It has **never been run**. Parts of it can run now; the rest
# is laid out as clearly marked placeholders, with the plan written into them.
#
# | part | status |
# |---|---|
# | train FNO / UNet with one-step MSE | runnable (GPU strongly recommended) |
# | field error, source error, spurious divergence | runnable once a model exists |
# | far-field error, **S-form vs T-form** side by side | runnable, *preliminary* (see chapter 05) |
# | blindness ratio B and B_S, teacher-forced | runnable, *preliminary* |
# | the two controls for B (rollout; matched random error) | **future: placeholder** |
#
# **Colab:** Runtime → Change runtime type → **T4 GPU**. The build plan estimates ~1 h of
# training on a T4 *[unverified]*; the cell below times it. The checkpoint is saved to
# Drive, so later sessions reload it instead of retraining.

# %% [bootstrap]

# %%
import json, time
import numpy as np
import matplotlib.pyplot as plt
import torch

from nbtools import spectra
from src.models.fno import FNO2d
from src.models.unet import UNet2d
from src.models.train import TrainConfig, make_pairs, train, predict_onestep, denormalise

ARCH = "fno"            # "fno" or "unet"
EPOCHS = 60             # the script default
HAS_GPU = torch.cuda.is_available()
RUN_TRAINING = HAS_GPU  # set True to train on CPU anyway (many hours at 128²)
CKPT = ENV.cache / f"model_{ARCH}_mse.pt"

nbenv.verify_dataset(ENV, "turb2d")
u, v, grid, dt = spectra.load(ENV, "turb2d")
print(f"GPU available: {HAS_GPU};  checkpoint {'found' if CKPT.exists() else 'not found'} at {CKPT}")

# %% [markdown]
# ## 1. Train (or reload) a vanilla surrogate
#
# `src/models/train.py` is deliberately standard: one-step pairs (snapshot t → t+1),
# normalised, MSE, AdamW, cosine schedule. The claim is about the *ordinary* recipe, so
# nothing here is tuned for acoustics.

# %%
def build_model(arch, n):
    return FNO2d(modes=min(24, n // 4), width=48) if arch == "fno" else UNet2d(base=48, depth=3)

model, stats, history = None, None, None
if CKPT.exists():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = build_model(ck["arch"], grid.n); model.load_state_dict(ck["state_dict"])
    stats, history = ck["stats"], ck.get("history")
    print(f"reloaded {ck['arch']} trained for {len(history or [])} epochs")
elif RUN_TRAINING:
    x, y, stats = make_pairs(u, v)
    model = build_model(ARCH, grid.n)
    print(f"{len(x)} one-step pairs; {ARCH.upper()} with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    t0 = time.time()
    model, history = train(model, x, y, TrainConfig(epochs=EPOCHS, loss="mse", domain_length=grid.length))
    print(f"training took {(time.time() - t0) / 60:.1f} min")
    torch.save({"state_dict": model.state_dict(), "arch": ARCH, "loss": "mse", "stats": stats,
                "history": history, "n": grid.n}, CKPT)
    del x, y
else:
    print("SKIPPED: no checkpoint and no GPU. In Colab, switch to a GPU runtime and re-run.")

if history:
    plt.semilogy([h["train"] for h in history], label="train"); plt.semilogy([h["val"] for h in history], label="val")
    plt.xlabel("epoch"); plt.ylabel("MSE (normalised)"); plt.legend(); plt.show()

# %% [markdown]
# ## 2. The error triple, first two parts
#
# Predictions are **teacher-forced**: every step starts from the true snapshot. That
# isolates the objective from rollout drift (`predict_onestep`'s docstring explains why).

# %%
from src.diagnostics.errors import error_triple
from src.diagnostics.divergence import divergence_error, dilatational_energy_fraction

if model is not None:
    x, _, _ = make_pairs(u, v)
    pred = denormalise(predict_onestep(model, x, TrainConfig()), stats).astype(np.float64); del x
    up, vp = pred[:, 0], pred[:, 1]; del pred
    ut, vt = u[1:], v[1:]
    trip = error_triple(up, vp, ut, vt, grid)
    print(f"relative field error           {trip['field']:.2%}   (the number papers report)")
    print(f"relative Lighthill source error {trip['source']:.2%}")
    print(f"spurious dilatational energy    {np.mean([dilatational_energy_fraction(up[i], vp[i], grid) for i in range(0, len(up), 8)]):.2e}")
    print(f"div(u) error: truth {np.mean([divergence_error(ut[i], vt[i], grid) for i in range(0, len(ut), 8)]):.2e}, "
          f"prediction {np.mean([divergence_error(up[i], vp[i], grid) for i in range(0, len(up), 8)]):.2e}")
else:
    print("SKIPPED: needs a model (section 1)")

# %% [markdown]
# ## 3. Far-field error — S-form and T-form side by side (preliminary)
#
# > **Not the publishable number yet.** Chapter 05 showed the S-form far field (what
# > `scripts/03` computes) is mostly a boundary artifact. Until the T-form passes the gate
# > check proposed in chapter 01 §6, both are reported. The comparison shows how much of an
# > S-form error is artifact.
#
# Two bands:
# - the Tier-0 band (ω ≥ c₀·dk, chosen by `select_band`);
# - a low band below c₀·dk. There the S-form has no physical content, and the T-form
#   carries the compact radiation.

# %%
from src.acoustics.lighthill import source_timeseries
from src.acoustics.greens import (source_spectrum, select_band, far_field_pressure,
                                  far_field_pressure_ibp, observer_ring)
from src.acoustics.komega import sound_speed

def far_fields(uu, vv, idx, obs, c0):
    S = source_timeseries(uu, vv, grid); sh, om = source_spectrum(S, dt); del S
    th = {k: source_spectrum(a, dt)[0] for k, a in (("uu", uu * uu), ("uv", uu * vv), ("vv", vv * vv))}
    return far_field_pressure(sh, om, grid, obs, c0, True, idx), far_field_pressure_ibp(th, om, grid, obs, c0, idx), sh, om

if model is not None:
    c0 = sound_speed(float(np.sqrt(np.mean(ut**2 + vt**2))), 0.1)
    obs = observer_ring(30 * grid.length, 36, (grid.length / 2, grid.length / 2))
    S = source_timeseries(ut, vt, grid); sh_t, om = source_spectrum(S, dt); del S
    band_hi, _ = select_band(om, c0, grid, 24, mode="auto", power=(np.abs(sh_t) ** 2).sum(axis=(1, 2)),
                             radius=30 * grid.length)
    band_lo = [i for i in range(1, len(om)) if om[i] < c0 * 2 * np.pi / grid.length][::4][:24]
    rel = lambda a, b: float(np.linalg.norm(np.abs(a) - np.abs(b)) / np.linalg.norm(np.abs(b)))
    rows = []
    for label, band in (("Tier-0 band, ω ≥ c0·dk", band_hi), ("low band, ω < c0·dk", band_lo)):
        pS_t, pT_t, _, _ = far_fields(ut, vt, band, obs, c0)
        pS_p, pT_p, _, _ = far_fields(up, vp, band, obs, c0)
        rows.append((label, rel(pS_p, pS_t), rel(pT_p, pT_t)))
    md("| band | far-field error, S-form | far-field error, T-form |\n|---|---|---|\n" +
       "".join(f"| {a} | {b:.1%} | {c:.1%} |\n" for a, b, c in rows))
else:
    print("SKIPPED: needs a model (section 1)")

# %% [markdown]
# ## 4. The blindness ratio B (preliminary, teacher-forced)
#
# $$B(M)=\frac{R_{loss}}{R_{signal}}=\frac{\text{cone share of the error } \hat e=\hat u_{pred}-\hat u_{true}}
# {\text{cone share of the truth } \hat u_{true}}.$$
#
# - **B = 1:** the model's error is spread like the signal.
# - **B ≫ 1:** the model fails disproportionately in the radiating cone.
# - **Reminder (chapter 05):** B ≥ 1 is not guaranteed.
#
# **B_S** is the same ratio on the Lighthill source. It is computed here on S. It should move
# to the T-form once that metric exists.

# %%
from src.acoustics.radiating import effective_weight_in_cone

if model is not None:
    U = float(np.sqrt(np.mean(ut**2 + vt**2)))
    sig = spectra.velocity_spectrum(ut, vt, grid, dt, 4)
    err = spectra.velocity_spectrum(up - ut, vp - vt, grid, dt, 4)
    S_t = source_timeseries(ut, vt, grid); S_p = source_timeseries(up, vp, grid)
    from src.acoustics.komega import welch_komega_spectrum
    sig_S = welch_komega_spectrum(S_t, grid, dt, n_segments=4)
    err_S = welch_komega_spectrum(S_p - S_t, grid, dt, n_segments=4); del S_t, S_p
    rows = []
    for M in (0.05, 0.1, 0.2, 0.3):
        c0 = sound_speed(U, M)
        rs, rl = effective_weight_in_cone(sig, c0, "l2"), effective_weight_in_cone(err, c0, "l2")
        ss, sl = effective_weight_in_cone(sig_S, c0, "l2"), effective_weight_in_cone(err_S, c0, "l2")
        rows.append((M, rs, rl, rl / rs, ss, sl, sl / ss))
    md("| M | R_signal | R_loss | **B** | R_S signal | R_S error | **B_S** |\n|---|---|---|---|---|---|---|\n" +
       "".join(f"| {r[0]} | {r[1]:.2e} | {r[2]:.2e} | **{r[3]:.3g}** | {r[4]:.2e} | {r[5]:.2e} | **{r[6]:.3g}** |\n"
               for r in rows))
    (ENV.cache / f"B_{ARCH}_teacher_forced.json").write_text(json.dumps(rows))
else:
    print("SKIPPED: needs a model (section 1)")

# %% [markdown]
# ## 5. FUTURE — the controls that decide what B means (not implemented)
#
# A large B could come from either of two mechanisms:
# - **energy weighting** (the project's thesis): the loss neglects the cone because the
#   cone holds little energy;
# - **temporal jitter** [unverified hypothesis]: the one-step loss has no time axis, so
#   step-to-step incoherence in the predictions lands in the high-ω part of the plane,
#   which at |k| = 1 *is* the cone.
#
# Two controls separate them:
#
# 1. **Free rollout vs teacher forcing.** Roll the model forward from one true snapshot and
#    recompute B on the rollout. Jitter behaves differently under rollout (errors
#    accumulate coherently) than under teacher forcing.
# 2. **Matched random-error baseline.** Replace the model's error with random noise of the
#    same per-snapshot spatial spectrum but random phases in time. Compute its B. If the
#    model's B is close to the baseline's, B measures temporal incoherence, not
#    energy-weighted neglect.
#
# Both are a few dozen lines each. They are left unimplemented until the T-form metric
# (chapter 05) is settled, so B is computed once, on the right quantity.

# %%
def blindness_ratio_controls(*args, **kwargs):
    """FUTURE: rollout B and matched-random-error B. See the cell above for the plan."""
    raise NotImplementedError("Stage 3 controls are not implemented yet -- see review_2026-10-01.md §4.2")

print("FUTURE: blindness_ratio_controls() is a placeholder (not called)")

# %% [markdown]
# ## What this chapter will establish (once complete)
# - The error triple for a standard surrogate, with the far field on the T-form.
# - B and B_S, and, through the two controls, whether a large B reflects energy weighting
#   or temporal jitter.
# - That result decides the paper's framing: "amplitude-based blindness" or "the loss has
#   no time axis".
