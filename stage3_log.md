# Stage 3 log — surrogate training and acoustic evaluation (2026-10-07)

Detailed record of every Stage-3 run, moved out of `HANDOFF.md` (which keeps the summary).
All runs: Colab T4, data `data/turb2d.npz` (sha256-verified), outputs on Drive under
`MyDrive/silent-field/runs/<run>/` (`model.pt`, `history.json`, `03_errors.json`,
`09_blindness*.json`, `diag/10_error_diagnostic.json` + PNGs). Workflow: `colabmcp.md`
(local, untracked). Literature context: `lit_review_2026-10-07.md`.

## Runs on Drive

| run | model / loss | seed | one-step val MSE | notes |
|---|---|---|---|---|
| `fno_mse_01` | FNO, MSE | 0 (init unseeded) | 2.17e-3 | + `eval_v2/`, `band_k/`, `diag/`, `machs/`, `09_blindness_machs.json` |
| `fno_mse_s1` | FNO, MSE | 1 | 2.30e-3 | 03/09/10 |
| `fno_mse_s2` | FNO, MSE | 2 | 2.52e-3 | 03/09/10 |
| `unet_mse_01` | UNet, MSE | 0 (init unseeded) | 9.2e-4 | 03/09/10, Mach sweep |
| `fno_h1_01` | FNO, relative H¹ | 0 (init unseeded) | 5.48e-3 (best 2.03e-3) | 03/09/10 |
| `fno_mse_unroll4` | FNO, MSE unrolled ×4 | 0 (init unseeded) | 3.18e-3 | 03/09/10 |

## Chronological record

**Stage 3:** FNO/UNet training (`scripts/02`, `03`). Not started, but Colab-ready
(2026-10-07): both scripts take `--smoke` and `--out <run dir>`; `02` checks the dataset
sha256, checkpoints every epoch to `--ckpt-dir` (`/content/ckpt` on Colab), copies to
`--out` every `--save-every` epochs, resumes from `<out>/last.pt`, and both end with a
SUMMARY block. Smoke runs pass locally. The workflow and runner cell are in `colabmcp.md`
(local, untracked); Drive layout as the notebooks (`MyDrive/silent-field/{data,runs}`).
**First run 2026-10-07 (Colab T4, code `bca1ed7`), `MyDrive/silent-field/runs/fno_mse_01/`:**
FNO 10.63M params, one-step MSE, 60 epochs in 6.7 min; train 7.05e-04, val 2.17e-03
(normalised). Teacher-forced `03` at M=0.1: field 3.04%, source 28.60%, **far field
(T-form) |p| 2.02%, complex 2.71%** (22 freqs, ω 0.154–78.5, 14 below c0·dk, 49% of the
stress's ω>0 power); div(u) error 0.182 (truth 9.2e-07); dilatational energy 5.35e-05.
**Copy-the-input baseline** (predict t+1 = t; same metrics, local) and the box-mode band
(`--band-mode k`, ω ≥ c0·dk = 11.3, 0.28% of the stress power; `runs/fno_mse_01/band_k/`):

| error | FNO | copy input | FNO gain |
|---|---|---|---|
| field | 3.04% | 9.63% | 3.2× |
| Lighthill source | 28.60% | 31.55% | **1.1×** |
| far field \|p\|, ω band | 2.02% | 0.20% | 0.1× |
| far field complex, ω band | 2.71% | 18.90% | 7.0× |
| far field \|p\|, k band | 11.43% | 0.13% | 0.01× |
| far field complex, k band | 15.47% | 52.62% | 3.4× |

Readings: (1) **\|p\| is not a usable far-field metric** — magnitudes ignore timing, so
doing nothing scores best; report the complex error. (2) The ω-band far field is
dominated by the low-ω compact term, which teacher forcing nearly hands to the model.
(3) In the box-mode radiating band the complex error is 15.5%, **5× the field error**,
but the FNO's gain over copying there (3.4×) equals its gain on the field (3.2×): harder,
not selectively neglected — no B-type evidence yet. (4) **Training improves the source
only 1.1× over copying** while improving the field 3.2×: the clearest support so far,
at source level.
**Built 2026-10-07 (smoke-tested, not yet run on `fno_mse_01`):** `03` now reports the
complex error as primary, the copy-input baseline beside every number (`--no-baseline`
to skip) and per-frequency far-field error with below/above-c0·dk aggregates.
`scripts/09_blindness.py` computes B = R(error)/R(truth) and B_S = R_T(error)/R_T(truth)
(`tform.cone_share_refined(..., minus=)`), plus phase-free ratios R, R_T, P_rad of
prediction/truth, for five variants: FNO teacher-forced, FNO free rollout
(`train.predict_rollout`), copy-input, matched solenoidal white noise, and the model's own
error time-shuffled (the jitter test of review §4.2). Smoke (untrained model) already
shows matched white B ~6e3 and shuffled B ~5e5: any temporally incoherent error lands in
the cone, so B must be read against these controls.
**Run on `fno_mse_01` (Colab, `86fff8f`; `runs/fno_mse_01/eval_v2/03_errors.json`,
`runs/fno_mse_01/09_blindness.json`):**
- Per-frequency far-field complex error (teacher-forced, FNO vs copy-input): 0.3–1.0% vs
  0.7–28% for ω < c0·dk; 7.2% vs 47% at ω=11.8; 63% vs 79% at 20.3; **263% vs 128% at 34.9;
  678% vs 186% at 59.9** — above ω≈30 the model radiates more than the true flow and is
  worse than copying. Aggregates: 0.86% below c0·dk, 18.97% above.
- `09` (M=0.1): B / B_S / R_pred/R_truth / R_T ratio / P_rad ratio:
  fno_tf 4.71e3 / 9.24 / 5.92 / 1.64 / 1.00; fno_rollout (1023 steps, all finite, field
  err 136%) 19 / 1.41 / **34.9** / 1.75 / 0.776; copy_input 21.5 / 3.16 / 0.994 / 0.997 /
  0.999; white_matched 9.46e4 / 6.08e3 / 98 / 98.8 / 1.13; shuffled_error 5.95e5 / 6.64e3
  / 705 / 511 / 1.68.
- Reading (interpretation, not proven): the model's error is far more cone-concentrated
  than the signal (B=4.7e3) but 20–130× *less* than an equally large temporally incoherent
  error, and far more than a coherent one (copy-input 21.5). If B mixes linearly, ≈0.8% of
  the error energy behaves like step-to-step jitter, and that jitter carries the cone
  content — review §4.2's jitter hypothesis is supported. The MSE barely sees that jitter
  (it is ~1% of a 3% error), yet it puts spurious supersonic content into the prediction
  (R 6× truth teacher-forced, 35× in rollout) and spurious high-frequency sound. Total
  radiated power stays right (compact, low-ω term). Next: UNet, Sobolev (standard H¹),
  multi-step loss; test whether a temporal-derivative/multi-step objective lowers B.
**Built (smoke-tested):** `scripts/10_error_diagnostic.py` — (k, ω) error map, per-|k|-shell
lag-1 temporal correlation of error vs truth, error share in the cone-k range and in the
aliasing band (|k| > 2/3 k_max), low-k ω-spectrum of error vs truth; model vs copy-input.
`09` variants renamed `model_tf`/`model_rollout` (the fno_mse_01 JSON still says `fno_*`).
Plan agreed 2026-10-07: run `10` on FNO + train/evaluate UNet; then FNO with standard H¹ and
with an unrolled multi-step loss; later noise injection / PDE-Refiner.
**Experiments 1–2 done (Colab T4, `63c2bba`).** UNet `runs/unet_mse_01/` (4.33M params,
60 ep in 12.1 min, val 9.2e-04, best 7.6e-04). FNO | UNet (copy-input):

| measure (M=0.1, teacher-forced unless noted) | FNO | UNet | copy |
|---|---|---|---|
| field / source error | 3.04% / 28.6% | 1.60% / 17.8% | 9.63% / 31.6% |
| far field complex: all / below / above c0·dk | 2.71 / 0.86 / 19.0% | 2.50 / 2.02 / 11.0% | 18.9 / 17.7 / 52.2% |
| far field complex at ω=34.9 / 59.9 | 263% / 678% | 132% / 388% | 128% / 186% |
| B / B_S | 4.71e3 / 9.24 | 3.93e3 / 6.97 | 21.5 / 3.16 |
| controls B: matched white / shuffled | 9.46e4 / 5.95e5 | 9.49e4 / 5.02e6 | |
| R_pred/R_truth: teacher-forced / rollout | 5.92 / 34.9 | 1.91 / 3.72 | 0.994 |
| radiated power ratio, rollout | 0.776 | 0.714 | |
| low-k error PSD flatness, high-ω/mid-ω (truth 2.4e-9) | 1.4e-3 | 2.1e-4 | 1.2e-7 |
| low-k error ÷ truth: high ω / mid ω | 191 / 3.2e-4 | 17.9 / 2.0e-4 | 2.2 / 4.5e-2 |
| cone error from |k|=1 shell | 39% | 58% | |
| aliasing band (|k|>42.7): error share / lag-1 corr (truth 0.64) | 17% / 0.29 | 26% / 0.48 | 2% / 0.11 |

Reading (interpretation): **both architectures show the same signature**, so it is the
objective, not the FNO. The cone error sits in the lowest shells with a near-white ω
spectrum that exceeds the true signal at high ω (191× FNO, 18× UNet) while being
~1e-4 of it at mid ω — the jitter fingerprint. The aliasing-band error is itself
incoherent (not deterministic folding) and is not where the cone error lives, so
aliasing is not the explanation. B barely moves between models (4.7e3 → 3.9e3) although
the field error halves: B characterises the objective, not the model quality. Caveat:
the energy-weighted lag-1 correlation is dominated by low-ω error and is not decisive
on its own. Next: FNO with standard H¹ (expect no B change) and with an unrolled
multi-step loss (jitter account predicts B and the high-ω far-field excess drop).
**Built for experiments 3–4 (smoke-tested):** `--loss h1` = standard relative H¹
(`losses.h1_loss`, neuraloperator-style ||e||_H1/||y||_H1, multiplicative (1+k²), full-plane
fft2); `--unroll K` = backprop through K autoregressive steps (`train.make_windows`; unroll=1
reproduces one-step training exactly — test). Validation stays one-step MSE so runs compare.
Runs: `fno_h1_01` (`--loss h1`), `fno_mse_unroll4` (`--unroll 4`, ~4× the FNO time).
**Experiments 3–4 done (Colab T4, `3682580`).** `fno_h1_01` 7.0 min, one-step val MSE
5.48e-03 final (best 2.03e-03: H¹ trades L2 accuracy away as it trains); `fno_mse_unroll4`
22.7 min, one-step val 3.18e-03. Teacher-forced unless noted, M=0.1:

| measure | FNO mse | FNO **h1** | FNO mse **unroll4** | UNet mse |
|---|---|---|---|---|
| field / source error | 3.04 / 28.6% | 3.23 / 44.8% | 3.89 / 35.7% | 1.60 / 17.8% |
| far field complex, ω ≥ c0·dk | 19.0% | 60.6% | 21.0% | 11.0% |
| far field at ω=34.9 / 59.9 | 263 / 678% | 911 / 2617% | 311 / 802% | 132 / 388% |
| B / B_S | 4.71e3 / 9.24 | **1.08e5** / 68.2 | 3.70e3 / 9.34 | 3.93e3 / 6.97 |
| low-k error ÷ truth at high ω | 191 | 2720 | 349 | 17.9 |
| rollout: R/R0 / R_T/R_T0 / P/P0 | 34.9 / 1.75 / 0.78 | 1340 / 9.88 / 2.42 | 13.1 / 1.08 / 0.44 | 3.72 / 0.99 / 0.71 |

Reading (interpretation; single seed each — seed variance not yet measured):
- **The standard H¹ remedy makes the acoustics much worse** (B 23×, high-ω far field 3–4×,
  rollout radiates 2.4× the true power) at an unchanged field error. Matches PDE-Refiner
  (Sobolev hurt there) and `scripts/06` (multiplicative H¹ moves weight off the cone).
- **Unrolling (4 steps, backprop through all) does not fix it.** Teacher-forced B −21%,
  high-ω excess slightly *worse*; it helps rollout statistics (spurious cone share 35×→13×,
  R_T ratio 1.75→1.08) but under-radiates (P 0.44). The jitter-account prediction
  ("unrolling removes the jitter") is **not confirmed**.
- **B tracks the objective's weighting, not the model or the training horizon:** ~4e3 for
  every MSE variant (FNO, UNet, unrolled), 1.1e5 under H¹. An unrolled MSE is still
  amplitude-weighted, so the low-amplitude, high-ω jitter stays invisible to it. The lever
  is the weighting in ω, i.e. the README's open question (a cone/temporal-frequency-
  weighted objective over a time window), with noise injection and PDE-Refiner as the
  literature baselines.
**Seeding (2026-10-07):** runs up to `f261e06` built the model before seeding, so their
weight initialisation was unseeded (batch order was seeded, seed 0). `scripts/02 --seed N`
now seeds both; two `--seed 1` smoke runs are identical. Seed runs: `fno_mse_s1`, `fno_mse_s2`.
**Mach sweep (built, smoke-tested):** every Stage-3 number so far is at M=0.1. Mach is an
analysis knob (c0 = U/M; predictions unchanged), so no retraining: `09 --machs 0.05 0.1 0.2
0.3` (writes `09_blindness_machs.json`; spectra computed once, reused per Mach) and
`03 --extra-machs 0.05 0.2 0.3` (far-field complex error per Mach in `far_field_by_mach`).
Expectation [unverified]: the cone widens with M (|k| ≤ 3.5 / 6.9 / ~21 at M = 0.05 / 0.1 /
0.3) and takes in better-predicted modes, so B should fall with M — blindness strongest at
low Mach. Incompressible Lighthill is only valid for M ≪ 1, so stop at 0.3.
**Seed runs done (`0e580c3`): FNO mse seeds 0/1/2** (`fno_mse_01`, `_s1`, `_s2`):
val 2.17/2.30/2.52e-3; field 3.04/3.09/3.16%; source 28.6/29.5/30.8%; far field above
c0·dk 19.0/19.6/22.7%; **B 4.71/4.92/5.56e3 (±9%)**; B_S 9.24/9.33/9.51; low-k error/truth at
high ω 191/208/226. Rollout metrics are seed-dominated: R/R0 35/62/49, R_T ratio 1.75/14/0.77,
P/P0 0.78/0.49/1.11. Consequences: teacher-forced conclusions stand (H¹ ~20× in B is far
outside seed spread); unroll4's B 3.70e3 and UNet's 3.93e3 sit just below the seed range
(possibly small real effects); unroll4's high-ω excess (349) is outside it (worse);
**no rollout comparison from a single run is meaningful** — needs seeds or longer averages. For any trained Stage-4 run use the standard multiplicative H¹ loss
(neuraloperator `H1Loss` style), not the additive λ=1e-2 form (review §2.3).

## Mach sweep (`09 --machs`, `03 --extra-machs`; no retraining — Mach is an analysis knob)

**`fno_mse_01`** (`6c987b8`; `09_blindness_machs.json`, `machs/03_errors.json`), teacher-forced:

| M | B model | B white | B copy | B_S | R_pred/R_truth | far field above c0·dk: model / copy | P/P0 rollout |
|---|---|---|---|---|---|---|---|
| 0.05 | 1.23e6 | 4.16e7 | 66.9 | 10.6 | 1270 | 218% / 107% | 0.75 |
| 0.1 | 4.71e3 | 9.46e4 | 21.5 | 9.24 | 5.92 | 19.0% / 52.2% | 0.78 |
| 0.2 | 154 | 2.58e3 | 7.94 | 6.85 | 1.16 | 5.0% / 33.4% | 0.79 |
| 0.3 | 29.1 | 449 | 4.92 | 5.6 | 1.03 | 3.9% / 27.8% | 0.80 |

Far-field complex error over all band frequencies stays 2.6–2.8% at every Mach (dominated by
the compact low-ω term). Reading (interpretation): **velocity blindness is a low-Mach effect** —
B ∝ ~M⁻⁶ (1.2e6 → 29 from M = 0.05 to 0.3), always ~3–6% of the matched-white control; the
radiating-band far-field error goes from worse-than-copying (218%) at M = 0.05 to 7× better than
copying (3.9%) at M = 0.3. **Source-side B_S is nearly Mach-independent** (10.6 → 5.6).
Expected, since the cone widens with M (|k| ≤ 3.5 / 6.9 / ~21) and takes in well-predicted,
energetic modes. Incompressible Lighthill is valid only for M ≪ 1, so 0.3 is the ceiling.

**`unet_mse_01`** (`6c987b8`, run by the user in RunStage3; `09_blindness_machs.json`, `machs/03_errors.json`):

| M | B model | B white | B copy | B_S | R_pred/R_truth | far field above c0·dk: model / copy | P/P0 rollout |
|---|---|---|---|---|---|---|---|
| 0.05 | 5.33e5 | 4.18e7 | 66.9 | 7.62 | 128 | 120% / 107% | 0.69 |
| 0.1 | 3.93e3 | 9.49e4 | 21.5 | 6.97 | 1.91 | 11.0% / 52.2% | 0.71 |
| 0.2 | 191 | 2.59e3 | 7.94 | 6.22 | 1.04 | 3.2% / 33.4% | 0.75 |
| 0.3 | 37.6 | 450 | 4.92 | 6.69 | 1.01 | 2.6% / 27.8% | 0.81 |

Same trend as the FNO: B ∝ ~M⁻⁵ (5.3e5 → 38), and at M = 0.05 the radiating-band far field is
worse than copying for **both** architectures; B_S flat at 6–8. Low-Mach blindness is not
architecture-specific.

**`fno_h1_01`** (run 2026-10-07 on a fresh T4 runtime; code as pulled from `tier0-review-notebooks`
at `44da032`; `09_blindness_machs.json`, `machs/03_errors.json`), teacher-forced:

| M | B model | B white | B copy | B_S | R_pred/R_truth | far field above c0·dk: model / copy | P/P0 rollout |
|---|---|---|---|---|---|---|---|
| 0.05 | 2.53e7 | 4.18e7 | 66.9 | 82.6 | 1.7e4 | 635% / 107% | 2.19 |
| 0.1 | 1.08e5 | 9.52e4 | 21.5 | 68.2 | 73.4 | 60.6% / 52.2% | 2.42 |
| 0.2 | 4.07e3 | 2.58e3 | 7.94 | 44.4 | 3.70 | 16.6% / 33.4% | 2.44 |
| 0.3 | 671 | 450 | 4.92 | 28.3 | 1.43 | 11.3% / 27.8% | 2.45 |

B is 21 / 23 / 26 / 23× the `fno_mse_01` value at M = 0.05 / 0.1 / 0.2 / 0.3, i.e. the H¹ penalty is
Mach-independent to ~25%. B ∝ ~M⁻⁶ (2.5e7 → 671), same as MSE. At M ≥ 0.1 B reaches or exceeds
the matched-white control (1.08e5 vs 9.5e4 at M=0.1) — the model's error is as cone-concentrated
as incoherent noise. B_S falls with M (82.6 → 28.3) but stays 3–8× the MSE values. All-band far
field complex error 8–9% at every Mach (MSE 2.6–2.8%); below c0·dk 3.7% at M=0.1 (MSE 0.86%).
Rollout radiates 2.2–2.5× the true power at every Mach. M=0.05 radiating band is the worst case
(635%, 6× copying).

**`fno_mse_unroll4`** (same session; `09_blindness_machs.json`, `machs/03_errors.json`), teacher-forced:

| M | B model | B white | B copy | B_S | R_pred/R_truth | far field above c0·dk: model / copy | P/P0 rollout |
|---|---|---|---|---|---|---|---|
| 0.05 | 1.12e6 | 4.16e7 | 66.9 | 10.3 | 1950 | 241% / 107% | 0.41 |
| 0.1 | 3.70e3 | 9.45e4 | 21.5 | 9.34 | 7.39 | 21.0% / 52.2% | 0.44 |
| 0.2 | 123 | 2.58e3 | 7.94 | 6.23 | 1.21 | 5.8% / 33.4% | 0.48 |
| 0.3 | 22.4 | 449 | 4.92 | 4.52 | 1.04 | 4.4% / 27.8% | 0.50 |

Versus `fno_mse_01`: B is −9 / −21 / −20 / −23% at M = 0.05 / 0.1 / 0.2 / 0.3 (seed spread at M=0.1 is
±9%, so the M ≥ 0.1 reductions are modestly outside it; single seed each); the radiating-band far
field is 10–12% *worse* (241 / 21.0 / 5.8 / 4.4% vs 218 / 19.0 / 5.0 / 3.9%), and B_S is about the
same (10.3 → 4.5). All-band far field 3.0–3.2% at every Mach. Rollout is the one clear change:
B_rollout 7.7 at M=0.1 (FNO 19) and falling to 0.86 at M=0.3, R_T ratio 1.08, but P/P0 0.41–0.50
(under-radiates; FNO 0.75–0.80). So the same pattern at every Mach as at M=0.1: unrolling changes
rollout statistics, not the teacher-forced cone error.

**Mach-sweep summary (4 runs).** Teacher-forced B falls ~M⁻⁵–M⁻⁶ for every run; at M=0.05 the
radiating-band far field is worse than copying for all four (UNet 120%, FNO 218%, unroll4 241%,
H¹ 635%, copy 107%). The ordering of the objectives (H¹ ≫ MSE ≈ unroll4 ≈ UNet in B) is the same at
every Mach, so the M=0.1 conclusions are not specific to that Mach. B_S is not flat in M for any run
(FNO 10.6 → 5.6, unroll4 10.3 → 4.5, H¹ 83 → 28; UNet 7.6 → 6.7), only much less steep than B.
The earlier statement "B_S nearly Mach-independent" is softened to "falls ≤3× over a 6× range of M".
Still one dataset, one seed per non-FNO-MSE run.

## kf16 replication (2026-10-07; `data/turb2d_kf16.npz`, forcing k=16, sha256 verified; all runs `--seed`)

Goal: does the baseline pattern (B between the copy and white controls, H¹ ≫ MSE in B, radiating
band worse than copying at low Mach) hold on a second flow? Runs on Drive `runs/kf16_<name>/`,
Colab T4, same recipe as the baseline (FNO, 60 epochs, batch 16). Pass criteria written beforehand
(plan agreed in session): (1) MSE B seed spread ≤ ±25%; (2) H¹ B ≥ 3× MSE; (3) MSE B between copy and
white; (4) radiating-band far field worse than copying at M=0.05. Compare ratios/ordering, not
absolute B (truth cone share differs: R_T 3.0e-9 kf16 vs 4.9e-9 baseline at M=0.1).

| run | status | one-step val (final / best) |
|---|---|---|
| `kf16_fno_mse_s0` | complete (03, 09 --machs, 10) | 6.56e-3 / 5.31e-3 |
| `kf16_fno_mse_s1` | complete | 6.32e-3 / 5.17e-3 |
| `kf16_fno_mse_s2` | complete | 6.57e-3 / 5.32e-3 |
| `kf16_fno_h1_s0` | complete (03, 09 --machs, 10; the `09`/`10` ran in a second session after a Colab usage-limit stop) | 1.68e-2 / 5.65e-3 |
| `kf16_fno_h1_s1` | complete | 1.64e-2 / 5.47e-3 |

FNO overfits on kf16: val bottoms at epoch ~12 and rises (baseline val 2.2e-3, no rise); field
error 5.3% vs 3.0%. Same recipe, so the comparison is unaffected, but absolute errors are higher.

Teacher-forced, MSE seeds 0 / 1 / 2 (copy-input in last column):

| measure (M=0.1) | s0 | s1 | s2 | copy |
|---|---|---|---|---|
| field / source error | 5.30 / 28.89% | 5.33 / 29.06% | 5.34 / 29.44% | 12.62 / 34.07% |
| far field complex: all / below / above c0·dk | 7.11 / 1.38 / 24.63% | 7.25 / 1.01 / 25.35% | 6.53 / 1.04 / 22.79% | 18.70 / 14.75 / 43.14% |
| B | 1.75e3 | 1.69e3 | 1.68e3 | 7.78 |
| B_S | 17.7 | 16.2 | 16.3 | 3.87 |
| low-k error ÷ truth, high ω | 279 | 265 | 270 | 2.2 |

B at M = 0.05 / 0.1 / 0.2 / 0.3: s0 1.37e5 / 1.75e3 / 161 / 47.8; s1 1.35e5 / 1.69e3 / 150 / 44.7;
s2 1.37e5 / 1.68e3 / 154 / 46.1. Controls (identical across runs): copy 26.6 / 7.78 / 4.03 / 3.34;
matched white 5.11e6 / 4.64e4 / 3.91e3 / 1.13e3. B_S (s0): 17.9 / 17.7 / 14.3 / 11.2. Far field above
c0·dk, s0 (model | copy): 113% | 82.3% at M=0.05, 24.6 | 43.1 at 0.1, 9.6 | 29.3 at 0.2, 8.7 | 27.1 at 0.3.
Rollout metrics vary by seed again (M=0.1: R/R0 62 / 60 / 79; R_T ratio 5.5 / 8.4 / 21.5; P/P0
1.36 / 1.12 / 1.55).

**H¹ (seeds 0 / 1), teacher-forced, vs the MSE mean of the three seeds:**

| measure (M=0.1) | H¹ s0 | H¹ s1 | MSE s0/s1/s2 |
|---|---|---|---|
| field / source error | 5.66 / 33.25% | 5.63 / 32.68% | 5.30–5.34 / 28.9–29.4% |
| far field complex above c0·dk | 76.67% | 68.00% | 24.63 / 25.35 / 22.79% |
| B | 4.28e4 | 4.63e4 | 1.75 / 1.69 / 1.68e3 (mean 1.71e3) |
| B_S | 165 | 187 | 17.7 / 16.2 / 16.3 |
| low-k error ÷ truth, high ω | 3940 | 5140 | 279 / 265 / 270 |

B by Mach (M = 0.05 / 0.1 / 0.2 / 0.3): s0 2.69e6 / 4.28e4 / 4.06e3 / 1.05e3; s1 2.95e6 / 4.63e4 / 4.23e3 /
1.11e3. Ratio to the MSE mean (1.36e5 / 1.71e3 / 155 / 46.2): s0 19.7 / 25.1 / 26.2 / 22.7×, s1 21.6 / 27.1 /
27.3 / 24.0× (baseline flow: 21 / 23 / 26 / 23×). H¹ B at M ≥ 0.1 equals or exceeds the matched-white
control (4.63e4 at M=0.1). Seed spread of H¹ B at M=0.1: 8%. Free rollout is unstable under H¹ here
(s0: field error ~1e37%, P/P0 ~1e131; s1: field error 270%, source 7458%, P/P0 1e4) — much worse than the
baseline H¹ rollout (P/P0 2.4); rollout is seed-dominated, so this is only "unstable", not a magnitude.

Reading (interpretation): the baseline pattern reproduces on a second flow for MSE — the same
position between the controls, tight seed spread, the same jitter fingerprint (low-k error ÷ truth
at high ω 265–279 vs 191–226 on the baseline), the same low-Mach failure. Differences: B_S is ~1.8×
higher (16–18 vs 9), and B falls with Mach by a similar factor (~M⁻⁵·⁹, 1.4e5 → 46). The H¹
H¹ comparison is now complete: 3 MSE seeds + 2 H¹ seeds on a second flow reproduce the baseline
ordering and ratio (H¹ 20–27× MSE in B at every Mach; far field above c0·dk ≈3× worse; B_S ≈10× worse).
Remaining limits: one extra flow, no UNet/unroll4 on kf16, seeds 1–2 of the Mach-0.05 far-field criterion 4 are
in their `03_errors.json` but unread. Notes: the run-1 cell was accidentally re-executed once (evaluation only,
deterministic; the logged numbers match the first pass).

## Experiment 5 — cone-window loss, baseline data (2026-10-08; code `050d062`, plan `EXP5_PLAN.md`)

`--loss cone`: MSE + λ × (error energy inside the cone ω ≥ c0|k|, k>0, over W=32 teacher-forced
consecutive one-step predictions, Hann in time, exact per-mode mask, ÷ numel so it equals
error-cone-fraction × MSE scale). FNO, `data/turb2d.npz`, seed 0, W=32, stride 16, batch 1 window
(32 frames/step; 62 windows), training Mach 0.1 (c0 = 11.308), 60 epochs. Cone-mode training
11.4–11.6 min on the T4 (plain MSE ~7), peak GPU 2.5 GB (probe). Runs `runs/cone_l<λ>_m0.1_w32_s0/`,
evaluated with `03 --extra-machs`, `09 --machs`, `10` exactly as before. λ=1e3 was evaluated twice by an
accidental cell re-run (identical numbers, evaluation is deterministic; the figures below are from the
first pass, the repeat matched on `03`).

Teacher-forced, M=0.1 (MSE = three FNO seeds 0/1/2, baseline):

| measure | MSE seeds | λ=1e2 | λ=1e3 | λ=1e4 | copy-input |
|---|---|---|---|---|---|
| B | 4.71 / 4.92 / 5.56e3 | 603 | 81.9 | **21.3** | 21.5 |
| B_S | 9.24 / 9.33 / 9.51 | 5.80 | 2.25 | 1.54 | 3.16 |
| field error | 3.04 / 3.09 / 3.16% | 3.17% | 3.17% | 3.36% | 9.63% |
| source error | 28.6 / 29.5 / 30.8% | 27.79% | 23.81% | 23.40% | 31.55% |
| far field complex: all / below / above c0·dk | 2.7 / 0.9 / 19.0–22.7% | 2.74 / 1.43 / 17.24% | 2.39 / 1.62 / **13.05%** | 3.17 / 2.40 / 15.52% | 18.9 / 17.7 / 52.2% |
| low-k error ÷ truth, high ω | 191 / 208 / 226 | 96 | 13.6 | 3.5 | 2.2 |
| one-step val MSE (final) | 2.17 / 2.30 / 2.52e-3 | 1.89e-3 | 1.49e-3 | 1.52e-3 | |
| val cone term | | 1.3e-7 | 2.4e-8 | 6.8e-9 | |

(Validation splits differ between window and pair modes — held-out last windows with a boundary
drop — so the val-MSE row is not a strict comparison.)

B by Mach, M = 0.05 / 0.1 / 0.2 / 0.3: λ=1e2 1.59e5 / 603 / 72.7 / 17.0; λ=1e3 1.73e4 / 81.9 / 19.8 /
6.92; λ=1e4 2.46e3 / 21.3 / 8.68 / 4.93; MSE seed 0 1.23e6 / 4.71e3 / 154 / 29.1; copy 66.9 / 21.5 /
7.94 / 4.92. Factor below MSE seed 0: λ=1e2 7.7 / 7.8 / 2.1 / 1.7×; λ=1e3 71 / 57 / 7.8 / 4.2×; λ=1e4
500 / 221 / 17.7 / 5.9×. B_S by Mach λ=1e3: 2.11 / 2.25 / 2.32 / 3.12 (flat, and below the copy-input
value 2.56 / 3.16 / 5.52 / 9.2 at every Mach); λ=1e4 1.37 / 1.54 / 2.34 / 4.14.

Free rollout (1023 steps, all finite), M=0.1; MSE seeds in parentheses:

| | λ=1e2 | λ=1e3 | λ=1e4 |
|---|---|---|---|
| field / source error | 134% / 159% | 238% / 634% | 309% / 2306% |
| P/P0 (radiated power ratio) | 1.98 | 30 | 233 (at M=0.05 256) |
| R_T/R_T0 | 1.33 | 28.9 | 89.5 |
| (MSE seeds) | P/P0 0.49–1.11, R_T ratio 0.77–14, field ~136% | | |

Pre-registered criteria (`EXP5_PLAN.md` §4), each λ: (i) B < ⅓ of the MSE mean (5.06e3) and outside
seed spread — **pass** (603 / 82 / 21); (ii) far field above c0·dk below 19.0–22.7% — **pass** (17.2 /
13.1 / 15.5%); (iii) field error ≤10% above 3.04–3.16% — **pass** (3.17 / 3.17 / 3.36%); (iv) B_S not
worse — **pass**; (v) effect at other Mach — **pass**, but the gain shrinks with M (λ=1e3: 71× → 4.2×).

Reading (interpretation unless stated):
- **Monotone dose-response.** B falls with λ (603 → 82 → 21), reaches the copy-input floor at λ=1e4, and
  low-k high-ω error ÷ truth falls 96 → 13.6 → 3.5 (truth-level 2.2). The jitter is removable with a
  loss that sees the cone; L2 simply never asked [consistent with the claim; not proven for other flows].
- **No field-accuracy cost up to λ=1e3**; λ=1e4 costs ~6–8% of field error (3.36 vs 3.04–3.16%) and its
  far-field above c0·dk is worse than λ=1e3's (15.5 vs 13.1%). λ=1e3 is the best balance on the
  teacher-forced metrics; λ=1e4 minimises B.
- **Not just the trained metric.** B at M=0.1 is close to what the loss optimises (same mask, same c0).
  The independent checks improve too: B_S (a quadratic source measure, 9.4 → 2.25), source error
  (29.6 → 23.8%), far field above c0·dk, and B at other Mach (where the cone differs: the trained M=0.1
  cone is a subset of the wider M ≥ 0.2 cones, which may explain the shrinking gain there [untested]).
- **Rollout gets worse with λ** (P/P0 1.98 → 30 → 233; R_T ratio 1.3 → 29 → 90), monotone over three λ
  and outside the MSE seed range for λ ≥ 1e2. Single seed per λ, so the magnitude is soft, but the trend
  is the open risk: the loss improves the teacher-forced error and degrades free-running forecasts
  [mechanism unknown; not tested].
- Limits: one seed per λ, one dataset (kf16 not yet run), window W=16 variant not yet run, training Mach
  0.05 variant not run, MSE comparison is to the baseline seeds (no matched-seed cone runs).

## Error-floor crossover vs observed M* (`scripts/11_crossover.py`, run 2026-10-08 on a Colab CPU runtime)

Hypothesis in the script's docstring (2026-10-08 review): at low |k| the true temporal spectrum falls
far faster in ω than a trained surrogate's error, so error exceeds signal above a crossover ω_x, and
the whole lowest-shell radiating band (ω ≥ U·dk/M) lies above ω_x once M ≤ M*_pred = U·dk/ω_x.
Compared with M*_obs from `03` (Mach where the model's far-field error above c0·dk equals the
copy-input error). Inputs: `10`/`03 --extra-machs`/`09` JSONs of all runs; outputs
`MyDrive/silent-field/analysis/11_crossover/11_crossover.{json,png}`. Script caveats: the low-k PSD
is summed over |k| ≤ ω_max/c0 at M=0.1 (not the dk shell alone), so ω_x is a proxy; M*_obs below
the lowest evaluated Mach (0.05) is extrapolated (tag `extrapolated_below`: cone λ=1e3, 1e4).

- **Slopes (fit over the top of the ω band) [computed]:** truth spectrum −14.2 (baseline), −13.7
  (kf16); error spectra −5.3 to −7.9 (MSE −6.2 baseline / −7.7…−7.9 kf16; H¹ −5.3 / −7.4…−7.5; cone
  −6.1 / −6.5 / −7.6 for λ=1e2 / 1e3 / 1e4). The truth falls ~8 powers of ω faster than every error.
- **ω_x (error PSD = truth PSD):** MSE 20.8–21.7 (baseline), 18.5 (kf16); UNet 30.6; unroll4 21.1;
  H¹ 16.6 / 13.6–13.7; cone 23.5 / 30.5 / 34.9 (λ=1e2 / 1e3 / 1e4).
- **Prediction vs observation, 12 runs with a Mach-swept `03`:** log-log correlation of M*_pred and
  M*_obs(copy) = 0.797; obs/pred median 1.41, range 1.26–3.82. Against M*_obs(100%) (the Mach where
  the radiating-band far-field error reaches 100%) the ratio from the printed table is 1.18–1.52
  [computed by hand from the table]: the prediction is systematically ~20–50% too low.
- Examples (M*_pred → M*_obs(copy)): MSE baseline 0.052 → 0.067; kf16 MSE 0.043 → 0.060–0.064; H¹
  0.068 → 0.113 (baseline), 0.058 → 0.18–0.22 (kf16); cone λ=1e2 0.048 → 0.062, λ=1e3 0.037 → 0.047
  (extrapolated), λ=1e4 0.032 → 0.042 (extrapolated).

Reading (interpretation): the crossover mechanism has the right sign and ordering — it gives a
falsifiable M* of 0.03–0.07 for the evaluated runs and ranks the loss families correctly (H¹ highest,
cone lowest) — which turns "blindness gets worse at low Mach" from a statement about the metric into a
statement about two spectra. Limits: 12 runs are not independent (two datasets, repeated seeds) and
the correlation is largely between loss families; there is a consistent 1.2–1.5× bias and ω_x is a
proxy (summed over a k range); the cone-trained M* values are extrapolated below M=0.05. The cone loss
moves M*_obs(copy) only from ~0.067 to ~0.047 / 0.042 despite B falling 57–220× — i.e. it shifts
the failure Mach by ~1.4–1.6×, not by orders of magnitude [computed from the table]. The 100%
definition is more stable for kf16 H¹ (0.085–0.088) than the vs-copy one (0.18–0.22).
