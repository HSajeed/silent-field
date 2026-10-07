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
| `kf16_fno_h1_s0` | trained, `03` done; **`09`, `10` not run** (Colab usage limit) | 1.68e-2 / 5.65e-3 |
| `kf16_fno_h1_s1` | **not started** | — |

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

**H¹ seed 0, `03` only (M=0.1):** field 5.66%, source 33.25%, far field complex 22.39% (below 5.63%,
above 76.67%) — vs MSE above-c0·dk 22.8–25.4% (≈3.1×; baseline 60.6 vs 19.0% = 3.2×). Its B is not measured.

Criteria status:
1. **Pass.** B(M=0.1) spread 4% across 3 seeds (baseline ±9%).
2. **Open** — needs `09` on `kf16_fno_h1_s0`. Far-field result above is consistent with the baseline.
3. **Pass.** B ≈ 3.7% of the matched-white value (baseline ~5%), 220× the copy value.
4. **Pass for s0** (113% vs 82.3%); s1/s2 are in their `03_errors.json` (`far_field_by_mach`), not read.

Reading (interpretation): the baseline pattern reproduces on a second flow for MSE — the same
position between the controls, tight seed spread, the same jitter fingerprint (low-k error ÷ truth
at high ω 265–279 vs 191–226 on the baseline), the same low-Mach failure. Differences: B_S is ~1.8×
higher (16–18 vs 9), and B falls with Mach by a similar factor (~M⁻⁵·⁹, 1.4e5 → 46). The H¹
comparison is not complete: **do not cite criterion 2 as met**. To finish: `09` (and `10`) on
`kf16_fno_h1_s0` (training resumes from `last.pt`, so the cell re-trains nothing), then `kf16_fno_h1_s1`
in full; ~30 min of T4. Notes: the run-1 cell was accidentally re-executed once (evaluation only,
deterministic; the logged numbers match the first pass).
