# Experiment 5 — cone-weighted loss over a time window

Plan written before running (pre-registered). Numbers taken from `HANDOFF.md` and
`stage3_log.md`; none invented. Status markers: [computed] = arithmetic shown here from
logged numbers, [interpretation] = reading/hypothesis, [unverified] = not measured yet.

## 1. Question and hypothesis

Question: does a loss that penalises the radiating-cone part of the one-step error over a
time window lower the cone-concentrated (jitter) error B and the high-frequency far-field
excess, without hurting the field?

Hypothesis [interpretation]: the lever is weighting in ω, not architecture or unrolling.
`stage3_log.md` shows B tracks the objective (every MSE variant — FNO, UNet, unroll×4 —
sits at B ≈ 4e3; standard H¹ gives B = 1.08e5, ~23×; unroll×4 gives B −21%, just outside
seed spread) and not the architecture or the training horizon.

## 2. The loss

As implemented by a coder agent in parallel (described here, not pasted):
`total = MSE + lambda * (windowed error energy inside the cone omega >= c0|k|, k>0,
divided by the number of error elements)`, i.e. the cone term equals
error-cone-fraction × MSE-scale. Teacher-forced windows of W=16 consecutive one-step
predictions, Hann window in time, exact per-mode cone mask (no radial binning, per the
Tier-0 fix in `HANDOFF.md`), c0 = the same convention as `scripts/04` and `09` at a
training Mach (default 0.1). CLI: `--loss cone --cone-lambda --cone-mach --cone-window
--cone-stride --cone-batch`.

**Expected scale [interpretation, arithmetic below].** `09` defines B = R(error)/R(truth)
(velocity cone share). HANDOFF's R(all k) at M=0.1, baseline flow = 3.2e-08 [computed,
`HANDOFF.md` Current conclusions table]. Logged B for FNO-MSE at M=0.1 = 4.71e3 (seed 0)
[computed, `stage3_log.md`]. So the error's own cone fraction ≈ B × R(truth):

```
cone_fraction(error) ≈ 4.71e3 × 3.2e-08 ≈ 1.51e-4
```

So under plain MSE training, roughly 1.5e-4 of the error's energy sits in the cone — this
is the "cone-fraction" the new term multiplies by lambda. For lambda·term to become
comparable to the MSE term (lambda·cone_fraction ≈ 1), lambda needs to be of order
1/1.51e-4 ≈ 6.6e3 [computed from the above]. That is why lambda in the 1e2–1e4 range is
the range where the cone term goes from negligible (1e2 → ~1.5% of MSE scale) to dominant
(1e4 → ~1.5× MSE scale) [interpretation: this is a rough scale estimate using the
MSE-trained error's cone fraction as a proxy for the cone-loss term itself, not an exact
prediction — the term is computed on a windowed, Hann-tapered, teacher-forced error, not
on the full-record error R(error) that defines B].

## 3. Runs

FNO, baseline data `data/turb2d.npz`, seed 0 first, then more seeds for any lambda that
works. lambda ∈ {1e2, 1e3, 1e4}, training Mach 0.1. Variant: training Mach 0.05 at the best
lambda. Same 60 epochs and recipe as existing Stage-3 runs. Out dirs
`runs/cone_l<lambda>_m<mach>_s<seed>`.

Evaluation exactly as the other runs:
- `scripts/03` with `--extra-machs 0.05 0.2 0.3`
- `scripts/09` with `--machs 0.05 0.1 0.2 0.3`
- `scripts/10`

Then replicate on `data/turb2d_kf16.npz` (flag: kf16 FNO-MSE baseline numbers are in
`stage3_log.md` "kf16 replication": B(M=0.1) 1.75e3 / 1.69e3 / 1.68e3 seeds 0/1/2, 4%
spread [computed]).

## 4. Success criteria (pre-registered)

(i) B at M=0.1 below one third of the matched-seed FNO-MSE mean (baseline seeds
4.71/4.92/5.56e3 [computed]) **and** outside the seed spread (±9% [computed]).
(ii) Far field complex error above c0·dk lower than FNO-MSE's 19.0–22.7% (baseline seeds)
[computed].
(iii) Field error not more than 10% worse than 3.04–3.16% (baseline FNO-MSE seeds)
[computed].
(iv) B_S not worse than baseline (9.24/9.33/9.51 [computed]).
(v) Effect persists at other Mach (0.05, 0.2, 0.3) — if the benefit only appears at the
training Mach, report that as a weaker, Mach-specific result.

**Failure modes and how to read them:**
- B down but field error up → a trade-off, not a fix.
- B unchanged at the lambda where the cone term dominates MSE (lambda ≳ 1e4 by the
  §2 estimate) → either the jitter is not reducible by this term, or the cone term (as
  defined here, on a W=16 window) is not the right proxy for the B metric (computed on
  the full record).
- Far-field error improves but B does not (or vice versa) → the two metrics are not
  measuring the same thing; report both, do not conflate.

## 5. Risks and controls

- **Minimisation shortcuts.** The term could be minimised either by making the error
  smooth in time (the intended fix) or by shrinking the prediction toward the input.
  Compare against the copy-input baseline (B = 21.5, source error 31.6%
  [computed, `stage3_log.md`]) and against rollout stability — a run that scores well on
  the cone term but collapses toward copy-input on rollout is not a real fix.
- **Window resolution.** W=16 gives a coarse ω grid: ω step = 2π/(16·0.04) = 9.8
  [computed, as given]. At M=0.1, c0·dk = 11.31 [computed, `HANDOFF.md`]; the first ω bin
  above that is 19.6 [computed, as given]. So the low-k cone coverage at M=0.1 is coarse —
  the window mostly captures the next bin up, not a fine sampling near the cone edge.
- **Rollout metrics are seed-dominated** (`stage3_log.md`: rollout R/R0 spans 35/62/49
  across FNO-MSE seeds [computed]) — judge the success criteria on teacher-forced numbers,
  use rollout only as a secondary check.
- **One dataset** for the primary sweep (kf16 is a replication, not an independent test).
- **Literature baselines still to run afterwards**: noise injection, PDE-Refiner-style
  multistep refinement (`lit_review_2026-10-07.md` §1, §3: PDE-Refiner's fix is multistep
  denoising refinement; Sobolev/H¹ losses "decrease performance also for FNOs"
  [verified, quoted in lit review]; unrolled loss with backprop through steps is
  "not found tested for this purpose" [quoted, lit review §3 table] — our own unroll×4
  result already tested that one, see §1 above).

## 6. Workflow

Code changes local only, smoke-tested locally by the coder agent; user pushes to GitHub;
then Colab (`colabmcp.md`, local/untracked — do not copy tokens or secrets from it).
Estimated time per run: FNO one-step MSE training ~7 min (`stage3_log.md`, `fno_mse_01`:
60 epochs in 6.7 min [computed]); cone-loss mode may be slower, amount unknown
[unverified]; evaluation (`03`+`09`+`10`) ~13 min, by analogy with existing runs
[interpretation, not separately timed in the logs].

## 7. Check added by the main agent after review: Hann leakage floor of the cone term [computed locally, turb2d]

The cone term is measured on a short windowed error, and almost all of the one-step error is
smooth low-ω, so a Hann window can leak that error into the first cone bins. Measured with
`cone_window_loss` (c0 = 11.308, dt = 0.04) on the real data, cone term ÷ MSE of the error:

| error | W=16 (stride 8) | W=32 (stride 16) |
|---|---|---|
| copy-input (y − x; coherent, smooth in time) | 2.9e-5 | 1.7e-6 |
| white noise of the same variance | 3.0e-3 | 3.0e-3 |
| copy-input error shuffled in time (incoherent) | 9.5e-2 | 9.4e-2 |

- The full-record equivalent for the copy-input error is B_copy × R_truth = 21.5 × 3.2e-8 ≈ 7e-7
  [computed from logged numbers], so at W=16 the windowed term is ~40× the full-record value
  (leakage), at W=32 ~2.4×.
- The target quantity is ~1.5e-4 of the MSE (§2). The W=16 floor for the *copy* error (2.9e-5, ~20%
  of target) is an upper bound for the FNO, whose smooth error is smaller, so W=16 is usable but
  the term is not clean at the lowest cone bins; W=32 lowers the floor ~17× and halves the ω step
  (4.9 instead of 9.8; first bin above c0·dk=11.3 is 14.7).
- [interpretation] Prefer W=32 as the primary setting (`--cone-window 32 --cone-stride 16
  --cone-batch 1`: 32 frames per step, ~54 steps per epoch, est. peak GPU memory ~1.9 GB for
  W=16 × batch 2, so the same for W=32 × batch 1 [unverified]); keep W=16 as a variant. This
  changes §3 and is a decision for the user; nothing in the code defaults to it yet (default is W=16).
