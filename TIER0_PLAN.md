# Tier 0 — correctness fixes

Derived from `branch_analysis.md` (2026-09-29) §7 Tier 0, re-verified numerically
before execution. Decisions taken 2026-09-30 by the project owner:

1. **Regenerate now** — fix the code, re-run `scripts/04`, update every quoted number
   with a dated note. Do not leave `figures/komega_metrics.json` disagreeing with the
   code that produced it.
2. **Correct the README now** — replace the "ten times worse" sentence, which is an
   artifact of comparing against a weighting the code never trains.
3. **Far-field band mode `auto`** — try `k` (lowest radiating box modes), fall back to
   `top` (strongest radiating source power) if the band captures <1% of cone energy.

Environment: conda env `dmdsae`
(`/home/sajeed/miniforge3/envs/dmdsae/bin/python`), numpy 2.4.6, scipy 1.17.1,
torch 2.12.1, matplotlib 3.11.1. Baseline before any change: **21 tests passed**.

---

## Verified findings

Measured on `data/turb2d.npz` (n=128, nt=1024, dt_save=0.04, L=2π, u_rms=1.1308,
c₀=11.308 at M=0.1), full-record single estimator unless stated.

| Check | Result |
|---|---|
| Current far-field band `om[:24]` | ω ∈ [0.15, 3.68]. **91.0%** of the Lighthill source's ω>0 power is at ω < 11.31 — the band sits entirely on the convective ridge |
| Radiating frequencies available | ω ≥ c₀·dk = 11.31 → **439** of 513 bins |
| Cone content of the source, lowest 24 radiating bins (ω 11.35–14.9) | **3.2%** of source ω>0 power |
| Multiplicative (1+k²) cone share, velocity | 4.69e-9 = **16.3× below** R_L2 (7.63e-8). This is the origin of the README's "10× worse" |
| Additive `1 + 2λw` at default `sobolev_weight=1e-2` | 7.628e-8 = **1.00× below** R_L2 — Sobolev changes the cone share by 0.02% |
| Same, λ = 1e2 / 1e3 / 1e6 | 2.56e-8 (3.0×) / 7.52e-9 (10.2×) / 4.69e-9 (16.3×) |
| Sobolev weight `w` inside the cone | 3.6e-4 … 1.7e-2, i.e. 100–1000× below its own mean; cone area fraction 3.3e-3 |
| **Binning bias (new, not in `branch_analysis.md`)** | Masking the binned `k_bins` instead of the true mode grid inflates R by **2.84×** (velocity 1.47e-7 vs 5.18e-8 exact) and **4.79×** (source 1.67e-8 vs 3.49e-9 exact). Cone edge at k=6.95, i.e. 0.45 of a bin above a bin centre |

`branch_analysis.md` quotes the additive case as 6.41e-8 (1.4× below L2). That figure
is **not reproducible** — it depends on a normalization the code never fixes. The
additive objective's per-mode weight is `1 + 2λw`, the 2 coming from the ratio of the
two cell counts in `losses.py:55-60` (mean over all cells vs mean over the rfft2
half-plane). We publish the **λ-sweep table**, not a single ratio, so the
normalization constant becomes a visible axis rather than a hidden factor.

---

## Design note: the resolved radial grid

`radiating_mask` takes `c0` as an argument, so the mask cannot be baked into the
spectrum at construction time — `c0` is an *analysis* knob that one dataset sweeps
(`komega.py:23-25`). Baking it in would freeze the Mach number into the data
structure.

Instead `KOmegaSpectrum` gains a radially-**resolved** companion to the binned power.
The grid has only O(n) distinct |k| values, not n²:

- `k_resolved` — the distinct |k| values present, extending to 90.5 so corner modes are
  included and the total is genuinely total;
- `power_resolved` — shape `(n_omega, n_k_resolved)`, power summed over the modes
  sharing each |k|, built with one `np.add.reduceat` over a |k|-sorted mode list.

Memory: 513 × ~600 ≈ 2.4 MB for the baseline; ≈20 MB for n=256, nt=4096. Negligible
against the ~7 GB budget noted in `HANDOFF.md:36`.

This is strictly better than masking before binning:
- the mask is exact **and** stays c₀-dependent, evaluated on demand at full |k|
  resolution, so the 2.84×/4.79× binning bias disappears;
- corner-mode truncation is fixed for free rather than merely documented;
- Item 2 can normalize the Sobolev weight with the correct **mode multiplicities**,
  matching `_k_grid` in `losses.py:32-34`.

**Compatibility.** `KOmegaSpectrum.power`, `.k_bins`, `.radiating_mask(c0)` and
`.radiating_fraction(c0)` keep their shapes and semantics. `radiating_mask` stays the
binned mask (plotting, and the two archived `runs/*/analysis.py` scripts). New:
`radiating_mask_resolved(c0)`, `radiating_fraction_resolved(c0)`, and
`radiating_fraction` delegating to the resolved version. The archived
`runs/*/results.json` are **not** re-run — they are historical pre-fix records.

---

## Item 3 — mask correctness (`src/acoustics/komega.py`)

**3a.** `radiating_mask` gains `(kk > 0) & (ww >= c0*kk)`. No effect on current data
(that column sums to 2.6e-32 after `detrend_mean`); pure robustness against a future
dataset that develops mean flow.

**3b.** `komega_spectrum` / `welch_komega_spectrum` also return `k_resolved` +
`power_resolved`. `radiating_fraction` switches to `radiating_mask_resolved`.
Docstring records the measured bias being removed and notes that corner modes are now
included in the total. The corner-mode contribution is to be quantified during
implementation and reported, not assumed.

Tests (`tests/test_komega.py`, +2):
- `test_mask_excludes_k_zero_column`
- `test_mask_uses_true_k_not_bin_centre` — plant a wave at **k=4.4** (off-centre), phase
  speed 1.3. The existing `test_sonic_mask` uses k=4.0, exactly on a bin centre, which
  is why this bias survived.

## Item 2 — Sobolev convention (`radiating.py`, `losses.py` docstring, `04`, `06`)

Split `effective_weight_in_cone` into a c₀-free primitive and a wrapper so the
normalization is testable in isolation:

```
weighted_cone_share(power, weight, mask) -> float
effective_weight_in_cone(spectrum, c0, kind="l2", order=1,
                         lam=1e-2, convention="multiplicative") -> float
```

- `convention="multiplicative"` — current behaviour exactly. Existing callers and
  `komega_metrics.json` keys unchanged.
- `convention="additive"` — the weight the code **actually trains**:
  `w_eff(k) = 1 + 2λ(1+k²)^order / mean_halfplane`, mean-normalized with `k_resolved`
  mode multiplicities so it matches `_k_grid`. Requires Item 3b.
- New `sobolev_lambda_sweep(spectrum, c0, order, lambdas)` → rows of
  `{lambda, cone_share, ratio_to_l2}`.
- `losses.py`: docstring cross-reference only. **No functional change** —
  `losses.py` is the reference definition; `radiating.py` bends to it.

Tests (`tests/test_radiating.py`, new, +4):
1. `test_additive_tends_to_l2_as_lambda_vanishes`
2. `test_additive_tends_to_multiplicative_as_lambda_grows`
3. `test_additive_weight_matches_torch_sobolev_loss` — **load-bearing.** Reproduces
   `losses.py`'s `base`/`high` from torch and asserts `weighted_cone_share` agrees to
   1e-6. Pins the factor 2; stops the two files drifting apart again.
4. `test_default_lambda_leaves_cone_share_unchanged` — at `TrainConfig().sobolev_weight`
   the additive share is within 1% of R_L2. **The paper claim, encoded as a test.**

Scripts:
- `scripts/04_komega_analysis.py` — existing `h1` column relabelled `h1_mult` with an
  inline note that it is not the trained objective; new sweep column. Existing JSON
  keys preserved, new keys added.
- `scripts/06_sobolev_argument.py` (**new**) — the Stage-4 analytical result: λ-sweep
  table as JSON plus one figure (cone share vs λ, log-log, both asymptotes marked as
  horizontal lines, `sobolev_weight=1e-2` flagged). No trained model, ~10 s.

Docs:
- `README.md` — replace the "ten times worse" sentence with the corrected finding; fix
  the stale test count in the Layout block.
- `HANDOFF.md` — refreshed R / R_S table, H¹ line, test count, dated Tier 0 entry with
  old→new numbers.
- `acoustic-blindness-build-plan.md` — Stage 4 becomes an analysis table plus an
  optional confirmation training run.

## Item 1 — far-field band (`greens.py`, `scripts/03_compute_sources.py`)

- New `radiating_band(omega, c0, grid, n_band, mode)` in `greens.py` beside
  `source_spectrum`, returning `(indices, metadata)`.
  - `k` — `ω_j = c₀·j·dk`, j=1..n_band. Record-length independent, physically
    anchored. kr becomes `j·30L` = 188…4524, so `far_field=True` is valid by a wide
    margin; this retires the audit's §2.3 concern for this script for free.
  - `top` — the `n_band` radiating bins with the largest **truth**-source power; the
    same band is applied to truth and prediction.
- `--band-mode {auto,k,top}` default `auto`; `--min-cone-fraction` default 0.01.
  `auto` tries `k`, and below the threshold prints a notice and switches to `top`.
- `03_errors.json` gains the band definition, ω range, `n_radiating_available`,
  `k_max_in_band`, `kr_min`/`kr_max`, `cone_energy_fraction_captured`, `band_mode_used`.
  Cost unchanged: 24 × 36 × 16384 = 1.4e7 evals, under the 5e7 guard.

Test (`tests/test_greens.py`, +1): every returned index satisfies `ω ≥ c₀·dk`, the set
is record-length independent, and `top` is a subset of the radiating set. No test
through `scripts/03` — it needs a checkpoint that does not exist.

---

## Verification order

```
python -m pytest -q
python scripts/00_validate_acoustics.py --n 128 --outdir figures   # must PASS, unchanged
python scripts/04_komega_analysis.py
python scripts/06_sobolev_argument.py
```

The gate at `00` is independent of all three items and **must not move**. Expected
headline shift from 3b: R(all k) and R_S both drop by a factor of a few — the claim
strengthens. Old→new recorded in `HANDOFF.md`.

Test count: 21 → 28.

## Out of scope

No new data, no training, no `cone_weighted_loss`, no `min_kr`/`greens.py` near-field
work, no packaging, no M⁸ retirement, no second QoI, no re-running the archived
`runs/*` analyses. Those are Tier 1+.

---

# OUTCOME — completed 2026-09-30

Test count **21 → 42**. Gate `00` re-run: unchanged, PASS. `05` re-run: slope
−3.981, unchanged.

| Item | Status | Result |
|---|---|---|
| 3a `k>0` guard | done | no effect on current data (that column is 2.6e-32) |
| 3b resolved grid | done | R 8.7e-08 → **3.2e-08**, R_S 1.0e-08 → **2.3e-09** (baseline, M=0.1) |
| 2 Sobolev convention | done | trained objective gives **1.000× L2** at default λ; multiplicative gives 0.053× |
| 1 far-field band | done | `greens.select_band`; on real data 0/24 → 24/24 radiating, kr 189–1309 |

New artifacts: `figures/06_sobolev_argument.{json,png}`, λ-sweep per Mach inside
`figures/komega_metrics.json`. New file: `scripts/06_sobolev_argument.py`
(torch-free, no model, ~1 min).

## Deviations from the plan above

1. **The load-bearing Sobolev test changed shape.** The plan called for
   `weighted_cone_share` to agree with a brute-force torch evaluation to 1e-6. It
   cannot, and chasing it would have been the wrong fix: the two disagree by a
   *convention*, not a bug — `losses.sobolev_loss` has no frequency axis while the
   analysis folds one, so the absolute prefactor relating its two terms is not
   defined by the code. The test is now `test_additive_weight_is_one_plus_scaled_sobolev_weight`,
   which pins the weight **elementwise** to `losses._k_grid` — exact up to float32 —
   and the λ-limits and inertness tests carry the rest. This is also *why* the result
   is published as a λ sweep rather than a single ratio; the prefactor is now a
   visible axis. Documented in the `radiating.py` module docstring.

2. **`mode="k"` was redefined mid-implementation.** As first written it selected
   `ω_j = c₀·j·dk`, the radiating box modes. At M = 0.1 in a 2π box the record's
   Nyquist only reaches k = 6.9, so the band came back with **6** frequencies where 24
   were requested — a test caught it. It now samples the radiating box modes
   log-uniformly in k from dk to ω_max/c₀, which is still physically anchored on both
   edges and still record-length independent, and returns the full 24.

## Not done, and why

- **The L=4π and k≥2 rows of the HANDOFF table are still pre-fix.** `scripts/04`
  cannot run on `data/turb2d_L4pi.npz` on this machine: at n=256 a 2-component field
  needs ~3.5 GB of intermediates and the box has 7 GB total. Pre-existing — those
  numbers came from `runs/box_L4pi/analysis.py`, which has its own memory-lean path,
  and `scripts/04` was never run at n=256. Fixing it means chunking
  `komega_spectrum` over time, which changes the estimation path every existing number
  depends on. That is Tier 1+, and it is now open item 1 in `HANDOFF.md`. The affected
  numbers are marked as pre-fix and not comparable in both `HANDOFF.md` and the README.
- **The noise-floor check was not re-run** (it is a `runs/*` script, same
  pre-fix situation). Logged as an open item; it should pass with more margin, since
  the cone share dropped ~3×.
- **`scripts/03` was not run end-to-end** — it needs a trained checkpoint, and Stage 3
  is out of scope. Its new logic lives in `greens.select_band` and is unit-tested there
  precisely because the script itself cannot be reached. The far-field path was
  smoke-tested standalone: 14.2M kernel evals (cap 5e7), 1.0 s, all 24 frequencies
  nonzero. The old band produced 11× *larger* pressure, confirming it was reporting
  trapped rather than radiated sound.

