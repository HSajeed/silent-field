# HANDOFF — start here

Entry point for a new session. Full audit trail: `rerun_log_2026-09-15.md`
(section "Follow-up session", B1–B8) and `branch_analysis.md` (2026-09-29).
Build plan: `acoustic-blindness-build-plan.md` (moved into this repo 2026-09-30).
Tier-0 correctness work: `TIER0_PLAN.md`. Read those only when an item below says so.
**Post-Tier-0 review (2026-10-01): `review_2026-10-01.md` — read before Stage 3.** It finds the
far field / R_S use the periodic-box-unsafe S-form (7–168× artifact; fix = `far_field_pressure_ibp`),
reframes Stage 4, corrects the M⁸/compactness decision, and audits `branch_analysis.md`.
**Colab notebooks (2026-10-01): `notebooks/README.md`.** Chapters 00–07 rerun the whole history
(old commit `2535bb4` beside current code, every logged number asserted) and lay out Stages 3–4.
Branch `tier0-review-notebooks`; remote `origin` = github.com/HSajeed/silent-field (private).

## Project in three lines
Claim: L2-trained surrogates are blind to the acoustic field because the radiating
cone (ω ≥ c0|k|) holds almost none of the energy, not because sound is high-k.
Stages 1–2 (gate plus the k–ω headline metric) are done and hardened, and were
corrected on 2026-09-30 — see "Tier 0" below. Stage 3 (training) is not started.
Stage 4 is now an *analysis*, not a training run.

## Current conclusions (M=0.1 unless noted)
All velocity numbers below are **post-Tier-0** and carry the exact cone mask.

| Measure | L=2π, forcing k=8 (baseline) | L=2π, forcing k=16 | L=4π, forcing k=8 |
|---|---|---|---|
| R, all k (velocity) | **3.2e-08** (band [8.1e-09, 3.2e-08], F=3.9) | 1.4e-07 (F=2.1) | 2.4e-07 (full record only) |
| R, \|k\|≥2 (absolute) | 2.0e-10 (full record only) | 2.9e-09 (full record only) | 6.9e-11 (full record only) |
| R_S (Lighthill source) | **2.3e-09** (band [1.1e-09, 2.3e-09], F=2.1) | 2.6e-09 (F=1.5) | 3.3e-09 (full record only) |
| Cone share, **trained** Sobolev objective (λ=1e-2) | **3.2e-08 — 1.000× L2** | 1.4e-07 — 1.000× L2 | — |
| Cone share, pure multiplicative H¹ (not trained) | 1.7e-09 — 0.053× L2 | 2.4e-09 — 0.017× L2 | — |

- **The L=4π column and the \|k\|≥2 row are now post-Tier-0 (2026-10-01).** They come
  from `notebooks/03_komega_headline` §6, which transforms one velocity component at a
  time (`notebooks/nbtools/spectra.py`, identical to the stacked transform). That path
  peaks at ~2.7 GB for n=256 here, so no chunking refactor was needed. Same estimator as
  `scripts/04` (Welch 4 segments), but no half-record bands yet for those cells.
  Pre-fix values were 7.5e-07 / 1.3e-10 / 7.3e-09 for the 4π box and 3.8e-10 for the
  baseline k≥2 row.
- **R(all k) depends on the domain, before and after the fix.** The 4π box gives 7.4×
  the baseline at M=0.1 (8.7× pre-fix). Cone energy sits in the largest box mode (95% in
  \|k\|=1 at M=0.1). Don't quote it as a flow property at M≤0.1. The likely cause is the
  S-form source, not the box size: see `review_2026-10-01.md` §1.
- **R_S is the most robust number:** within **1.4×** across baseline, kf16 and L=4π
  post-fix. It is still built on S, so robust is not the same as physically complete
  (review §1).
- **Noise floor passes.** The float32 floor is ~13 orders below the spectrum at the
  M=0.1 threshold. *(run pre-Tier-0; the cone share has since dropped ~3×, so the
  margin is larger, not smaller. Worth re-running.)*
- **Cone share ≤ ~1e-5 at M≤0.1 everywhere.** The qualitative claim holds; the
  Tier-0 mask fix moved the numbers down, not up.
- **Energy-spectrum slope:** −3.98 on k∈[12,36], time-averaged. No −3 range (linear
  drag plus ~5× scale separation). Unchanged by Tier 0.
- **Sobolev is inert, not harmful.** See "Stage 4" below.


## Decisions already made (don't reopen)
- The Green's function is outgoing for numpy's e^{+iωt} convention: G = (−i/4)H₀⁽²⁾(kr). The gate's CHECK 3 (1/√r decay) and CHECK 4 (dφ/dr = −k) guard it.
- Uncertainty band = envelope over {full record, first half, second half}.
- Mach scaling is described by local exponents (10.7 / 7.1 / 5.7), not a power law. The Gaussian random-sweeping model (erfc(1/M)) was checked and does not fit. **M⁸ must be retired the same way**: it requires a *compact* source, and compactness is ωL/c₀ ~ M, so the low-Mach limit is precisely where it breaks. Steep exponents are expected, not anomalous.
- Resolution test = integrated cone energy per estimator (Hann4, BH4, DPSS4, Hann-full) agreeing within 1.5×.
- **The cone mask is evaluated on a radially resolved grid**, not on bin centres. See Tier 0 item 3.
- **"The Sobolev weighting" is two different objectives.** `losses.sobolev_loss` is additive; analysing it multiplicatively is a different objective. See Tier 0 item 2.
- The Sobolev weight uses angular k (length-aware), normalised by its own mean over the rfft2 half-plane, exactly as `losses._k_grid` does (in float32 — the analysis agrees to ~1e-7, not exactly).

## Stage 4 — resolved without a training run
`python scripts/06_sobolev_argument.py` (~1 min, no model, no torch).

The trained Sobolev objective applies the per-mode weight `1 + λ·(N_cells/N_half)·(1+k²)^order`. The second term is 7.3e-4 … 1.7e-2 of its own mean everywhere in the cone, because the cone is at low k (it reaches only k ≤ 6.7, i.e. 1.2% of the resolved |k| values) and the weight's mass is at high k. So `|w_eff − 1| ≤ 3.3e-4` across the whole cone and the objective is **identical to L2 there**, at every Mach tested:

| λ | 1e-3 | 1e-2 | 1e-1 | 1 | 10 | 1e2 | 1e3 | 1e6 |
|---|---|---|---|---|---|---|---|---|
| share ÷ L2 | 1.000 | 1.000 | 0.997 | 0.974 | 0.794 | 0.303 | 0.086 | 0.053 |

The 0.053 asymptote is the *pure multiplicative* reading — a different objective, and
the source of the old "10× worse" claim. The corrected claim is stronger and needs
no training: the standard remedy is inert here, for 100× the default λ, because the
cone is starved by where the weight's mass is rather than by its exponent. A
λ-sweep training run is now optional confirmation, not the argument.


## Tier 0 — correctness, done 2026-09-30 (`TIER0_PLAN.md`)

Three bugs fixed, all found by re-deriving the numbers rather than reading the logs.

1. **Cone mask on bin centres** (`komega.py`). A bin on a square grid holds several
   distinct |k| (5, √26, √29 all land in the bin centred on 5; an earlier version of this
   note said √21, but no grid mode has |k|=√21 since 21 is not a sum of two squares), and testing them all
   against the centre admits a slab of non-radiating modes near the cone edge.
   Measured inflation: **2.84× (velocity), 4.79× (source)** on a single full-record Hann
   estimate (`TIER0_PLAN.md:34`); on the headline Welch-4 estimate the same fix is
   **2.71× / 4.52×** (8.69e-08 → 3.20e-08, 1.03e-08 → 2.27e-09). Quote the one matching the
   number being corrected. `KOmegaSpectrum` now
   carries a radially *resolved* grid (`k_resolved` / `power_resolved`, ~1850 values at
   n=128, O(n) not O(n²), built with one chunked `np.add.reduceat`) and the metric uses
   it. `radiating_mask` is kept at its old shape for plotting and the archived
   `runs/*` scripts, which are **not** re-run and will not reproduce these numbers.
   Headline moved **down** (claim strengthened): R 8.7e-08 → 3.2e-08, R_S 1.0e-08 → 2.3e-09.
   Also added the `k > 0` guard, and corner modes (|k| up to 90.5) are now inside the
   total — worth <1% here.
2. **Sobolev convention** (`radiating.py`, `04`, new `06`). The analysis used
   `(1+k²)^order` multiplicatively; `losses.sobolev_loss` trains `base + λ·high`, i.e.
   the per-mode weight is `1 + λ(N_cells/N_half)(1+k²)^order`. At the default
   `sobolev_weight` the trained objective gives the cone **1.000× the L2 share**;
   the multiplicative reading gives 0.053×. The old "10× worse" was the second number
   reported for the first objective. `figures/06_sobolev_argument.{json,png}` + the
   λ-sweep table in `figures/komega_metrics.json`.
3. **Far-field band by array index** (`03:78`, `greens.select_band`). `[:24]` took the
   24 *lowest* frequencies; at M=0.1 that is ω ∈ [0.15, 3.68] and **0 of 24** satisfy
   ω ≥ c₀k. 91% of the source's ω>0 power is below c₀·dk, on the convective ridge. Now
   selected by the radiation condition, log-spaced in |k| from dk to ω_max/c₀, which
   is record-length independent (kr lands at 189–1309, so the far-field asymptotic is
   comfortably valid — this retires audit §2.3 for this script). `auto` falls back to
   the strongest radiating frequencies if the physical band sees <1% of the radiating
   power, and says so.

Tests: **21 → 42**. Gate `00` re-run, unchanged, PASS.

## Method pitfalls (learned the hard way)
- Compare power **density** (P/dω) across estimators with different segment lengths. Per-bin power is 2.5× off.
- Single-bin comparisons against a full-record periodogram (2 dof) are noise. Integrate instead.
- "The spectrum is still decaying" can be window leakage (Hann4 beyond ω≈35).
- RAM is ~7 GB: compute spectra as power(u)+power(v) separately, never a stacked float64 array. `scripts/04` still stacks, so it only works on the n=128 datasets. The per-component path in `notebooks/nbtools/spectra.py` runs n=256 at ~2.7 GB peak (verified 2026-10-01).
- `turb2d.run` progress prints come from the recording loop, not spin-up. The manifest does **not** record spin-up.
- `losses._k_grid` builds its wavenumbers in **float32** (torch's default), so the analysis can only match it to ~1e-7, not exactly. Don't write a tighter test.
- A radial bin on a square grid is not a single |k|. Any mask or threshold applied after binning inherits that error.

## Existing data and artifacts (don't regenerate)
- `data/turb2d.npz` — baseline, manifest 111e13e318c9, spin-up 5000.
- `data/turb2d_kf16.npz` — forcing k=16, manifest e5d144b83287, spin-up 10000.
- `data/turb2d_L4pi.npz` — 4π box, n=256, 496 MB, manifest 34d5ea324579, spin-up unverifiable.
- `runs/noise_floor_check/` and `runs/box_L4pi/` — analysis.py, results.json, figures.
  **Both are pre-Tier-0.** Their inline arithmetic is on the binned grid and will not
  reproduce current numbers. Kept as historical records.
- `figures/komega_metrics.json` — headline table, post-Tier-0, with the λ-sweep per Mach.
- `figures/06_sobolev_argument.json` / `.png` — Stage-4 λ sweep and the geometric reason.

## Commands (run from `silent-field/`)
```
python scripts/00_validate_acoustics.py --n 128 --outdir figures   # gate, must PASS
python scripts/04_komega_analysis.py                               # R table + bands + λ sweep
python scripts/05_make_figures.py                                  # slope, [12,36]
python scripts/06_sobolev_argument.py                              # Stage 4, no model needed
python -m pytest -q                                                # 42 passed
python scripts/01_generate_data.py --length <L> --forcing-k <k> --spinup <n> --out <path>
```

## Open items (priority order)
1. **Switch the far field to the T-form** (review §5, step 1). Prerequisite measured
   2026-10-01 (`notebooks/01_validation_gate` §6): on the vortex pair at its tone the T-form
   (`far_field_pressure_ibp`) matches the S-form to ~2% in magnitude, ~6% as complex
   fields. At 4Ω (weak harmonic) magnitudes agree ~10% but phases differ ~70%. Next: choose
   a gate tolerance (understand the 6% first), add the gate check, switch `scripts/03`.
2. **Port the per-component spectrum into `scripts/04`** (replaces "chunk `komega_spectrum`").
   Post-fix L=4π and \|k\|≥2 numbers already exist (table above), but without
   half-record bands and not from the script.
3. **Manifest:** record `spinup` and `save_every` in the meta written by `scripts/01_generate_data.py`.
4. **Re-run the noise-floor check** post-Tier-0. It should pass with more margin, since
   the cone share dropped ~3×, but the unresolved cells should be re-examined against
   the exact mask.
5. **Müller–Obermeier absolute-amplitude check** in the gate — superseded by Rucz et al.
   2024 (open access, error bounds vs. the exact parameters varied here). See
   `branch_analysis.md` §6C.
6. **`min_kr` diagnostic in `greens.py`.** The far-field asymptotic is now used at
   kr ≈ 10³ by `scripts/03`, but the gate still uses it at kr = 0.015. Errors there were
   measured at 0.02–0.12% (audit §2.3) — luck, not design, and untested.
7. **Comment nits in `scripts/00_validate_acoustics.py`:** preliminary checks still
   carry "CHECK 1/2" comment headers, and the CHECK 2 comment says "four-lobe".
8. **Packaging:** `pyproject.toml` (git, `.gitignore` excluding `data/`, and the GitHub remote now exist).
9. **Stage 3:** FNO/UNet training (`scripts/02`, `03`). Not started. `03` is now
   instrumented (band diagnostics in `03_errors.json`) but has never been run.

## Working style
The main agent orchestrates. It delegates execution to the `coder` and `logger` subagents
(`~/.claude/agents/`), assigns file ownership per agent so parallel agents don't collide, and
critically checks their reports before logging.

