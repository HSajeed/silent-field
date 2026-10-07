# HANDOFF — start here

Entry point for a new session. Full audit trail: `rerun_log_2026-09-15.md`
(section "Follow-up session", B1–B8) and `branch_analysis.md` (2026-09-29).
Build plan: `acoustic-blindness-build-plan.md` (moved into this repo 2026-09-30).
Tier-0 correctness work: `TIER0_PLAN.md`. Read those only when an item below says so.
**Post-Tier-0 review (2026-10-01): `review_2026-10-01.md` — read before Stage 3.** It finds the
far field / R_S use the periodic-box-unsafe S-form (7–168× artifact; fix = `far_field_pressure_ibp`),
reframes Stage 4, corrects the M⁸/compactness decision, and audits `branch_analysis.md`.
**Literature check (2026-10-07): `lit_review_2026-10-07.md`** — PDE-Refiner is the closest framing; no prior (k, ω)/sonic-cone surrogate analysis found (limited search).
**Colab notebooks (2026-10-01): `notebooks/README.md`.** Chapters 00–07 rerun the whole history
(old commit `2535bb4` beside current code, every logged number asserted) and lay out Stages 3–4.
Branch `tier0-review-notebooks`; remote `origin` = github.com/HSajeed/silent-field (private).
**Session 2026-10-07:** review §5 steps 1–3 done (T-form far field + R_T; Stage-4 factor-2 fix;
Mach scaling: radiated power follows the compact 2D law). Details in "Decisions", "Stage 4",
"Open items". Stage 3 (training) is next.

## Project in three lines
Claim: L2-trained surrogates are blind to the acoustic field because the radiating
cone (ω ≥ c0|k|) holds almost none of the energy, not because sound is high-k.
Stages 1–2 (gate plus the k–ω headline metric) are done and hardened (Tier 0 on
2026-09-30; T-form and Stage-4 fixes on 2026-10-07). Stage 3 (training) is not started.
Stage 4 is an *analysis*, not a training run.

## Current conclusions (M=0.1 unless noted)
All velocity numbers below are **post-Tier-0** and carry the exact cone mask.

| Measure | L=2π, forcing k=8 (baseline) | L=2π, forcing k=16 | L=4π, forcing k=8 |
|---|---|---|---|
| R, all k (velocity) | **3.2e-08** (band [8.1e-09, 3.2e-08], F=3.9) | 1.4e-07 ([8.4e-08, 1.8e-07], F=2.1) | 2.4e-07 ([1.8e-07, 2.4e-07], F=1.3) |
| R, \|k\|≥2 (absolute) | 2.0e-10 ([6.1e-11, 2.0e-10], F=3.4) | 2.9e-09 ([2.1e-09, 4.4e-09], F=2.1) | 6.9e-11 ([6.9e-11, 9.4e-11], F=1.35) |
| R_S (Lighthill source, S-form) | **2.3e-09** (band [1.1e-09, 2.3e-09], F=2.1) | 2.6e-09 ([2.1e-09, 3.0e-09], F=1.5) | 3.3e-09 ([3.0e-09, 3.3e-09], F=1.1) |
| R_T (Lighthill source, T-form, `scripts/07`) | **4.9e-09** ([2.4e-09, 4.9e-09], F=2.0) | 3.0e-09 ([2.3e-09, 3.5e-09], F=1.55) | 3.2e-09 ([3.2e-09, 3.5e-09], F=1.10) |
| Cone share, **trained** Sobolev objective (λ=1e-2) | **3.2e-08 — 1.000× L2** | 1.4e-07 — 1.000× L2 | — |
| Cone share, pure multiplicative H¹ (not trained) | 1.7e-09 — 0.053× L2 | 2.4e-09 — 0.017× L2 | — |

Bands = envelope over {full, first half, second half}; F = max/min of that envelope.
- **All R / R_S cells now come from `scripts/04`** (per-component spectrum,
  `komega.velocity_komega_spectrum`; identical to the stacked transform to 1e-12; works at
  n=256, peak RSS 2.68 GB for L4π). No more "full record only" cells. Non-baseline datasets
  write `figures/komega_metrics_<stem>.json`. Pre-fix values were 7.5e-07 / 1.3e-10 / 7.3e-09
  for the 4π box and 3.8e-10 for the baseline k≥2 row.
- **R(all k) depends on the domain, before and after the fix.** The 4π box gives 7.4×
  the baseline at M=0.1 (8.7× pre-fix). Cone energy sits in the largest box mode (95% in
  \|k\|=1 at M=0.1). Don't quote it as a flow property at M≤0.1.
- **"R_S is robust within 1.4×" is superseded.** It held at M=0.1 only; at M=0.05 R_S spans
  5.3e-12 / 1.6e-11 / 2.3e-11 (4.3×). R_S misses the sub-dk (compact) radiating content:
  R_T/R_S at M=0.05/0.1/0.2/0.3 is turb2d 14.2 / 2.15 / 1.16 / 1.08, kf16 2.88 / 1.13 / 1.05 / 1.01,
  L4π 2.07 / 0.97 / 1.01 / 0.99. R_S agrees with R_T within 16% at M ≥ 0.2.
- **R_T (`tform.cone_share_refined`) full record by Mach (0.05 / 0.1 / 0.2 / 0.3):**
  turb2d 7.5e-11 / 4.9e-09 / 3.9e-07 / 5.9e-06; kf16 4.7e-11 / 3.0e-09 / 2.3e-07 / 3.6e-06;
  L4π 4.7e-11 / 3.2e-09 / 2.5e-07 / 3.9e-06. Sub-dk part at M=0.1: 3.2e-09 (turb2d, of 4.9e-09),
  8.1e-10 (kf16), 6.6e-10 (L4π).
- **R_T acceptance (review §5 step 1), stated honestly.** The 2π/4π point-estimate ratio is
  4.27 / 1.43 / 1.35 / 1.39 for R_S and 1.60 / 1.55 / 1.54 / 1.51 for R_T (M = 0.05/0.1/0.2/0.3),
  so the literal criterion (R_T changes less than R_S) is met only at M=0.05. Mitigating:
  R_T's half-record bands overlap across all three datasets at every Mach (common range at
  M=0.1: [3.16, 3.46]e-9), while R_S's do not at M ≤ 0.1 (M=0.1: [1.07, 2.27], [2.07, 3.01],
  [2.95, 3.26]e-9); and the same-flow window test shows R_T is box-size independent within
  scatter (Decisions). The residual 2π-vs-4π gap is attributed to two different flows with
  finite records (baseline F≈2). **That attribution is an interpretation, not proven.**
- **Noise floor passes.** The float32 floor is ~13 orders below the spectrum at the
  M=0.1 threshold. *(run pre-Tier-0; the cone share has since dropped ~3×, so the
  margin is larger, not smaller. Worth re-running.)*
- **Cone share ≤ ~1e-5 at M≤0.1 everywhere.** The qualitative claim holds; the
  Tier-0 mask fix moved the numbers down, not up.
- **Energy-spectrum slope:** −3.98 on k∈[12,36], time-averaged. No −3 range (linear
  drag plus ~5× scale separation). Unchanged by Tier 0.
- **Radiated power scales as the compact 2D law** (exponent ≈ 4 at fixed U, = M⁷ at fixed c0) on all three datasets, while the cone share R has local exponents 4.5–11. The blindness (R ≪ 1) is not a statement about how loud the flow is. See Decisions.
- **Sobolev weighting can never raise the cone share** (w_eff ≥ 1, ≈ 1 in the cone). At the
  default λ it is idle everywhere; where it matters it lowers the share. See "Stage 4" below.


## Decisions already made (don't reopen)
- The Green's function is outgoing for numpy's e^{+iωt} convention: G = (−i/4)H₀⁽²⁾(kr). The gate's CHECK 3 (1/√r decay) and CHECK 4 (dφ/dr = −k) guard it.
- Uncertainty band = envelope over {full record, first half, second half}.
- **Mach scaling (corrected 2026-10-07, review §3).** The steep local exponents (10.7 / 7.1 / 5.7) are exponents of R, an energy *fraction*, and cannot be compared with a radiated-power law (category error, review §3.4). The radiated power itself, `tform.radiated_power` (T-form far field, tapered, exact Fraunhofer form; test ties it to `far_field_pressure_ibp` to 1e-3), follows the **compact 2D quadrupole law**: at fixed flow (c0 = U/M) the local exponent d ln P/d ln M over M = 0.02–0.4 is 3.91–3.99 (turb2d), 3.86–4.28 (kf16), 3.89–4.21 (L4π); half records 3.69–4.31. P/P_compact = 0.87–1.00 / 0.84–1.03 / 1.00–1.12, so the compact (box-integrated stress) term carries the sound. Fixed-U exponent 4 ⇔ P ∝ ρU⁷ℓ/c0⁴ ⇔ the 2D "M⁷" law at fixed c0 [cited: Schoder et al., arXiv:2608.19236, "the 𝑀7 power law expected for compact quadrupole sources in two dimensions (Ffowcs Williams 1969)", Mach varied "by changing Δ𝑈"; Ffowcs Williams 1969 itself not fetched]. M⁸ is the 3D law and does not apply. The old rationale ("compactness breaks at low Mach") was backwards: He ∝ M, so low Mach is the compact regime. `scripts/08_mach_scaling.py`, `figures/08_mach_scaling*.{json,png}`. The Gaussian random-sweeping model (erfc(1/M)) for R was checked and does not fit.
- Resolution test = integrated cone energy per estimator (Hann4, BH4, DPSS4, Hann-full) agreeing within 1.5×.
- **The cone mask is evaluated on a radially resolved grid**, not on bin centres. See Tier 0 item 3.
- **"The Sobolev weighting" is two different objectives.** `losses.sobolev_loss` is additive; analysing it multiplicatively is a different objective. See Tier 0 item 2.
- The Sobolev weight uses angular k (length-aware), normalised by its own mean over the rfft2 half-plane, exactly as `losses._k_grid` does (in float32 — the analysis agrees to ~1e-7, not exactly).
- **Sobolev c-factor convention (2026-10-07).** Per full-plane mode the weight is `1 + λ·c·w̃(k)`: c = N_cells/N_half on the rfft2 kx=0 and Nyquist columns, N_cells/(2N_half) elsewhere (each interior coefficient stands for ±k); on a shell c = N_cells/(2N_half)·(1+`shell_axis_fraction(k, grid)`). The trained loss is anisotropic: kx=0/Nyquist columns get 2× the Sobolev weight.
- **The far field is the T-form (2026-10-07).** `scripts/03` uses `far_field_pressure_ibp(..., far_field=False)` (exact ∂i∂j of (−i/4)H₀⁽²⁾(kr)), temporal mean of T_ij removed, Tukey(0.5) spatial taper, band `radiating_band(mode="omega")` (default `--band-mode omega`: log-spaced from the first nonzero bin to Nyquist, **no c0·dk cut**, because the compact term radiates below c0·dk), observer radius = max(30L, 10/k_min) so kr ≥ 10. JSON gains `far_field_complex`, `far_field_form`, `far_field_radius`. Smoke test (turb2d, M=0.1, pred = truth + 5% white noise): 22 frequencies, ω ∈ [0.154, 78.5], 14 below c0·dk=11.31, r=736, kr ∈ [10, 5110], 49% of the stress's ω>0 power; |p| error 26.9%, complex 27.4%; truth vs truth 0. **Never run on a trained model.**
- **Gate CHECK 5 rationale.** The S-vs-T gap on the vortex pair was ~6%; it is the box surface term, not a bug: with exact kernels |p_S−p_T|/|p_T| at the tone = 5.26% (L=2π), 0.73% (4π), 0.05% (8π), identical at r = 10L0, 40L0, 160L0. The pair has net circulation, so its velocity does not vanish at the box edge. The asymptotic ∂i∂jG kernel adds O(1/kr): 2.6% at kr=77, 0.64% at kr=308; the old 5.9% (r=40L0) is the two combined. CHECK 5 (exact-kernel S vs T, L=2π n=128 and 4π n=256) asserts <2% at 4π and shrinking ≥3× with L. Result 0.0526 → 0.0073, gate PASSED.
- **R_T definition (`tform.cone_share_refined`, `scripts/07_tform_metric.py`).** Cone share of S̃ = −k_i k_j T̃_ij, T̃ = continuous Fourier transform of the box-truncated, dealiased stress (Sarkar & Hussaini's "finite volume embedded in a medium at rest"), on a wavenumber grid `pad`=4 times finer than the box's, so 0<|k|<dk (compact term) is included. Denominator = total S power, same treatment. Bands as `scripts/04` (Welch-4/Hann/half-record). pad=1 and no taper reproduces R_S to 1e-9 (test); pad 4 vs 8 within 2%. **Tukey(0.5) taper is the default:** a hard edge leaks convective-ridge content into small k via the box sinc. Same-flow test (M=0.1): 2π windows cut from the 4π flow gave 1.36× the full-box R_T untapered, 1.11× tapered (window scatter 2.8–4.0e-9 vs full-box 3.16e-9). Full-box R_T 4.77e-9 untapered; 3.27 / 3.16 / 3.04e-9 for Tukey α = 0.25 / 0.5 / 1.

## Stage 4 — resolved without a training run
`python scripts/06_sobolev_argument.py` (~1 min, no model, no torch).

The trained Sobolev objective applies, per full-plane mode, the weight `1 + λ·c·w̃(k)` with c as in Decisions (N_cells/N_half on the kx=0 and Nyquist columns, N_cells/(2N_half) elsewhere; factor-2 bug fixed 2026-10-07, review §2.5). The weight's mass is at high k and the cone is at low k (k ≤ 6.7, 1.2% of the resolved |k| values), so `max|w_eff − 1|` in the cone at the default λ is **1.7e-4** (was 3.3e-4 with the bug). At λ=1e-2 the Sobolev term moves the cone share by 1.3e-4, ~0.01% of the objective (review said ~0.03% pre-fix).

Share ÷ L2 at M=0.1, new (old):

| λ | new | old |
|---|---|---|
| 1e-3 | 1.000 | 1.000 |
| 1e-2 | 1.000 (0.99987) | 1.000 |
| 1e-1 | 0.999 | 0.997 |
| 1 | 0.987 | 0.974 |
| 10 | 0.883 | 0.794 |
| 1e2 | 0.452 | 0.303 |
| 1e3 | 0.134 | 0.086 |
| 1e4 | 0.081 | — |
| 1e6 | 0.075 | 0.053 |

Pure multiplicative H¹ (not trained) is still 0.053 (~19×). The trained objective's floor is
0.075 (~13×) at λ=1e6; the pure-H¹ value is not reachable by it.

**Claim (review §2.2):** w_eff ≥ 1 everywhere and ≈ 1 in the cone, so Sobolev weighting can
never raise the cone share. At the default λ it is idle everywhere, not specifically in the
cone; where it matters it lowers the share, by up to ~13× for the trained form (~19× for the
standard multiplicative H¹, e.g. neuraloperator `H1Loss`). "Identical to L2 at 100× default λ"
is now "within 1.3% at λ=1 (100× default)". The old circular test is replaced by one through
torch `sobolev_loss` on single cosines (n=64, λ=0.37; matches to <1e-5 for (3,5) 1.8221e-02,
(0,4) 1.7701e-02, (7,0) 2.6031e-02 and Nyquist modes (0,32), (32,0), (32,5)).
`figures/06_sobolev_argument.{json,png}` and `figures/komega_metrics.json` regenerated. A
λ-sweep training run is optional confirmation, not the argument.


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
   carries a radially *resolved* grid (`k_resolved` / `power_resolved`, 1621 integer shells
   at n=128, grouped on `rint(k²/dk²)`; was 1850 before 2026-10-07, metric unchanged to 6e-16;
   O(n) not O(n²), built with one chunked `np.add.reduceat`) and the metric uses it. `radiating_mask` is kept at its old shape for plotting and the archived
   `runs/*` scripts, which are **not** re-run and will not reproduce these numbers.
   Headline moved **down** (claim strengthened): R 8.7e-08 → 3.2e-08, R_S 1.0e-08 → 2.3e-09.
   Also added the `k > 0` guard, and corner modes (|k| up to 90.5) are now inside the
   total — worth <1% here.
2. **Sobolev convention** (`radiating.py`, `04`, new `06`). The analysis used
   `(1+k²)^order` multiplicatively; `losses.sobolev_loss` trains `base + λ·high`, i.e.
   the per-mode weight is `1 + λ(N_cells/N_half)(1+k²)^order` (factor corrected 2026-10-07; see Stage 4). At the default
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

Tests: **21 → 42** (64 after 2026-10-07). Gate `00` re-run, unchanged, PASS (CHECK 5 added 2026-10-07). Fix 3's ω ≥ c₀·dk band is superseded by `--band-mode omega` (Decisions).

## Method pitfalls (learned the hard way)
- Compare power **density** (P/dω) across estimators with different segment lengths. Per-bin power is 2.5× off.
- Single-bin comparisons against a full-record periodogram (2 dof) are noise. Integrate instead.
- "The spectrum is still decaying" can be window leakage (Hann4 beyond ω≈35).
- RAM is ~7 GB: compute spectra as power(u)+power(v) separately, never a stacked float64 array. `scripts/04` now does this (`komega.velocity_komega_spectrum`), so it runs n=256 at 2.68 GB peak.
- `turb2d.run` progress prints come from the recording loop, not spin-up. The manifest does **not** record spin-up.
- `losses._k_grid` builds its wavenumbers in **float32** (torch's default), so the analysis can only match it to ~1e-7, not exactly. Don't write a tighter test.
- A radial bin on a square grid is not a single |k|. Any mask or threshold applied after binning inherits that error.
- **The S-form far field in a periodic box contains a surface term** (net circulation → nonzero velocity at the box edge; Sarkar & Hussaini Eq. 1 vs Eq. 2). Vortex pair: 5.26% (2π), 0.73% (4π), 0.05% (8π). Turbulence: 7–168× (review §1.3). Use the T-form.
- **Hard box truncation leaks convective-ridge content into small k** through the box sinc. Taper (Tukey 0.5) before any sub-dk or compact-term estimate; untapered, 2π windows of the 4π flow give 1.36× the full-box R_T (1.11× tapered).
- **The asymptotic ∂i∂jG kernel has an O(1/kr) error** (2.6% at kr=77, 0.64% at kr=308 on the vortex pair). Use the exact kernel (`far_field=False`) unless kr is large.

## Existing data and artifacts (don't regenerate)
- `data/turb2d.npz` — baseline, manifest 111e13e318c9, spin-up 5000.
- `data/turb2d_kf16.npz` — forcing k=16, manifest e5d144b83287, spin-up 10000.
- `data/turb2d_L4pi.npz` — 4π box, n=256, 496 MB, manifest 34d5ea324579, spin-up unverifiable.
- `runs/noise_floor_check/` and `runs/box_L4pi/` — analysis.py, results.json, figures.
  **Both are pre-Tier-0.** Their inline arithmetic is on the binned grid and will not
  reproduce current numbers. Kept as historical records.
- `figures/komega_metrics.json` — headline table, post-Tier-0, with the λ-sweep per Mach.
- `figures/06_sobolev_argument.json` / `.png` — Stage-4 λ sweep and the geometric reason (regenerated 2026-10-07).
- `figures/komega_metrics_turb2d_kf16.json`, `komega_metrics_turb2d_L4pi.json`, `04_komega_turb2d_kf16.png`, `04_komega_turb2d_L4pi.png` — `scripts/04` outputs for the other datasets.
- `figures/tform_metrics.json`, `tform_metrics_turb2d_kf16.json`, `tform_metrics_turb2d_L4pi.json` — R_T per Mach with bands and sub-dk part (`scripts/07`).
- `figures/08_mach_scaling{,_turb2d_kf16,_turb2d_L4pi}.{json,png}` — radiated power, compact limit, local exponents vs M (`scripts/08`).

## Commands (run from `silent-field/`)
```
python scripts/00_validate_acoustics.py --n 128 --outdir figures   # gate, must PASS
python scripts/04_komega_analysis.py                               # R table + bands + λ sweep (also works for n=256)
python scripts/05_make_figures.py                                  # slope, [12,36]
python scripts/06_sobolev_argument.py                              # Stage 4, no model needed
python scripts/07_tform_metric.py                                  # R_T (T-form cone share, sub-dk included)
python scripts/08_mach_scaling.py                                  # radiated power vs M, compact 2D law (~1.5 min; 4π ~6 min)
python -m pytest -q                                                # 64 passed
python scripts/02_train_surrogate.py --model fno --out runs/fno_mse_01   # Stage 3 (GPU); --smoke for a quick check
python scripts/03_compute_sources.py --model runs/fno_mse_01/model.pt --out runs/fno_mse_01
python scripts/09_blindness.py --model runs/fno_mse_01/model.pt --out runs/fno_mse_01      # B, B_S + controls
python scripts/10_error_diagnostic.py --model runs/fno_mse_01/model.pt --out runs/fno_mse_01/diag   # jitter vs aliasing
python scripts/01_generate_data.py --length <L> --forcing-k <k> --spinup <n> --out <path>
```

## Open items (priority order)
1. **Manifest:** record `spinup` and `save_every` in the meta written by `scripts/01_generate_data.py`.
2. **Re-run the noise-floor check** post-Tier-0. It should pass with more margin, since
   the cone share dropped ~3×, but the unresolved cells should be re-examined against
   the exact mask.
3. **Müller–Obermeier absolute-amplitude check** in the gate — superseded by Rucz et al.
   2024 (open access, error bounds vs. the exact parameters varied here). See
   `branch_analysis.md` §6C.
4. **`min_kr` diagnostic in `greens.py`.** `scripts/03` now enforces kr ≥ 10 with the exact
   kernel, but gate CHECKs 2–4 still use the asymptotic S-form kernel (kr = 0.015). Errors
   there were measured at 0.02–0.12% (audit §2.3) — luck, not design, and untested.
5. **Comment nits in `scripts/00_validate_acoustics.py`:** preliminary checks still
   carry "CHECK 1/2" comment headers, and the CHECK 2 comment says "four-lobe".
6. **Packaging:** `pyproject.toml` (git, `.gitignore` excluding `data/`, and the GitHub remote now exist).
7. **Stage 3:** FNO/UNet training (`scripts/02`, `03`). Not started, but Colab-ready
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
   Runs: `fno_h1_01` (`--loss h1`), `fno_mse_unroll4` (`--unroll 4`, ~4× the FNO time). For any trained Stage-4 run use the standard multiplicative H¹ loss
   (neuraloperator `H1Loss` style), not the additive λ=1e-2 form (review §2.3).
8. **Notebooks: chapter refresh for 2026-10-07 changes — see notebooks/README.md.**

## Working style
The main agent orchestrates. It delegates execution to the `coder` and `logger` subagents
(`~/.claude/agents/`), assigns file ownership per agent so parallel agents don't collide, and
critically checks their reports before logging.

