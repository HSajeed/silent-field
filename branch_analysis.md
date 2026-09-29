# Project Analysis — `silent-field`

**Date:** 2026-09-29
**Scope:** Full read of `HANDOFF.md`, `README.md`, `rerun_log_2026-09-15.md`,
`../acoustic-blindness-build-plan.md`, all of `src/`, `scripts/`, `tests/`, and
the JSON artifacts under `figures/` and `runs/`. Test suite executed. Several
claims re-derived numerically. External prior-art and physics-literature search performed.

**Status of this file:** analysis only. No source file, script, test, or existing
markdown document was modified in producing it.

---

## 1. What this project is

A physics-informed ML critique. The claim (`HANDOFF.md:8`, `README.md:3`): neural PDE
surrogates trained with L2 are blind to the acoustic field, but **not** for the reason
the literature assumes. Spectral bias says low-wavenumber content is learned first — and
acoustics *is* low-wavenumber (`λ_acoustic/λ_hydro ≈ 1/M`). So spectral bias predicts
acoustics should be easy. It is destroyed anyway, because the "radiating cone"
`ω ≥ c₀|k|` in (k,ω) space holds a vanishing fraction of the energy, and an L2 loss
weights by energy.

Four stages are laid out in `../acoustic-blindness-build-plan.md`:

| Stage | Content | State |
|---|---|---|
| 1 | Acoustic validation gate | done, hardened |
| 2 | The k–ω headline metric | done, hardened |
| 3 | Surrogate training (error triple) | **not started** |
| 4 | Sobolev counterexample | **not started** |

**Repo state:** ~2,700 lines of numpy/scipy plus a small torch FNO/UNet. 21/21 tests
pass. Three datasets on disk (118 MB, 125 MB, 496 MB). Two analysis runs preserved under
`runs/`. The code quality is genuinely high — the docstrings encode hard-won method
pitfalls, and `HANDOFF.md` is an unusually honest audit trail that records its own
mistakes and corrections.

---

## 2. Verified findings — including three real bugs

Claims below were checked by execution, not by reading the logs. Tests pass 21/21.
Three issues surfaced that are not recorded in `HANDOFF.md` or the build plan.

### 2.1 🔴 `scripts/03_compute_sources.py:78` — the far-field band contains zero acoustic frequencies

```python
band = [i for i in range(1, len(om)) if om[i] > 0][:24]
```

`[:24]` takes the **lowest** 24 frequencies: ω ∈ [0.15, 3.69]. At M = 0.1 with
c₀ = 11.3, radiation requires `ω ≥ c₀k ≥ 11.3`. Measured directly: **0 of the 24
selected frequencies are supersonic.**

The far-field error that Stage 3 reports — the number the entire "error triple"
contrast rests on (`src/diagnostics/errors.py:1-11`) — is therefore computed entirely
inside the hydrodynamically-trapped region. The script measures radiated-sound error
on non-radiating frequencies. It also scales with record length rather than selecting a
physically fixed band, so it is not even a stable quantity.

This is the most consequential bug in the repo, precisely *because* Stage 3 has never
been run: the fix is free right now, whereas after a training run it invalidates the
paper's central contrast.

### 2.2 🔴 The Sobolev counterexample measures a different objective than the one trained

`README.md:31-33` claims Sobolev training "makes it about ten times worse," backed by
`h1 = 4.6e-09` versus `l2 = 8.7e-08` in `figures/komega_metrics.json`. But the two
halves of the code define different objectives:

- `src/acoustics/radiating.py:35` — analysis treats the weight as **multiplicative**:
  `w = (1+k²)^order`, then `share = Σ(w·P·mask) / Σ(w·P)`.
- `src/models/losses.py:60` — training is **additive**:
  `return base + weight * high`, i.e. total weight `1 + λ(1+k²)`.

Measured on the actual baseline dataset at M = 0.1:

| objective | cone share |
|---|---|
| L2 | 8.69e-08 |
| **additive `1 + 0.01(1+k²)`** (what training does, default `sobolev_weight`) | **6.41e-08** — only 1.4× below L2 |
| multiplicative `(1+k²)` (what the analysis reports) | 4.58e-09 — 19× below L2 |

**The "10× worse" claim is an artifact of comparing against a weighting the code never
trains.** At the default Sobolev weight, H¹ barely moves the cone share at all.

Stage 4 is described in the build plan as "where the project stops being a demonstration
and becomes an argument" (`build-plan.md:291`), and its headline number does not survive
contact with the training code. This is recoverable and the corrected version is a
stronger argument: a λ sweep showing the cone share is nearly λ-independent in the
regime where Sobolev training actually operates is a much tighter statement of "the
standard remedy cannot work here" than a factor-of-10 that turns out to be a
normalization artifact.

### 2.3 🟡 A guardrail in `greens.py` that is never exercised where it matters

`greens.py:116` caps kernel evaluations at `max_kernel_evals = 5e7`. At `scripts/03`
defaults (24 freqs × 36 observers × 16,384 sources = 1.4e7) it passes. But the
far-field asymptotic is being used at `kr = 0.015` in the validation gate and
`kr ≤ 0.21` on the turbulence data — far inside the near field, where the Hankel
asymptotic is not valid.

Checked exact Hankel (`far_field=False`) against the far-field branch on real source data:

| ω | k·dx | rel. amplitude error |
|---|---|---|
| 6.14 | 0.027 | 0.12% |
| 18.41 | 0.080 | 0.04% |
| 30.68 | 0.133 | 0.02% |
| 49.09 | 0.213 | 0.02% |

**Not a live bug** — but the small error is luck, not design. Individually the near-field
source points carry little weight. There is no test asserting `far_field=True` is valid at
the `kr` values actually used, and no warning when it is not. A `min_kr` diagnostic would
close this.

### 2.4 🟡 The cone mask at k = 0 is degenerate

`komega.py:63` — at `k_bins[0] = 0` the condition `ω ≥ 0` makes the **entire** column
count as radiating. Harmless for the current data (zero mean flow, plus
`detrend_mean=True` at `komega.py:110`), but it means `R` has a silent floor
contribution. A `k > 0` guard makes the metric robust to a dataset that develops mean flow.

### 2.5 🟡 Spectral binning discards corner modes

`n_k_bins = n//2 = 64` covers `|k| ≤ 63.5`, but the grid resolves `|k|` up to
`√2·64 ≈ 90`. Corner modes are dropped by the `valid` mask at `komega.py:153`. The
energy is small, but "fraction of *total* energy" is then not quite total.

### 2.6 ✅ The validation gate is genuinely good

`scripts/00_validate_acoustics.py` now carries four checks, including the two the
independent audit log records as having been added. I re-verified the outgoing
convention independently: the Green's function is `H₀⁽²⁾` under numpy's `e^{+iωt}`,
and direct far-field computation agrees with the code to 0.02–0.12%.

The rotating-quadrupole subtlety — that a spinning quadrupole is nearly axisymmetric in
*magnitude* and carries its structure in the *phase* — is correctly identified and
correctly tested (`00_validate_acoustics.py:109-121`). That is a genuine trap, and
avoiding it is evidence the author understood the physics rather than pattern-matched.

---

## 3. The physics has a conceptual gap that Stage 3 must close

This is the deepest issue, and it is not flagged anywhere in `HANDOFF.md` or the
build plan.

**R measures the share of the *signal* in the cone. The loss is computed on the *error*.**
These are different quantities.

```
R        = ∫_cone |û|² / ∫ |û|²                    (signal — what is measured)
R_loss   = ∫_cone |ê|² / ∫ |ê|²,  ê = û_pred − û_true   (loss — what actually matters)
```

The claim "L2 assigns weight R to the cone" silently assumes the model's error is
*proportional to the field*, i.e. the model is uniformly bad in relative terms.
Computed on the actual dataset:

| error spectrum shape | cone share of the **loss** at M = 0.1 |
|---|---|
| `e ∝ û` (assumed by R) | 8.7e-08 |
| **white in (k,ω)** (typical of an undertrained model) | **6.2e-02** |
| `e ∝ k²û` (Sobolev-like) | higher still |

A spectrally white error field places **~6% of the total loss inside the cone — about
six orders of magnitude more than R.** The geometric area fraction of the cone at
M = 0.1 is 6.2% (verified: it is a triangle reaching `ω_max` at `k = 1`).

**Implication: R is the *optimistic* bound.** The true cone loss share is ≥ R, and
probably orders of magnitude larger.

This is a gift rather than a problem:

- **The corrected argument is far stronger.** "The cone holds ~1e-8 of the *signal*,
  while the model parks ~1e-2 of its own *loss* there, so the model is not merely
  unpenalised in the radiating subspace — it is failing *relative to its own weighting*."
- **It yields a better metric.** Define the **blindness ratio**

  ```
  B = R_loss / R_signal   ≥ 1
  ```

  B = 1 means the model is uniformly bad (loss weighting exactly matches signal
  distribution). B ≫ 1 means the model is *selectively* failing in the subspace that
  radiates. B is dimensionless, architecture-agnostic, and directly falsifiable.
- **It costs almost nothing.** It requires only Stage 3, already planned at
  `build-plan.md:279` (step 15: "k–ω spectrum of the prediction error"). No new data,
  no new solver, ~50 lines.

Checked: nothing in the current codebase computes this. `grep` for
`cone_weighted|divergence_penalty` shows `divergence_penalty` (`losses.py:63`) is
**defined but never called** by any training path. The docstring says this is
deliberate, but nothing enforces it — and it is a ready-made control for Stage 4.

---

## 4. Prior art — the novelty is real, but thinner than the plan assumes

Searched specifically for direct prior art. **None of the project's central claim was
found.** No paper computes the fraction of an ML training objective inside a
supersonic-phase-speed region of (k,ω) space, and none frames acoustic failure as
*amplitude*-based rather than *frequency*-based blindness.

### Closest works, and how to position

| Work | Relevance |
|---|---|
| **Duraisamy, arXiv:2604.20061** (Apr 2026) | The single closest paper. Makes exactly the "small MSE, large QoI error" argument via NTK + spectral bias, and calls for better reporting standards. **Its entire mechanism is frequency-based** (high-k under-resolution). The inversion here — failure at *low* k, caused by *energy* — is genuinely outside it. This is the best "here is what it does not cover" paragraph available. |
| **Khodakarami et al., arXiv:2602.19265** (Feb 2026) | Already cited in the plan. Confirms the field treats spectral bias as a low-vs-high *frequency* problem and that remedies are "spectral-aware losses," i.e. all aimed at high-k. Directly supports the Stage-4 punchline. |
| **Rucz, Ulveczki, Heinz & Schoder, J. Sound Vib. 118496 (2024)** | **Found — supersedes the Müller–Obermeier open item.** Open access. Derives the co-rotating vortex-pair far-field sound pressure *ab initio*, with closed-form source terms for the Scully vortex model, and establishes error bounds versus Mach number, core radius, and source truncation. A better and more modern validation target than Müller–Obermeier 1967, and free. It also quantifies exactly what `vortex_pair.py:22-26` currently declines to do: absolute amplitude. |
| **Kolster et al. (AIAA 1999) / NASA-CR-208109 (1997)** | Independently confirm that the acoustically active content is the *low*-wavenumber part of the source — i.e. **the Stage-2 finding is textbook in physics and unknown in ML.** Cuts both ways: strong physics backing, but weakens any claim that the physics itself is new. Frame as *"known in aeroacoustics, never measured as a training-objective weight."* |
| **Colonius & Lele, Prog. Aerosp. Sci. 2004** | Explicit: "radiated noise is far smaller than near-field pressure fluctuations"; "acoustic inefficiency... is a delicate cancellation... numerical errors that may upset this balance can lead to serious overestimates." `lighthill.py:26-31` says the same thing. Cite it. |
| **ML-for-aeroacoustics** (jet noise, FW-H surrogates, propellers) | Exists, but all such work learns *from* acoustics or predicts SPL directly. **None** discusses loss weighting toward the radiating subspace. This is the clean lane. |
| **Sobolev training** (Czarnecki et al., NeurIPS 2017) | Confirmed. Weights by `k^n`. The claim that it structurally cannot help is correct and unexploited. |

### Two over-claims in the build plan

1. `build-plan.md:35-46` presents "only a sliver radiates" as "the crown jewel and the
   part no ML paper contains." The *first two* supporting facts (acoustic field at low k;
   energetically negligible) are Lighthill 1952/1954 and extremely well known. The
   novelty is only the **third** fact *as an ML loss-weight measurement*.
2. `build-plan.md:36` invokes "Lighthill's eighth-power law" as a scaling to match.
   But the M⁸ law requires a **compact** source, and the compactness parameter is
   `ωL/c₀ ~ M` — so **the low-Mach limit is precisely the regime where the compact-source
   picture breaks down.** The measured local exponents (10.7 / 7.1 / 5.7, steeper than
   M⁸) are therefore *expected*, not anomalous. `HANDOFF.md:28` already records that
   erfc(1/M) does not fit; M⁸ should be retired the same way, with compactness as the
   explanation.

### Useful negative result

The Gaussian random-sweeping check failing (`HANDOFF.md:331-338`) is consistent with the
log's own stated explanation: the box-scale modes carry the cone energy and they *do* the
sweeping rather than being swept. That is a correct diagnosis, not a loose end.

---

## 5. Real, useful cases

The abstract claim generalizes well beyond acoustics. The pattern is: *an L2-trained
surrogate is blind to a quantity of interest that is (a) decision-relevant and
(b) energetically negligible in the field it predicts.* Aeroacoustics is the cleanest
instance because the loss is quadratic and the answer is tiny. Other instances of the
same structure:

- **Wall-pressure / trailing-edge noise.** Dipole sources on a body; same conical
  criterion with the surface term. Directly relevant: literature search surfaced a study
  showing quadrupole sound is *present* at M∞ = 0.1 and *absent* at M∞ = 0.05 (JFM 2023)
  — a Mach threshold at which a source class switches on. That is a real design decision.
- **Combustion instability / thermoacoustic modes.** The eigenmode is a small-amplitude
  linear feature; L2 on the state variable barely constrains it. The strongest analogue,
  because the failure mode (a growing oscillation) is a safety issue.
- **Dilute species / trace constituents in turbulent combustion.** Same structure, and
  there is an existing ML literature (CFD surrogates for species) that is a natural test bed.
- **Stability precursors / transition prediction.** Small-amplitude linear growth.
- **Near-wall tail quantities** — the opposite (high-k) failure mode. Useful as the
  *control* case showing the mechanism is not simply "the loss is bad."

**Industrial relevance.** The compute economics are brutal, and that is the pitch.
Direct CAA at low Mach requires resolving acoustic wavelengths ~1/M smaller than
hydrodynamic scales (Wang, Freund & Lele 2006 is the canonical review). Hybrid
LES + analogy is the standard workaround — and it *needs a velocity-field surrogate* to be
cheap. So the people who need a surrogate most are exactly the people whose QoI this
metric diagnoses as unlearnable. That is a real, fundable niche.

**Software deliverable.** The diagnostic itself. `R(M)` and the blindness ratio `B`,
computed on any (CFD, surrogate) pair in a few minutes with no retraining, is something
an aeroacoustics engineer would actually run. The method — Lighthill source from a
velocity field, a (k,ω) cone mask, a loss-share ratio — is simple enough to port into a
commercial CAA code.

---

## 6. Novel directions

Ranked by (value × feasibility).

### A. The blindness ratio `B` — the strongest single addition
As derived in §3. Replaces the fragile "R is a loss weight" framing with a *measured*
quantity from Stage 3. Predicted to be O(1e6) rather than O(1e-8), which is a far more
arresting result — and unlike R it cannot be dismissed as an artifact, because it is
measured rather than assumed. **Do this first.** ~50 lines.

### B. Fix the Sobolev argument and make it a λ sweep
Report `R_loss` for `1 + λ(1+k²)` across λ ∈ [1e-3, 1e3] for both architectures. The
corrected finding is more interesting than the broken one: *the cone share is
essentially λ-independent in the regime where Sobolev training actually operates, because
the additive penalty's mass is dominated by the high-k bulk it never touches.* That is a
much tighter version of "the standard remedy cannot work here." Also directly addresses
`build-plan.md:287`, Stage 4.

### C. Validate against Rucz et al. 2024 instead of Müller–Obermeier 1967
Replaces open item 4 in `HANDOFF.md:59`. Open access, modern, with error bounds as a
function of exactly the parameters already varied (M, core radius, source truncation).
Yields an **absolute-amplitude** gate check — the thing `vortex_pair.py:22-26` currently
declines and the audit log flags as the plan's original pass/fail criterion. High value,
low cost.

### D. Mach as a continuous knob on a trained model
Nobody does this. Train a single model on one dataset; the sonic line is a *post-hoc*
parameter (`komega.py:24-26` says so explicitly). Sweep M and show how the surrogate's
acoustic content *emerges* as the threshold is lowered — a phase transition in what the
model knows. The most novel-feeling figure available, and it costs one trained model.

### E. Generalize the metric to any energy-elusive QoI
Once `B` exists as a framework, apply it to a second, non-acoustic QoI (dilute species,
wall pressure, an eigenmode amplitude). One extra experiment converts a niche acoustics
paper into a general SciML contribution. **Highest-leverage move for scope.**

### F. Drop the periodic-box far field, or own it explicitly
`greens.py` uses a **free-space** Hankel kernel with a **periodic** source. This is
inconsistent: a strictly periodic source has no far field (Rayleigh's theorem — it
radiates zero net power). The `R` metric is unaffected (purely spectral), but Stage 3's
far-field error inherits the inconsistency. Options: (i) use a sub-window as the source
and state it; (ii) implement the periodic (Rayleigh/helicoidal) Green's function;
(iii) report only near-field pressure and drop far-field claims. Decide explicitly before
Stage 3 rather than after.

### G. Fix the flow quality
The E(k) slope is −3.98, not −3 (`HANDOFF.md:23`). The `drag = 0.08` linear drag is
roughly 5× the inverse-cascade turnover and steepens the spectrum. `B3` correctly notes
the local slope becomes *shallower* toward high k, which contradicts the "drag steepens
it" explanation. Cleaner dissipation (`drag = 0.01`, longer spin-up) would give a
textbook −3 and remove a referee objection. Cost: one ~10-minute run.

---

## 7. Recommended sequence

### Tier 0 — correctness (before anything else; Stage 3 is unrun so all of this is free)
1. Fix `03_compute_sources.py:78` frequency band → select by the radiation condition
   `ω ≥ c₀k`, not by array index. *The one that would have invalidated the paper.*
2. Reconcile the Sobolev-weight convention; re-derive the Stage-4 number from
   `losses.py`'s actual additive form. Add a λ sweep.
3. Add a `k > 0` guard in `radiating_mask`; document the corner-mode truncation.

### Tier 1 — the argument (before writing)
4. Implement the blindness ratio `B = R_loss / R_signal`; reframe the headline around it,
   replacing the "R is the loss weight" claim.
5. Retire the M⁸ comparison; replace with the compactness argument (`ωL/c₀ ~ M`, so
   low-Mach is *outside* the M⁸ regime and steep exponents are expected).
6. Swap Müller–Obermeier for Rucz et al. 2024; add an absolute-amplitude gate check.

### Tier 2 — scope (once the argument holds)
7. Run Stage 3 end-to-end (FNO + UNet, MSE) → error triple + `B`. This is the paper.
8. Stage 4 Sobolev λ-sweep on both architectures.
9. One non-acoustic instance of the same framework (§6E).

### Tier 3 — hygiene
10. Regenerate README (open item 1) — but the table should now carry `B`, not `R`.
11. `pyproject.toml`, `.gitignore`, `git init` (open item 6); record `spinup` in the
    manifest (open item 2).
12. Optional: cleaner-dissipation run to recover a −3 slope (§6G).

**Explicitly do not do:** implement `cone_weighted_loss`. The open question at
`README.md:77-82` is a genuine asset *if* the diagnosis is airtight. Fixing the cone
costs the paper.

---

## 8. Bottom line

The project is real, the codebase is better than most, and the central claim is **not**
in the literature — the amplitude-vs-frequency inversion is a genuine gap, and
`arXiv:2604.20061` is the best available framing against.

But the project is currently **measuring the wrong thing for its own argument.** R is a
property of the signal, not of the loss, and the gap between the two is about six orders
of magnitude in the direction that makes the claim *stronger*. Everything else — the
domain dependence, the M⁸ mismatch, the Sobolev factor — is a symptom of that same
imprecision. Fix the measurement, and the caveats at `HANDOFF.md:18-20` become details
rather than load-bearing walls.

Two of the three bugs above sit in Stage 3 code that has never run. That is the best
possible time to find them.

---

## Appendix — how these findings were produced

- Read all markdown in the project plus the parent build plan.
- Read every `.py` file under `src/`, `scripts/`, `tests/`, and both `runs/*/analysis.py`.
- Executed `python -m pytest -q` → 21 passed.
- Re-derived: Welch segment counts; the `k = 0` mask degeneracy; the far-field
  asymptotic against exact Hankel on real source data; the cone area fraction of the
  (k,ω) plane; the additive-vs-multiplicative Sobolev cone share.
- Simulated the `scripts/03` frequency selection against the radiation condition.
- Websearch and webfetch for ML spectral bias (2024–2026), ML-for-aeroacoustics, Sobolev
  training, Lighthill M⁸ and compactness, Crighton/Lilley, random-sweeping space-time
  spectra, the Müller–Obermeier vortex-pair solution, and 2D-vs-3D radiated power scaling.
