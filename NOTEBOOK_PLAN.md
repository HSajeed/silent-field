# Plan — Colab notebook series for silent-field

**Date:** 2026-10-01. **Status:** plan only; nothing built yet. **Decisions taken with the author:**

| Question | Decision |
|---|---|
| Getting code and data into Colab | Private GitHub repo for code + Google Drive for datasets |
| Layout | A series of chapter notebooks plus a small shared helper package |
| Showing history | Run the **old** code (initial commit `2535bb4`) and the **current** code side by side on the same data |
| Future steps | Run whatever can be checked now; mark everything else as clearly labelled placeholder cells |

**Goal.** A reader opens the notebooks in Colab, runs them top to bottom, and understands:

- what the project claims and the physics it rests on;
- what was done first, and what it produced;
- every change since then: what was wrong, how it was found, what the fix changed;
- what comes next, and why.

---

## 1. Project timeline the notebooks will narrate

Sources: `rerun_log_2026-09-15.md`, `TIER0_PLAN.md`, `branch_analysis.md`,
`review_2026-10-01.md`, git history.

| Date | Event | In git? |
|---|---|---|
| ~09-14 | Build plan written; Stages 1–2 code written; README numbers from a 64² smoke dataset | no |
| 09-15 | **Production rerun**: gate PASS; `turb2d.npz` generated; headline R = 8.7e-08 at M = 0.1; H¹ "10–20× below L2" | no |
| 09-15 | **Independent audit**: Green's function was *incoming* (H₀⁽¹⁾); wrong 3D kernel still passes the m = 2 check; cone energy sits at \|k\| = 1; Mach scaling is not a power law; R_S measured | no |
| 09-15 | **Follow-up B1–B8**: kernel → (−i/4)H₀⁽²⁾; gate CHECK 3 and CHECK 4 added; `losses._k_grid` used mode index instead of angular k (fixed); Sobolev weight normalised (penalty/MSE 2728 → 0.9955); bands added; noise floor; kf16 and L4π datasets; "R(all k) is domain-scoped" | no |
| 09-29 | **Initial commit `2535bb4`** (post-follow-up, pre-Tier-0, 21 tests); `branch_analysis.md` written | **yes** |
| 09-30 | **Tier 0**: cone mask on true \|k\| (8.7e-08 → 3.2e-08); Sobolev additive convention (1.000× L2); far-field band by radiation condition (0/24 → 24/24); 42 tests | working tree, uncommitted |
| 10-01 | **Review**: S-form far field is 7–168× artifact; Stage-4 reframing; factor-2 bug; M⁷ not M⁸; audit of the branch_analysis plan | working tree, uncommitted |

**Consequence for "run old vs new".** Only the 09-29 → present changes (Tier 0 and the
review) can be run from two real code versions. Changes from 09-15 never reached git. Those
are shown by small, exact reconstructions inside the notebook. Examples: evaluating H₀⁽¹⁾ vs
H₀⁽²⁾ directly with scipy, and `_k_grid` with mode index vs angular k. Each such cell is
labelled "reconstruction" so it is never mistaken for the historical code.

---

## 2. Repository layout to add

```
notebooks/
  00_overview_and_setup.ipynb
  01_validation_gate.ipynb
  02_turbulence_dataset.ipynb
  03_komega_headline.ipynb
  04_tier0_fixes_old_vs_new.ipynb
  05_review_2026-10-01_findings.ipynb
  06_stage3_surrogate.ipynb          (future; partly runnable)
  07_stage4_and_roadmap.ipynb        (future; partly runnable)
  nbtools/
    __init__.py
    env.py        Colab/local detection, repo clone, Drive mount, data paths, manifest-hash check
    versions.py   load the initial commit's src/ as package `src_old` beside the current `src`
    cache.py      save/load heavy intermediates (spectra) to Drive so chapters don't recompute
    style.py      shared matplotlib style, consistent colours for old/new, table helpers
  _src/           percent-format .py sources each .ipynb is built from (diffable, reviewable)
  build_notebooks.py   converts _src/*.py → *.ipynb with plain json (no nbformat needed)
```

### How "old vs new" works (`nbtools/versions.py`)
1. `git worktree add <tmp>/silent-field-old 2535bb4`, or `git archive` if no worktree is
   wanted. In Colab the clone has full history, so this works offline.
2. Load the old `src/` with `importlib.util.spec_from_file_location("src_old", …,
   submodule_search_locations=[…])`. Its relative imports (`from ..spectral import Grid`)
   then resolve inside `src_old`, and both versions coexist in one kernel:
   `from src_old.acoustics.komega import komega_spectrum as komega_old`.
3. Every comparison cell prints **old value → new value → ratio → which log recorded it**.
   It asserts the old value matches the logged number (e.g. 8.691e-08 from
   `komega_metrics.json`), so the notebook proves it is reproducing history, not inventing it.

### Colab specifics (`nbtools/env.py`)
- **Code.** Clone the private repo with a GitHub token stored in Colab Secrets
  (`google.colab.userdata.get("GH_TOKEN")`). Never paste the token into a cell.
- **Data.** Mount Drive and point `DATA_DIR` at `MyDrive/silent-field/data/`. Verify each
  `.npz` against its manifest hash (111e13e318c9 / e5d144b83287 / 34d5ea324579) before use.
- **Dependencies.** numpy, scipy, matplotlib and torch are preinstalled in Colab, so only
  `pytest` is needed. Pin nothing unless a version difference shows up in a number.
- **Local runs.** Every Colab-only cell is guarded by `IN_COLAB`, so the same notebooks also
  run locally against `data/`.
- **No dataset regeneration.** Respects `CLAUDE.md`. Chapter 02 has one *toy* run (n = 64,
  short) written to a temp folder, only to show how the solver works.

---

## 3. Chapter-by-chapter content

Each chapter opens with four lines (what you will learn, inputs, runtime, what is old / new /
future) and ends with a "what this chapter established" box. Physics is explained at the
level of a CFD reader new to aeroacoustics and ML; derivations are shown, not just stated.

### 00 — Overview and setup (~5 min)
- **Content:**
  - The claim, in three sentences, and why it inverts the spectral-bias story.
  - Physics primer:
    - phase speed and the radiating cone ω ≥ c₀|k|;
    - the convective ridge ω ≈ U|k|;
    - the Lighthill analogy and why an incompressible flow suffices at low M;
    - 2D vs 3D Green's functions.
  - Repo map: which module does what.
  - The timeline from §1, as an interactive table.
- **Runs:** setup (clone, mount, hash checks); `pytest -q` → expect 42 passed.

### 01 — The validation gate (Stage 1) (~3 min)
- **Content:** the co-rotating vortex pair, the two forms of the Lighthill source, the 2D
  Hankel Green's function, and what each of the 4 checks guards against.
- **History (reconstruction, 09-15):**
  - The kernel was incoming. Plot dφ/dr for H₀⁽¹⁾ vs H₀⁽²⁾: +k vs −k.
  - The 3D kernel passes the m = 2 check. Show it passing, then show CHECK 3's |p| ratio,
    2.00 for 2D vs 4.00 for 3D, catching it.
- **Gate blind spot noted by the audit:** the 2Ω tone check follows from the pair's symmetry,
  so it can hardly fail.
- **Runs:** `scripts/00_validate_acoustics.py`, with its figure displayed inline.
- **Future, runnable now:** the S-form vs T-form far field on the vortex pair. They must agree,
  because the pair's source decays inside the box. This is the validation `review_2026-10-01.md`
  §1.5 requires before the T-form can be trusted on turbulence. If it passes, it becomes a new
  gate check (separate code change, author approval).

### 02 — The turbulence dataset (~5 min, plus an optional toy run of a few minutes)
- **Content:** the forced 2D Navier–Stokes solver (pseudospectral, 2/3 dealiasing, Kolmogorov
  forcing at k = 8, linear drag); the acoustic sampling bound dt < M·L/(2u_rms) (1.7×
  oversampled); manifest and hash.
- **Diagnostics run on `turb2d.npz`:**
  - stationarity: CV 4.4%, drift 6%, about 6 correlation times;
  - the "spin-up print" correction from the audit;
  - E(k) slope −3.98 on [12, 36], and why it isn't −3: friction steepening (Boffetta & Ecke
    2012 §4.3), and why the lower-drag "fix" risks a condensate (§4.2);
  - integral scale 0.51 vs L = 6.28.
- **Optional:** a toy n = 64 run to watch the solver evolve vorticity.
- **Datasets kf16 and L4π:** introduced here, with what each was built to test.

### 03 — The k–ω headline (Stage 2) (~5–10 min)
- **Content:** building the (k, ω) spectrum with windowing, folding and Parseval normalisation.
  Plot it with the sonic line and the convective ridge.
- **Metrics:** R and R_S; the Mach sweep; the uncertainty band (full record and both halves).
- **Robustness findings from the 09-15 audit:** window sensitivity, the noise floor, the
  per-bin-vs-density pitfall, and the |k| = 1 concentration.
- **Domain tests (B6, B7):**
  - k ≥ 2 rows and the kf16 / L4π comparison, loaded from `runs/*/results.json` and labelled
    **pre-Tier-0**.
  - **Opportunity:** Colab's ~12 GB RAM may fit the n = 256 L4π spectrum that the local 7 GB
    machine could not. One guarded cell attempts it and reports peak memory. If it runs,
    HANDOFF open item 1 gets post-fix L4π numbers without the chunking refactor.
    *Unverified until tried.*
- **Runs:** `scripts/04_komega_analysis.py`; cached spectra are written to Drive for chapters
  04–05.

### 04 — Tier-0 fixes, old vs new (~10 min)
Each fix follows the same five beats: **what the old code did → how it was found → old
number (asserted against the log) → new number → why the new one is right.**

1. **Cone mask on bin centres vs true |k|.**
   - Show that √21, 5 and √29 share one bin.
   - Draw the slab of non-radiating modes the old mask admitted.
   - Velocity R 8.7e-08 → 3.2e-08 (2.84×); source R_S 1.0e-08 → 2.3e-09 (4.79×).
   - Plus the `k > 0` guard and the corner modes.
2. **Sobolev convention.**
   - Multiplicative (1 + k²) vs what `sobolev_loss` trains (additive).
   - The old "10× worse" vs 1.000× L2.
   - Run `scripts/06_sobolev_argument.py` and plot the λ sweep.
3. **Far-field band.**
   - Old `[:24]` lowest frequencies: 0 of 24 radiating, ω ∈ [0.15, 3.68].
   - New `select_band`: 24 of 24, kr 189–1309. The old band gave 11× larger pressure.
4. **09-15 follow-up fixes (reconstructions):**
   - `_k_grid` mode index vs angular k (matters for L ≠ 2π);
   - Sobolev normalisation (penalty/MSE 2728 → 0.9955).
- **Closing table:** every Tier-0 number, old → new, with the log reference.

### 05 — Review findings, 2026-10-01 (~10 min)
Mirrors `review_2026-10-01.md`, running its Appendix A scripts.
- **§1:**
  - box mean of S = 0 vs the fluctuating box-integrated Tᵢⱼ;
  - **S-form vs T-form far field on the truth data (7–168×)**;
  - the compact share;
  - why this explains the domain dependence (Sarkar & Hussaini 1993, quoted).
- **§2:**
  - the Sobolev term is 0.03% of the objective at λ = 1e-2;
  - the robust "never raises the cone share" statement;
  - the factor-2 check against torch;
  - neuraloperator `H1Loss` (multiplicative).
- **§3:** compactness He = 2·Sr·M improves at low M; 2D compact quadrupole power ∝ M⁷; why
  steep exponents alone don't prove an artifact.
- **§4:**
  - corrected `branch_analysis` §3 numbers (white error 3.0e-3, e ∝ k²û 8.8e-12);
  - why "B ≥ 1" is not a theorem;
  - the one-step loss having no ω axis, so jitter becomes a hypothesis to test;
  - the "Rayleigh's theorem" claim vs Bragg orders.
- **Labels:** every claim shows its tag from the review ([computed] / [cited] / [code] /
  [unverified]).

### 06 — Stage 3, surrogate demonstration (future; partly runnable)
- **Runnable now (GPU runtime):**
  - Train FNO and UNet with one-step MSE (`scripts/02_train_surrogate.py`); checkpoints
    go to Drive.
  - Field error and source error (`error_triple`).
  - Spurious dilatation.
- **Runnable, with a banner:** the far-field error through the **T-form**
  (`far_field_pressure_ibp`) next to the current S-form. Shows how much of the S-form error
  is artifact. Labelled "not the published number until review step 1 is accepted".
- **Placeholders (marked `# FUTURE — not implemented`):** B and B_S, with the two controls
  (teacher-forced vs rollout; a random-error baseline at matched amplitude). The plan and the
  expected outputs are written out in the cells.
- **Runtime:** the build plan estimates ~1 h on a T4 *[unverified]*. The first run records
  the actual time.

### 07 — Stage 4 and the roadmap (future; partly runnable)
- **Runnable:** the λ sweep with the standard H¹ form; the factor-2-corrected additive weight
  (inline, analysis only).
- **Placeholders:**
  - trained Sobolev runs (both architectures);
  - the causal test (multi-step / temporal loss: does B drop?);
  - the Rucz et al. 2024 absolute-amplitude gate;
  - Mach as a continuous knob;
  - a second QoI.
- **The revised plan** from `review_2026-10-01.md` §5, as a living status table (step,
  acceptance check, status, link to the chapter that will show it).

---

## 4. Build and verification

1. **Prerequisite commits (need your approval).** Commit the working tree so Colab can clone
   it:
   - Tier-0 code and tests, `scripts/06`, figures;
   - `TIER0_PLAN.md`, `acoustic-blindness-build-plan.md`, `review_2026-10-01.md`, this plan.
   Commit on a branch, not `main`.
2. **Write `nbtools/` first, with tests** (`tests/test_nbtools.py`): the old/new loader
   resolves both packages, and the hash check catches a corrupted file.
3. **Write chapter sources** in `notebooks/_src/*.py`. Build the `.ipynb` files with
   `build_notebooks.py`, which uses plain `json` (no extra dependency).
4. **Verify locally before handing over.**
   - Execute every chapter against the local data and check that each asserted number
     matches its log.
   - Local limits: 7 GB RAM and no GPU. The L4π cell and the training cells are skipped
     locally and are only verifiable in Colab.
   - **To execute the `.ipynb` files themselves**, I need `nbclient` + `ipykernel` in the
     `dmdsae` env. Otherwise I verify by running each chapter's code cells as a script, which
     is equivalent for correctness but doesn't test the notebook JSON.
5. **First Colab run (you):** run 00 → 05 on a CPU runtime and 06 on a T4. Report any cell
   that fails or any number that differs from the local run; I fix and rebuild.
6. **After that,** HANDOFF gets a pointer to the notebooks. When a future step lands, its
   placeholder in 06/07 is replaced and the status table is updated.

**Effort estimate:** `nbtools` plus chapters 00–05 are the bulk. Chapters 06–07 are lighter,
since much of them is placeholders. Chapter 03's L4π cell and chapter 06's training are the
only real unknowns for runtime and memory.

---

## 5. Still need from you

1. **GitHub.** `gh` is not installed here, so you create the private repo (name?) and push,
   or install `gh` and authenticate so I can. You also create a token for Colab Secrets
   (fine-grained, read-only on that one repo).
2. **What to commit.** `agent-operator-playbook-full.md` is untracked. Include it or keep it
   out (default: out)? `HANDOFF.md` and `CLAUDE.md` stay untracked as before, so the notebooks
   will be self-contained and won't depend on them.
3. **Drive.** Is the folder `MyDrive/silent-field/data/` fine? Upload just the baseline
   (118 MB), or all three (~745 MB)? kf16 and L4π are only needed for the domain cells in
   chapter 03.
4. **Local tooling.** May I `pip install nbclient ipykernel` into the `dmdsae` env so the
   notebooks are executed for real before you get them?
