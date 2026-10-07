# silent-field — chapter notebooks

A guided, runnable walk through the whole project: what was done first, what changed and
why, and what comes next. Every number that also appears in a log is recomputed and
checked against that log. Old code (the initial commit `2535bb4`) runs side by side with
the current code.

| # | notebook | covers | runtime |
|---|---|---|---|
| 00 | `00_overview_and_setup` | claim, physics, repo, timeline, tests, data checks | ~3 min |
| 01 | `01_validation_gate` | vortex-pair gate; Green's-function fix; S-form vs T-form on the pair | ~3 min |
| 02 | `02_turbulence_dataset` | solver, sampling, stationarity, −3.98 slope, integral scale | ~5 min |
| 03 | `03_komega_headline` | (k, ω) spectrum, R and R_S, bands, box tests (post-fix L = 4π) | ~10 min |
| 04 | `04_tier0_fixes_old_vs_new` | the Tier-0 bugs, run in both code versions | ~10 min |
| 05 | `05_review_findings` | 2026-10-01 review, recomputed | ~10 min |
| 06 | `06_stage3_surrogate` | Stage 3: training, error triple, preliminary B (future, partly runnable) | GPU ~1 h |
| 07 | `07_stage4_and_roadmap` | Stage 4, causal test, roadmap status | ~3 min |

Runtimes are first-run CPU estimates; cached spectra make re-runs much faster.

## Running in Google Colab

1. **Token.** On GitHub create a fine-grained personal access token with *read-only*
   access to the contents of `HSajeed/silent-field`. In Colab, open *Secrets* (key icon,
   left bar), add it as `GH_TOKEN`, and enable notebook access. The setup cell clones with
   it and then removes it from the clone's git config.
2. **Data.** Upload the datasets to Google Drive at `MyDrive/silent-field/data/`:
   - `turb2d.npz` (118 MB, required by chapters 02–07);
   - `turb2d_kf16.npz` (125 MB, optional, chapter 03 §6);
   - `turb2d_L4pi.npz` (496 MB, optional, chapter 03 §6).

   Each file is checked against the author's sha256 before use. Manifests come from git.
3. **Open** a notebook from GitHub in Colab (*File → Open notebook → GitHub*, include
   private repos, branch `tier0-review-notebooks`) or use the badge at the top of each
   notebook. Run the first cell, then *Run all*.
4. **GPU** is only needed for chapter 06 (and the opt-in cell in 07): *Runtime → Change
   runtime type → T4 GPU*.

Caches (spectra, checkpoints, results) go to `MyDrive/silent-field/cache/`, so chapters
share work and a reconnected session does not recompute. Delete that folder to force a
clean run. To pull newer commits into an existing Colab session, run
`nbenv.update_repo()` and restart the runtime.

## Running locally

From the repo root, with the project's Python environment:

    python notebooks/build_notebooks.py        # regenerate .ipynb from notebooks/_src/*.py
    python notebooks/run_notebooks.py 00 01    # execute headless (needs nbclient + ipykernel)

The `.ipynb` files are generated from the percent-format sources in `_src/`. Edit the
sources, not the notebooks. Local caches go to `.nb_cache/` (git-ignored). On a machine
with a CUDA GPU, chapter 06 trains by default; set `RUN_TRAINING = False` to skip.

## Change note, 2026-10-07

The code changed (factor-2 fix in `additive_sobolev_weight` plus `shell_axis_fraction`;
`komega.py` shells grouped on `rint(k²/dk²)`, 1621 at n = 128 instead of 1850; T-form far
field in `scripts/03`; new `tform.py`). Chapter 05 §2.1–2.2 now show the pre-fix formula
inline (reproduces the review's 2×) next to the current function (1×), and chapters 04, 05
and 07 prose was made version-honest. Cached spectra from the old shell grouping were
deleted from `.nb_cache/`. Re-run headless after the change: chapters 01, 03, 04, 05, 07
(all asserts pass). Chapters 00, 02 and 06 were not re-run.
