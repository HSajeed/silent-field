# The Silent Field: Why L2-Trained Surrogates Cannot Hear

**One-line claim.** Neural PDE surrogates lose the acoustic field not because it is
high-wavenumber — it is *low*-wavenumber — but because it is energetically negligible.
The standard spectral-bias explanation predicts acoustics should be easy to learn. It is
destroyed anyway. The L2 training objective assigns essentially zero weight to the only
region of wavenumber–frequency space that radiates sound.

**The figure the whole repo exists to produce.** A wavenumber–frequency (k–ω) spectrum of
a turbulent flow, with the sonic line ω = c₀|k| overlaid, and a single number: the
fraction of the total L2 loss that lives inside the radiating cone. For a Mach 0.1 flow
that number is of order 10⁻⁶.

---

## Part 0 — The Physics You Are Standing On

Get this right before writing code; the whole project is one argument and it fails if the
argument is loose.

### The three facts that stack

**1. The acoustic field is at *low* wavenumber.**
At Mach number M, acoustic wavelength / hydrodynamic length ≈ c₀/U = 1/M. At M = 0.1 the
acoustic content sits an order of magnitude *lower* in k than the turbulence producing it.
Every spectral-bias argument in the ML literature says low-k is learned first and learned
well. This is the inversion your paper rests on.

**2. The acoustic field is energetically negligible.**
Lighthill's eighth-power law: radiated acoustic power scales as M⁸ for quadrupole sources.
The acoustic pressure fluctuation relative to hydrodynamic pressure fluctuation scales
roughly as M^(5/2) in the far field. At low Mach the ratio is 10⁻⁴ to 10⁻⁸. An L2 loss
weights by energy, so it is blind by construction.

**3. Only a sliver of the source actually radiates — and it is the sliver with no energy.**
This is the crown jewel and the part no ML paper contains. In the (k, ω) plane:

- Hydrodynamic turbulence lives on the **convective ridge**, ω ≈ U·k
- Acoustic radiation requires **supersonic phase speed**, i.e. |k| ≤ ω/c₀ — the
  **radiating cone** around k = 0
- At low Mach these regions are widely separated. Most of the source is
  hydrodynamically trapped ("silent"); only the radiating-cone content makes sound

So the acoustically decisive content occupies a region of (k, ω) space containing
essentially none of the energy. **The loss weight in the radiating cone is your headline
number.**

### The convenient identity

For an incompressible field, the Lighthill source reduces to the pressure Poisson RHS:

    ∂²(ρ u_i u_j)/∂x_i∂x_j  =  ρ (∂u_i/∂x_j)(∂u_j/∂x_i)   [incompressible]
    ∇²p = −ρ (∂u_i/∂x_j)(∂u_j/∂x_i)

This means **you never need a compressible solver.** You compute the acoustic source as a
post-hoc functional of an incompressible field, exactly as the hybrid LES + acoustic
analogy methods in CAA do. This is what makes the project cheap.

It also means the source is a *near-cancellation of large gradient terms* — so even
before the radiating-cone argument, it is an ill-conditioned functional of the field.

---

## Part I — Reading List

Read in this order. Aim for two weeks of reading running *in parallel* with Stage 1 code.

### A. The aeroacoustic foundation (your home advantage — the ML side has never read any of this)

| Paper | Why |
| --- | --- |
| Lighthill (1952), *On sound generated aerodynamically I: General theory*, Proc. R. Soc. A 211 | The source term and the M⁸ law. Read properly, not from memory |
| Lighthill (1954), *…II: Turbulence as a source of sound* | The turbulence-specific scaling |
| Crighton (1975), *Basic principles of aerodynamic noise generation*, Prog. Aerospace Sci. | The best pedagogical treatment of why the source is a small residual |
| Wang, Freund & Lele (2006), *Computational prediction of flow-generated sound*, Annu. Rev. Fluid Mech. 38 | **The single most important one for you.** It is an entire review about the scale-disparity problem you are reframing for ML |
| Colonius & Lele (2004), *Computational aeroacoustics: progress on nonlinear problems of sound generation*, Prog. Aerospace Sci. | Direct computation vs hybrid; why direct is brutal at low Mach |
| Curle (1955) / Ffowcs Williams & Hawkings (1969) | Needed only if you extend to solid boundaries. Skip for v1 |
| Müller & Obermeier (1967), co-rotating vortex pair | **Your validation case** — has a closed-form acoustic solution |

Extract from these: the M⁸ scaling, the radiating-wavenumber condition, and the standard
CAA framing of "acoustically silent" source content. These are your weapons.

### B. The ML literature you must position against (read defensively)

| Paper | Why |
| --- | --- |
| Rahaman et al. (2019), *On the spectral bias of neural networks* | The canonical spectral-bias reference. Your case is the counterexample |
| Xu et al., *Frequency Principle / F-Principle* | Same idea, independent line. Cite both |
| **arXiv 2604.20061, *Predictivity and Utility of Neural Surrogates of Multiscale PDEs* (2026)** | **Closest existing work.** Makes the general "small MSE, large QoI error" argument via NTK/frequency weighting. You must read this carefully and state precisely what it does not cover — namely amplitude-based rather than frequency-based invisibility |
| Czarnecki et al. (2017), *Sobolev training for neural networks* | The standard proposed remedy. Note that Sobolev training weights by k^n and therefore *still misses* low-k acoustics — this is a strong point for you |
| arXiv 2602.19265, *Spectral bias in physics-informed and operator learning* | Recent mitigation-guidelines survey; shows what the field currently believes |
| arXiv 2606.03936, *Correcting Neural Operator Spectral Bias via Diffusion Posterior Sampling* | Current state of the art in "fixing" spectral bias. Note it fixes the high-k problem, not yours |

### C. Mandatory survey before you commit (do this in week 1)

Search for existing ML-for-aeroacoustics work: DNN jet-noise prediction, learned FW-H
surrogates, CNN trailing-edge noise. **This field exists.** Your claim is not "ML can't do
aeroacoustics" — it is "the standard training objective is structurally blind to the
radiating subspace, and here is the measurement." Know what has been done so you can draw
that line explicitly in the README. If someone has already published the k–ω loss-weight
argument, you need to know in week 1, not month three.

### D. Models you will actually run

- Li et al. (2021), *Fourier Neural Operator* — the surrogate you will train
- Optionally Kochkov et al. (2021), *ML-accelerated CFD*, and Stachenfeld et al. (2022),
  *Learned coarse models for efficient turbulence simulation* — for context on what
  "good" field accuracy currently means

---

## Part II — Stack

Deliberately boring. Nothing here should be a research problem.

```
python >= 3.11

# solver / numerics
numpy, scipy
jax, jax-cfd              # OR: dedalus3 (spectral PDE framework)
h5py or zarr              # time-resolved data is large

# models
torch >= 2.2
neuraloperator            # official FNO package — do NOT rewrite FNO
einops

# analysis
numpy.fft / scipy.fft     # everything spectral
scipy.signal              # Hann windows for k–ω estimation
scipy.special             # Hankel functions for the 2D Green's function

# figures
matplotlib
scienceplots              # publication styling, one line

# hygiene
pytest, hydra-core or simple YAML configs, tqdm
```

### What to borrow rather than build

| Component | Source | Note |
| --- | --- | --- |
| FNO / operator baselines | `neuraloperator` (pip) | Official NeuralOperator library |
| 2D spectral NS solver | `jax-cfd` | Well-tested, differentiable, fast on CPU at 256² |
| Alternative solver | `Dedalus v3` | If you want more general spectral machinery |
| UNet baseline | any standard 2D UNet | 100 lines; or lift the spec from the ISTA paper's Appendix E |
| Reference PDE datasets | PDEBench, The Well | Only as a sanity cross-check — **generate your own primary data** |
| Plot styling | `scienceplots` | Don't spend time on this |

### What you must write yourself

The acoustic analysis. That is the contribution; borrowing it would be borrowing the
paper.

---

## Part III — Data

**Generate your own.** This is the one place where borrowing hurts you: public datasets
are saved at coarse temporal stride, and the k–ω analysis lives or dies on temporal
sampling. You need Δt fine enough to resolve acoustic frequencies, and a long enough
record to resolve them.

### Sampling requirements (get this right or the whole analysis is leakage)

- **Temporal Nyquist.** The radiating cone extends to ω = c₀|k|. You need
  f_sample > 2·c₀·k_max/(2π). At M = 0.1 that is ~10× the hydrodynamic sampling rate you
  would otherwise choose. This is the single most common way to ruin this analysis.
- **Record length.** Frequency resolution Δf = 1/T. You need Δf small enough that the
  radiating cone spans several bins at the k values of interest. Budget a few thousand
  snapshots.
- **Spatial periodicity.** Use doubly-periodic domains so spectral differentiation is
  exact. This kills the discretization-error objection to the Lighthill source outright.

### Three datasets, in order

**D1 — Co-rotating vortex pair (validation).**
Two point/Gaussian vortices orbiting. Müller & Obermeier give a closed-form far-field
acoustic solution. 2D, tiny, runs in seconds. **This is your validation gate:** your
acoustic pipeline must reproduce the analytical solution before it is pointed at any
surrogate.

**D2 — 2D decaying or forced isotropic turbulence (primary).**
256² or 512², doubly periodic, pseudospectral with 2/3 dealiasing. Forced case gives you a
statistically stationary record, which is much better for k–ω estimation than a decaying
one. This is where the headline figure comes from.

**D3 — 2D cylinder wake (recognizability).**
The Aeolian tone is *the* canonical aeroacoustic configuration. Anyone in fluids
recognizes it instantly, and you already have flowTorch's cylinder2D pipeline. Use it as
the third panel — "and here it is on the case you know."

---

## Part IV — Repo Structure

```
silent-field/
├── README.md                    # THE deliverable. Figure first, three paragraphs.
├── src/
│   ├── solvers/
│   │   ├── vortex_pair.py       # D1 + analytical acoustic solution
│   │   ├── turb2d.py            # D2 forced/decaying 2D NS, pseudospectral
│   │   └── io.py                # zarr/h5 writers, run manifests
│   ├── acoustics/
│   │   ├── lighthill.py         # source term, spectral differentiation
│   │   ├── greens.py            # 2D free-space Green's fn (Hankel), far-field p'
│   │   ├── komega.py            # k–ω spectra, windowing, sonic-line masks
│   │   └── radiating.py         # THE metric: loss weight inside the cone
│   ├── models/
│   │   ├── fno.py               # thin wrapper on neuraloperator
│   │   ├── unet.py              # baseline
│   │   └── train.py             # vanilla one-step MSE. Deliberately standard.
│   └── diagnostics/
│       ├── spectra.py           # radial E(k)
│       ├── divergence.py        # ∇·u violation of the prediction
│       └── errors.py            # field error vs source error vs far-field error
├── scripts/
│   ├── 00_validate_acoustics.py # gate: reproduce Müller–Obermeier
│   ├── 01_generate_data.py
│   ├── 02_train_surrogate.py
│   ├── 03_compute_sources.py    # Lighthill source from truth and from prediction
│   ├── 04_komega_analysis.py    # the headline figure
│   └── 05_make_figures.py
├── figures/
├── notebooks/
│   └── walkthrough.ipynb        # for the CV reader who won't clone the repo
└── tests/
    ├── test_lighthill.py        # spectral derivative vs analytic on a known field
    ├── test_greens.py           # monopole/quadrupole with known solution
    └── test_komega.py           # planted wave at known (k,ω) lands in the right bin
```

The `tests/` directory is not optional. It is what separates "referee-grade treatment of a
small question" from "weekend project," and it is the first thing a careful reader opens.

---

## Part V — Stages

Target: **six weeks**, with a shippable first figure at the end of week two.

### Stage 1 — Acoustic pipeline validation (week 1)
*Gate: does your acoustics code produce correct sound for a case with a known answer?*

1. Implement the co-rotating vortex pair (D1)
2. Implement `lighthill.py` with spectral differentiation
3. Implement `greens.py` — **note: the 2D free-space Green's function is a Hankel
   function, not 1/r.** Work in the frequency domain
4. Reproduce the Müller–Obermeier analytical far-field solution
5. Convergence check: compute the source at two resolutions, show it converges

**Kill criterion:** if you cannot match the analytical solution, nothing downstream means
anything. Do not proceed on a broken pipeline.

### Stage 2 — The headline figure (week 2)
*This is the shippable milestone. If you stopped here you would still have something.*

6. Generate D2 with correct temporal sampling
7. `komega.py`: 3D FFT (x, y, t) with a Hann window in time; radially bin in k
8. Plot the k–ω spectrum, overlay the sonic line ω = c₀|k| and the convective ridge ω = Uk
9. Compute **the number**: fraction of total L2 loss weight inside the radiating cone
10. Sensitivity: repeat at M = 0.05, 0.1, 0.2. Does the number scale as the theory says?

Push the repo at the end of this week with the figure and a three-paragraph README. Everything
after this is strengthening.

### Stage 3 — The surrogate demonstration (weeks 3–4)
11. Train an FNO and a UNet on D2 with a standard one-step MSE loss. **Keep this
    deliberately vanilla** — the point is that the *standard* recipe fails
12. Report field error (expect ~1%)
13. Compute the Lighthill source from truth and from prediction; report source error
    (expect ≫ 100%)
14. Compute far-field acoustic pressure from both; report spectral error
    > **Corrected 2026-09-30 — select the band by the radiation condition, not by
    > array index.** `omega[:24]` takes the 24 *lowest* frequencies, which at M = 0.1
    > is ω ∈ [0.15, 3.68] while radiation needs ω ≥ c₀·dk = 11.3. **0 of those 24
    > frequencies are supersonic**, and 91% of the source's ω > 0 power is below
    > c₀·dk, on the convective ridge. The far-field error would be measured entirely on
    > hydrodynamically trapped frequencies, and would scale with record length rather
    > than describing a physically fixed band. Use `greens.select_band`, which returns
    > the band plus the diagnostics that make it auditable. A side benefit: the
    > physically anchored band puts the Hankel argument at kr ≈ 10³, where the
    > far-field asymptotic is unambiguously valid.
15. Compute the k–ω spectrum *of the prediction error* and show it is not small in the
    radiating cone — and that the cone is where the model was never penalized
16. `divergence.py`: the prediction's ∇·u is not zero. Quantify. It is a second,
    independent channel of spurious dilatation

### Stage 4 — The counterexample that closes the loop (week 5) — **REVISED 2026-09-30**
*Originally "train a Sobolev variant and show it still fails". The training run is no
longer the argument; it is optional confirmation. See the correction below.*

17. **Analysis, no training.** `python scripts/06_sobolev_argument.py` — sweep the
    cone share of the Sobolev objective against λ. Done; result below.
18. **Optional confirmation run.** Train the H¹ variant, show it improves high-k
    fidelity and leaves the cone share where MSE left it. This *confirms* the analysis
    on a real optimiser trajectory; it is no longer load-bearing.
19. State the design principle plainly: the correct weighting is not by wavenumber
    magnitude but by *distance to the sonic line in (k, ω)*.

**Why the revision.** The original step 17 was going to be answered by a number that
was measuring the wrong objective. `losses.sobolev_loss` minimises a **sum** of two
separately normalised terms, so the per-mode weight it applies is

    w_eff(k) = 1 + λ·(N_cells / N_half)·(1 + k²)^order / ⟨(1 + k²)^order⟩

Analysing it as a pure `(1 + k²)^order` reweighting — which is what the codebase did
until 2026-09-30 — describes a different objective and makes Sobolev look ~19× worse
than L2. That is where the old "ten times worse" claim came from.

Measured on D2 at M = 0.1, the trained objective's cone share relative to L2:

| λ | 1e-3 | 1e-2 (default) | 1e-1 | 1 | 10 | 1e2 | 1e3 | 1e6 |
|---|---|---|---|---|---|---|---|---|
| share ÷ L2 | 1.000 | **1.000** | 0.997 | 0.974 | 0.794 | 0.303 | 0.086 | 0.053 |

The second term of `w_eff` is 7.3e-4 … 1.7e-2 of its own mean everywhere inside the
cone — the cone is at low k (it reaches only k ≤ 6.7, 1.2% of the resolved |k|
values) and the weight's mass is at high k, where the cone has no area at all. So
`|w_eff − 1| ≤ 3.3e-4` across the entire cone and the objective is *identical* to L2
there, at 100× the default λ. Same at M = 0.05, 0.2, 0.3.

**The corrected argument is stronger than the original one.** "The standard remedy is
inert here, and here is the geometric reason" is a tighter, cheaper, and more
falsifiable claim than "we trained it and it still failed" — and it is a statement
about the objective rather than about one optimiser's trajectory. The 0.053 asymptote
is the multiplicative reading, retained as the λ → ∞ limit of the same sweep.

This is still where the project stops being a demonstration and becomes an argument.

### Stage 5 — Recognizability and write-up (week 6)
20. Run the whole battery on D3 (cylinder wake / Aeolian tone)
21. README, walkthrough notebook, figures
22. Leave the open question visible: *what loss function fixes this?* — a
    radiating-cone-weighted objective is the obvious candidate and the natural
    continuation. State it as future work, not as a gap

---

## Part VI — Post-Processing Methods

The part most likely to be done wrong. Specifics:

**Spectral differentiation.** All derivatives via FFT on the periodic domain. Exact to
machine precision, which removes the "your source error is really your discretization
error" objection before anyone raises it. Show the convergence check anyway.

**Dealiasing.** The Lighthill source is quadratic in u. Apply the 2/3 rule when forming
u_iu_j, or aliasing errors will contaminate exactly the high-k content you are measuring.

**k–ω estimation.** 3D FFT over (x, y, t). Hann window in time (rectangular windowing
produces spectral leakage that will smear energy into the radiating cone and manufacture
your own result). Consider Welch averaging over overlapping time segments for a cleaner
estimate. Radially bin in |k| for isotropic flows.

**The sonic line.** Overlay ω = c₀|k|. Everything with |k| < ω/c₀ radiates; everything
outside is hydrodynamically trapped. Define the radiating cone mask once, in
`radiating.py`, and use it everywhere.

**The headline metric.** Define precisely and defend it:

    R = Σ_{(k,ω) ∈ cone} |û(k,ω)|²  /  Σ_{all (k,ω)} |û(k,ω)|²

This is literally the fraction of the L2 objective that the radiating subspace receives.
Report it per Mach number with error bars over realizations.

> **Corrected 2026-09-30 — the mask must not be applied to the radially binned
> spectrum.** On a square periodic grid one radial bin holds several distinct |k|
> (√21, 5 and √29 all land in the bin centred on 5), so evaluating `ω ≥ c₀·k` at bin
> centres tests every mode in the bin against the wrong k, and near the cone edge that
> admits a slab of non-radiating modes. Measured on D2 at M = 0.1: it inflated R by
> **2.84×** (velocity) and **4.79×** (source). Binning correctly — centring bins on
> multiples of dk — is necessary but not sufficient. `KOmegaSpectrum` therefore keeps
> a radially *resolved* companion grid (O(n) distinct |k| values, not O(n²), so it is
> cheap) and the metric runs on that. See `TIER0_PLAN.md` item 3.

> **Also corrected 2026-09-30 — "the fraction of the L2 objective" is an assumption,
> not a measurement.** R describes the *signal*. The loss is computed on the *error*,
> and for a spectrally white error field (the typical undertrained model) the cone
> takes ~6% of the total loss — about six orders of magnitude more than R. So R is the
> *optimistic* bound. Stage 3 step 15 should report both, and the ratio
> `B = R_loss / R_signal ≥ 1` is the falsifiable quantity (see `branch_analysis.md` §3).

**Far-field pressure.** In 2D, convolve the Lighthill source with the free-space Green's
function in the frequency domain — Hankel function H₀⁽¹⁾, not the 3D 1/r kernel. Getting
this wrong is a classic 2D-CAA error and a referee will catch it.

**Error reporting.** Always report the triple together: field L2 error, source error,
far-field spectral error. The rhetorical force is entirely in the contrast, so never show
one without the others.

---

## Part VII — Compute

Negligible. This is the point.

| Task | Cost |
| --- | --- |
| D1 vortex pair | seconds, CPU |
| D2 forced 2D turbulence, 256², few thousand snapshots | 1–3 h CPU, minutes on any GPU |
| FNO training (2D, 256², modest dataset) | ~1 h on a T4; a few hours CPU-only |
| 3D FFT for k–ω, 256² × 4096 timesteps | memory-bound, not compute-bound — ~2 GB in fp32. Chunk over time segments with Welch averaging if tight |
| Lighthill source + far-field | seconds |

**Verdict: runs entirely on your laptop.** The only resource question is disk and RAM for
the time-resolved record — budget 10–50 GB of storage and use zarr with chunked time
access rather than loading the whole array. If RAM is tight, Welch segmentation solves it
and improves the spectral estimate anyway.

Free Colab is more than sufficient for the training step if your laptop lacks a GPU.

---

## Part VIII — Framing Risks

**The one that kills you.** If a reader files this as "another spectral bias
demonstration," you lose everything. Defense: lead with the low-k inversion in the title
and the first sentence of the README. *Acoustics is low-wavenumber. Spectral bias predicts
it should be easy. It isn't.* Make Stage 4's Sobolev counterexample prominent — it is the
proof that frequency-based explanations do not cover this case.

**The second.** Someone may have published the k–ω loss-weight argument already. Do the
week-1 survey. If they have, the project pivots to the Sobolev counterexample and the
radiating-cone-weighted loss, which is still yours.

**The third.** "Nobody trains surrogates for acoustics anyway." Answer it directly in the
README: aeroacoustics is the cleanest instance of a general failure — a QoI that is
large-scale and energetically negligible — and the same structure appears in trace species
in combustion, small-perturbation stability, and boundary-layer transition. One paragraph,
near the end. Do not over-claim it; the acoustic case is the one you actually demonstrate.
