# The Silent Field

**Neural PDE surrogates lose the acoustic field — but not for the reason the literature assumes.**

Spectral bias says networks learn low wavenumbers first and lose high ones. The
acoustic field is at **low** wavenumber: at Mach number `M` the acoustic
wavelength exceeds the hydrodynamic length scale by `1/M`. By every spectral-bias
argument it should be the easiest content in the flow to learn.

It is destroyed anyway, because it is invisible in **amplitude**, not in frequency.

## The measurement

A disturbance radiates only if its phase speed is supersonic, `ω/|k| ≥ c₀`. That
carves a cone out of the wavenumber–frequency plane. Turbulence does not live
there — it lives on the convective ridge `ω ≈ U|k|`, whose slope is smaller by
exactly the Mach number.

An L2 loss weights every mode by its energy. So the share of the training
objective assigned to the only region that makes sound is:

| Mach | L2 objective | Lighthill source |
|------|--------------|------------------|
| 0.05 | 1.8 × 10⁻¹¹  | 5.3 × 10⁻¹² |
| 0.10 | 3.2 × 10⁻⁸   | 2.3 × 10⁻⁹  |
| 0.20 | 4.8 × 10⁻⁶   | 3.4 × 10⁻⁷  |
| 0.30 | 6.2 × 10⁻⁵   | 5.5 × 10⁻⁶  |

*(128², L = 2π, `data/turb2d.npz`, Welch 4-segment, full record:
`python scripts/04_komega_analysis.py`. Reproduce in ~3 min on a laptop.)*

**Sobolev training — the standard remedy for spectral bias — does not touch this
number at all.** That is the finding, and it is stronger than "it makes it worse."

A Sobolev objective is a *sum* of two separately normalised terms, so the per-mode
weight it applies is `1 + λ(N_cells/N_half)(1+k²)^order`, not `(1+k²)^order`. The
second term is 7 × 10⁻⁴ … 2 × 10⁻² of its own mean everywhere inside the cone, because
the cone sits at low k — it reaches only `k ≤ 6.7`, 1.2% of the resolved `|k|` values —
while the weight's mass is at high k, where the cone has no area at all. So
`|w_eff − 1| ≤ 3 × 10⁻⁴` across the entire cone, the weight inside it *is* the L2
weight, and the cone is left exactly as starved:

| λ | 10⁻³ | 10⁻² (default) | 10⁻¹ | 1 | 10 | 10² | 10³ | 10⁶ |
|---|------|------|------|---|---|----|----|----|
| share ÷ L2 | 1.000 | **1.000** | 0.997 | 0.974 | 0.794 | 0.303 | 0.086 | 0.053 |

(`python scripts/06_sobolev_argument.py` — no trained model, no torch, ~1 min. The
objective is indistinguishable from L2 on the cone for every λ up to 1, i.e. 100× the
default. The 0.053 asymptote is the *pure* multiplicative reading — a different
objective, and what an earlier version of this file wrongly reported as the trained
one. A frequency-based fix cannot repair an amplitude-based blindness.)


## Reproduce

```bash
pip install -r requirements.txt
python scripts/00_validate_acoustics.py      # gate: known analytic answer
python scripts/04_komega_analysis.py         # the headline number — no model needed
python scripts/06_sobolev_argument.py        # why Sobolev cannot help — no model needed
```

`scripts/01_generate_data.py` regenerates the dataset if `data/` is missing. The
headline figure and the Stage-4 argument need neither torch nor a GPU.

Training (optional, Stage 3):
```bash
python scripts/02_train_surrogate.py --model fno
python scripts/03_compute_sources.py         # field vs source vs far-field error
```

## The validation gate

`scripts/00` radiates a co-rotating vortex pair and checks two facts with known
answers: the tone is at **twice** the orbital frequency, and the far-field phase
winds with **azimuthal mode number 2**. Both pass to three decimals. Nothing
downstream is trustworthy without this.

## Notes for anyone reading the code

Five things in here were bugs and are worth knowing about if you write your own
version:

- **Parseval normalisation.** FFT power needs dividing by `N_total²`, not
  `N_total`. Invisible on a log-scale plot; inflates every energy by ~10⁵.
- **Radial bin centring.** Bins must be centred *on* multiples of `dk`, not
  starting at them, or a half-bin bias tilts the apparent phase speed and
  corrupts the radiating fraction.
- **Masking a binned spectrum.** Fixing the bin centring is not enough. A cone
  condition evaluated at bin *centres* tests every mode in a bin against the wrong
  `k`: on a square grid one bin holds several distinct `|k|` (√21, 5 and √29 all
  land in the bin centred on 5). Near the cone edge that admits a slab of
  non-radiating modes — measured, it inflated the headline number by 2.8×. The
  metric therefore runs on a radially *resolved* grid (`KOmegaSpectrum.k_resolved`),
  not the binned one. `radiating_mask` is kept for plotting only.
- **Rotating-quadrupole directivity.** A co-rotating pair radiates a *spinning*
  pattern, so `|p'|` at the tone frequency is nearly axisymmetric. Testing the
  magnitude for four lobes returns a near-null for a perfectly correct answer;
  the structure is in the phase.
- **Selecting a frequency band by array index.** `[i for i in range(...)][:24]`
  picks the 24 *lowest* frequencies, which at M = 0.1 are all below `c₀·dk` — the
  lowest frequency that can radiate at all. On `data/turb2d.npz`, **0 of 24** were
  supersonic, and 91% of the source's ω > 0 power is down there on the convective
  ridge. Select by the radiation condition; see `greens.select_band`.

A sixth is a feature: the two Lighthill formulations (gradient form and double
divergence) are algebraically identical for solenoidal fields, so their
disagreement is a free **under-resolution detector** — 10⁻⁸ at n=96 with a
0.18 core, 10⁻¹⁴ at n=128.

## The open question

What objective *would* work? Not one built on `|k|` -- `scripts/06` shows the cone
is starved by where a `k`-based weight's mass is, not by its exponent, so the
weight has to be built from proximity to the sonic line in `(k, w)`. That needs a
time window to define `w`, so it is a multi-step objective, not a one-step one.
Deliberately left unimplemented in `src/models/losses.py` -- the diagnosis is worth
more intact than the fix.

## Layout

```
src/spectral.py            exact periodic derivatives, Helmholtz projection
src/acoustics/lighthill.py source term, two independent formulations
src/acoustics/komega.py    (k,w) spectra, sonic line, THE metric
src/acoustics/radiating.py loss weighting: L2 vs the trained Sobolev objective
src/acoustics/greens.py    2D Hankel Green's function, far field, radiating band
src/solvers/               2D NS (pseudospectral), vortex pair, IO
src/models/                FNO, UNet, losses, vanilla training
src/diagnostics/           spectra, spurious dilatation, the error triple
tests/                     42 tests, all passing
```
